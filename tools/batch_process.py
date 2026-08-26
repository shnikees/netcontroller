#!/usr/bin/env python3

# netcontroller -- live speech-to-text and callsign matching for ham radio nets
# Copyright (C) 2026 Michelle Michaels
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

"""Run a folder of recordings through the pipeline, so they can be mined.

`mine_roster.py` reads the session files the app writes, which means every
recording has to go through `app.py --file X --batch` first. With a scheduled
capture dropping three nets a day into a folder, doing that by hand stops being
reasonable within a week.

    python tools/batch_process.py --recordings ~/netcontroller-recordings
    python tools/batch_process.py --recordings ~/recordings --model small

Two things make it safe to run on a cron of its own, or just whenever:

- **It remembers what it has already done.** A manifest beside the recordings
  records which files have been processed and what came out, so re-running only
  picks up what is new. Re-processing everything nightly would waste hours and,
  worse, double-count every station in the mining that follows.
- **It refuses to touch a recording that is still being written.** A capture in
  progress looks exactly like a finished one until you notice the file is still
  growing; transcribing it would produce a truncated session that then looks
  like a complete net.

A failure on one recording is reported and skipped rather than stopping the
run, because the usual cause is a single corrupt file and the other nineteen
are still worth having.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = ".batch-processed.json"
STILL_WRITING_SECONDS = 120
"""How recently a file must have changed to be assumed still recording."""


def duration_of(path: Path) -> float:
    try:
        with wave.open(str(path)) as handle:
            return handle.getnframes() / handle.getframerate()
    except (wave.Error, OSError):
        return 0.0


def still_being_written(path: Path) -> bool:
    """A capture in progress, rather than a finished recording.

    Growth is the only trustworthy signal: a file that is not getting bigger is
    not being written, whatever its timestamps say. An earlier version treated a
    recent mtime as proof on its own, which fell over the first time a folder
    was copied between machines -- `scp` stamps every file with the current
    time, so all 32 recordings looked like they were mid-capture and the whole
    batch was skipped.

    A recent mtime is still worth something, as a hint that growth is worth
    waiting a little longer to rule out. On an old file the check returns
    immediately.
    """
    before = path.stat().st_size
    fresh = time.time() - path.stat().st_mtime < STILL_WRITING_SECONDS
    time.sleep(2.0 if fresh else 0.5)
    return path.stat().st_size != before


def sessions_in(directory: Path) -> set[str]:
    return {p.name for p in directory.glob("net-*.jsonl")}


def transcripts_dir(override: str | None, config_path: str) -> Path:
    """Where `app.py` will actually write, not where we happen to be standing.

    This was a bug worth spelling out. The directory used to be
    `Path(args.transcripts)` -- relative to *this tool's* working directory --
    while `app.py` runs with `cwd=REPO` and writes relative to that. Launched
    from anywhere else (a scheduled task, a cron entry, another folder) the two
    disagreed, the before/after diff came back empty, and every recording was
    filed with an empty session name. A later cleanup trusted those empty names
    and deleted the very transcripts it was meant to protect.

    So the path is resolved the way the app resolves it: from the config when
    there is one, and always anchored to the repository rather than to $PWD.
    """
    if override:
        chosen = Path(override)
        return chosen if chosen.is_absolute() else REPO / chosen

    configured = "transcripts"
    try:
        sys.path.insert(0, str(REPO))
        from config import load_config

        configured = load_config(config_path).transcripts.dir or configured
    except Exception:
        pass  # no config, or an unreadable one: the app's default applies

    chosen = Path(configured)
    return chosen if chosen.is_absolute() else REPO / chosen


def _newest_session_since(directory: Path, started: float) -> str:
    """Fallback when the before/after diff finds nothing.

    A resumed session appends to an existing file rather than creating one, so
    the diff is legitimately empty and the newest file touched during the run is
    the right answer.
    """
    if not directory.exists():
        return ""
    fresh = [
        p for p in directory.glob("net-*.jsonl") if p.stat().st_mtime >= started - 1
    ]
    return max(fresh, key=lambda p: p.stat().st_mtime).name if fresh else ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--recordings", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--roster", help="default: roster.csv, or an empty one")
    parser.add_argument("--model", default="base")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="recordings to transcribe at once. This, not --cpu-threads, is "
        "where the speed is: one transcription uses about one core however "
        "many threads it is given, because beam-search decoding is sequential. "
        "Measured on an 8-core desktop, 16 threads on one recording took 238s "
        "against 292s with the library default -- 18%%. Separate recordings are "
        "independent, so N at once is close to N times the throughput.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=2,
        help="threads per recording. Low on purpose: the gain past a couple is "
        "small, and --jobs spends the cores far better.",
    )
    parser.add_argument(
        "--transcripts", help="default: from the config, relative to the repo"
    )
    parser.add_argument(
        "--redo", action="store_true", help="process everything again from scratch"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    recordings = Path(args.recordings).expanduser()
    if not recordings.exists():
        raise SystemExit(f"No such folder: {recordings}")
    transcripts = transcripts_dir(args.transcripts, args.config)

    manifest_path = recordings / MANIFEST
    done: dict = {}
    if manifest_path.exists() and not args.redo:
        done = json.loads(manifest_path.read_text())

    # An empty roster is fine here: nothing will match, but raw_text is written
    # either way and that is all the mining reads.
    roster = Path(args.roster) if args.roster else Path("roster.csv")
    temporary_roster = None
    if not roster.exists():
        temporary_roster = recordings / ".empty-roster.csv"
        temporary_roster.write_text("callsign,name,position,sources\n", encoding="utf-8")
        roster = temporary_roster
        print(f"No roster given; using an empty one ({roster.name})")

    waiting = sorted(p for p in recordings.glob("*.wav") if p.name not in done)
    if not waiting:
        print(f"Nothing new in {recordings} ({len(done)} already processed)")
        return 0

    print(
        f"{len(waiting)} recording(s) to process, {len(done)} already done, "
        f"{args.jobs} job(s) x {args.cpu_threads} thread(s)\n"
    )
    processed = failed = skipped = 0
    lock = threading.Lock()

    def run_one(path: Path) -> str:
        length = duration_of(path)
        if still_being_written(path):
            return f"  {path.name:34} still recording, leaving it"
        if length < 60:
            return f"  {path.name:34} only {length:.0f}s, skipping"
        if args.dry_run:
            return f"  {path.name:34} would process ({length/60:.0f} min)"

        # Name the session after the recording. The caller knows which file it
        # is handing over, so there is no reason to work it out afterwards by
        # diffing a directory -- which was ambiguous with two replays running
        # and silently produced nothing when the paths disagreed.
        session = path.stem
        started = time.time()
        result = subprocess.run(
            [sys.executable, str(REPO / "app.py"), "--file", str(path),
             "--batch", "--model", args.model, "--roster", str(roster),
             "--cpu-threads", str(args.cpu_threads),
             "--session-name", session,
             "--config", args.config, "--no-log-file"],
            cwd=REPO, capture_output=True, text=True,
        )
        if result.returncode != 0:
            tail = (result.stderr or result.stdout).strip().splitlines()
            return f"  {path.name:34} FAILED: {tail[-1][:60] if tail else 'no output'}"

        written = transcripts / f"net-{session}.jsonl"
        if not written.exists():
            return f"  {path.name:34} FAILED: no session at {written.name}"

        with lock:
            done[path.name] = {
                "session": written.name,
                "minutes": round(length / 60, 1),
                "took_seconds": round(time.time() - started, 1),
            }
            manifest_path.write_text(json.dumps(done, indent=2), encoding="utf-8")
        return f"  {path.name:34} {length/60:5.1f} min {time.time()-started:5.0f}s -> {written.name}"

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for line in pool.map(run_one, waiting):
            print(line, flush=True)
            if "FAILED" in line:
                failed += 1
            elif "->" in line:
                processed += 1
            else:
                skipped += 1

    if temporary_roster and temporary_roster.exists():
        temporary_roster.unlink()

    print(f"\nprocessed {processed}, failed {failed}, skipped {skipped}")
    if processed:
        print(
            "\nNow mine them:\n"
            f"  python tools/mine_roster.py --transcripts {transcripts} "
            "--validate --out roster.draft.csv"
        )
    return 1 if failed and not processed else 0


if __name__ == "__main__":
    raise SystemExit(main())
