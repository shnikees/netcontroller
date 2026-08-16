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
import subprocess
import sys
import time
import wave
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

    Checked two ways because either alone can be fooled: a file written slowly
    may look untouched between glances, and a file whose header is still a
    placeholder reports a length that does not match its size.
    """
    if time.time() - path.stat().st_mtime < STILL_WRITING_SECONDS:
        return True
    before = path.stat().st_size
    time.sleep(1.5)
    return path.stat().st_size != before


def sessions_in(directory: Path) -> set[str]:
    return {p.name for p in directory.glob("net-*.jsonl")}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--recordings", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--roster", help="default: roster.csv, or an empty one")
    parser.add_argument("--model", default="base")
    parser.add_argument("--transcripts", default="transcripts")
    parser.add_argument(
        "--redo", action="store_true", help="process everything again from scratch"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    recordings = Path(args.recordings).expanduser()
    if not recordings.exists():
        raise SystemExit(f"No such folder: {recordings}")
    transcripts = Path(args.transcripts)

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

    print(f"{len(waiting)} recording(s) to process, {len(done)} already done\n")
    processed = failed = skipped = 0
    for path in waiting:
        length = duration_of(path)
        if still_being_written(path):
            print(f"  {path.name:34} still recording, leaving it")
            skipped += 1
            continue
        if length < 60:
            print(f"  {path.name:34} only {length:.0f}s, skipping")
            skipped += 1
            continue
        if args.dry_run:
            print(f"  {path.name:34} would process ({length/60:.0f} min)")
            continue

        before = sessions_in(transcripts) if transcripts.exists() else set()
        started = time.time()
        print(f"  {path.name:34} {length/60:5.1f} min ... ", end="", flush=True)
        result = subprocess.run(
            [sys.executable, str(REPO / "app.py"), "--file", str(path),
             "--batch", "--model", args.model, "--roster", str(roster),
             "--config", args.config, "--no-log-file"],
            cwd=REPO, capture_output=True, text=True,
        )
        if result.returncode != 0:
            tail = (result.stderr or result.stdout).strip().splitlines()
            print(f"FAILED: {tail[-1][:60] if tail else 'no output'}")
            failed += 1
            continue

        new = (sessions_in(transcripts) - before) if transcripts.exists() else set()
        done[path.name] = {
            "session": sorted(new)[0] if new else "",
            "minutes": round(length / 60, 1),
            "took_seconds": round(time.time() - started, 1),
        }
        manifest_path.write_text(json.dumps(done, indent=2), encoding="utf-8")
        print(f"{time.time() - started:5.0f}s -> {done[path.name]['session'] or '(no session)'}")
        processed += 1

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
