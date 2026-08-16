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

"""Propose a roster from nets already recorded.

The rest of this project treats the roster as ground truth. This is for the
case where there is no roster yet -- a net you do not run, or a new event --
and the only thing available is a pile of transcripts.

**Recurrence is the signal.** A real station identifies repeatedly, on night
after night. A mis-transcription is a one-off: the specific way Whisper mangled
one sentence on one evening does not reproduce, because it depends on exactly
how somebody spoke into exactly that much noise. So ranking callsign-shaped
tokens by *how many separate sessions* they appear in separates stations from
noise without anybody labelling anything.

Two rules keep it honest, and they are the same ones the matcher lives by:

- **Distinct sessions, never total mentions.** One garbled transmission
  repeating a fragment five times is still one piece of evidence. Counting
  mentions would promote it above a real station who checked in once a night.
- **It proposes, you approve.** The output is a draft for a human to read, not
  a roster to run with. Adopting a callsign the machine invented would let its
  own mistakes become the ground truth everything else is measured against,
  which is the one failure this design refuses everywhere else.

    python tools/mine_roster.py --transcripts transcripts/
    python tools/mine_roster.py --min-sessions 3 --out roster.draft.csv

Feed it sessions written by the app itself -- run each recording through
`app.py --file X --batch` first, with whatever roster you have (an empty one is
fine; unmatched lines still carry `raw_text`, which is all this reads).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from callsign_match import (  # noqa: E402
    CALLSIGN_RE,
    extract_candidates,
    normalize,
)


def sessions_from(directory: Path) -> list[tuple[str, list[str]]]:
    """Every session file, as (name, raw transcript lines)."""
    found = []
    for path in sorted(directory.glob("net-*.jsonl")):
        texts = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a truncated final line from a power cut
            if record.get("type") in ("entry", "correction") and record.get("raw_text"):
                texts.append(record["raw_text"])
        if texts:
            found.append((path.stem, texts))
    return found


def mine(sessions: list[tuple[str, list[str]]]) -> dict[str, dict]:
    """Callsign-shaped tokens, with the evidence for each."""
    seen: dict[str, dict] = {}
    for name, texts in sessions:
        for text in texts:
            # Strict shapes only. The loose pattern exists to salvage a
            # candidate the operator can confirm by ear, which is the opposite
            # of what is wanted here -- nobody is going to confirm a roster
            # entry that was never a callsign.
            for candidate in extract_candidates(normalize(text)):
                if not CALLSIGN_RE.fullmatch(candidate):
                    continue
                record = seen.setdefault(
                    candidate, {"sessions": set(), "mentions": 0, "example": text}
                )
                record["sessions"].add(name)
                record["mentions"] += 1
    return seen


# --------------------------------------------------------------------------
# Checking a callsign is real, and plausibly on this net
# --------------------------------------------------------------------------

NON_US = re.compile(r"^(V[AEOY]|C[FGIJKYZ]|X[EFJ]|[GM]|2[EMW]|E[IJ]|Z[LS]|JA|VK)")
"""Prefixes a US licence database cannot answer for. On a Puget Sound repeater
the VE7s across the border are regulars, not noise, and marking them invalid
because the FCC has never heard of them would be exactly the wrong call."""


def grid_to_latlon(grid: str) -> tuple[float, float] | None:
    """Centre of a Maidenhead square, enough for a rough distance."""
    grid = grid.strip().upper()
    if len(grid) < 4 or not grid[:2].isalpha() or not grid[2:4].isdigit():
        return None
    lon = (ord(grid[0]) - 65) * 20 - 180 + int(grid[2]) * 2
    lat = (ord(grid[1]) - 65) * 10 - 90 + int(grid[3])
    if len(grid) >= 6 and grid[4:6].isalpha():
        lon += (ord(grid[4]) - 65) * 5 / 60
        lat += (ord(grid[5]) - 65) * 2.5 / 60
        return lat + 1.25 / 60, lon + 2.5 / 60
    return lat + 0.5, lon + 1.0


def km_apart(a: str, b: str) -> float | None:
    import math

    first, second = grid_to_latlon(a), grid_to_latlon(b)
    if not first or not second:
        return None
    (lat1, lon1), (lat2, lon2) = first, second
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371 * math.asin(min(1.0, math.sqrt(h)))


def _state_of(line: str) -> str:
    """Just the state from "SEATTLE, WA 98177".

    Slicing the tail of the line looked equivalent and was not: some addresses
    end in a zip+4 or a unit number and printed "101-1725" where a state
    belonged.
    """
    found = re.search(r"\b([A-Z]{2})\b(?:\s+\d{5}(?:-\d{4})?)?\s*$", (line or "").strip())
    return found.group(1) if found else ""


def look_up(callsign: str, cache: dict, pause: float = 0.4) -> dict:
    """Licence status and rough location for one callsign.

    Deliberately keeps only status, state and grid square. The service also
    returns the licensee's name and street address, and none of that is needed
    to decide whether a callsign is real -- so none of it is written to disk.
    A cache of who lives where is not a thing this project should create.
    """
    if callsign in cache:
        return cache[callsign]
    if NON_US.match(callsign):
        result = {"status": "non-us", "state": "", "grid": ""}
    else:
        try:
            time.sleep(pause)  # a free service, queried politely
            with urllib.request.urlopen(
                f"https://callook.info/{callsign}/json", timeout=10
            ) as response:
                data = json.load(response)
            if data.get("status") == "VALID":
                result = {
                    "status": "valid",
                    "state": _state_of(data.get("address", {}).get("line2", "")),
                    "grid": data.get("location", {}).get("gridsquare", ""),
                }
            else:
                result = {"status": "not-issued", "state": "", "grid": ""}
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            result = {"status": "unchecked", "state": "", "grid": ""}
    cache[callsign] = result
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--transcripts", default="transcripts")
    parser.add_argument(
        "--min-sessions",
        type=int,
        default=2,
        help="how many separate nets a callsign must appear on (default 2)",
    )
    parser.add_argument("--out", help="write a draft roster CSV here")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="check each candidate against a licence database, and how far away it is",
    )
    parser.add_argument(
        "--near",
        default="",
        help="grid square to measure distance from (e.g. CN87). Shown for "
        "information only -- a linked net legitimately draws stations from "
        "everywhere, so distance never counts against a candidate. Default: "
        "wherever most of the valid candidates turn out to live",
    )
    parser.add_argument("--cache", default=".callsign-cache.json")
    args = parser.parse_args()

    directory = Path(args.transcripts)
    if not directory.exists():
        raise SystemExit(f"No transcripts at {directory}")

    sessions = sessions_from(directory)
    if not sessions:
        raise SystemExit(
            f"No session files in {directory}. Run each recording through "
            "`app.py --file <wav> --batch` first."
        )
    found = mine(sessions)
    ranked = sorted(
        found.items(), key=lambda kv: (-len(kv[1]["sessions"]), -kv[1]["mentions"], kv[0])
    )
    likely = [(c, r) for c, r in ranked if len(r["sessions"]) >= args.min_sessions]
    once = [(c, r) for c, r in ranked if len(r["sessions"]) < args.min_sessions]

    checked: dict[str, dict] = {}
    if args.validate:
        cache_path = Path(args.cache)
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        print(f"checking {len(ranked)} candidate(s) against the licence database...")
        for callsign, _ in ranked:
            checked[callsign] = look_up(callsign, cache)
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")

        home = args.near.upper()
        if not home:
            grids = [c["grid"][:4] for c in checked.values() if c.get("grid")]
            home = max(set(grids), key=grids.count) if grids else ""
        if home:
            print(f"measuring distance from {home}")

    print(f"\n{len(sessions)} session(s), {sum(len(t) for _, t in sessions)} transmissions\n")
    header = f"{'callsign':10} {'nets':>5} {'heard':>6}"
    if args.validate:
        header += f"  {'licence':11} {'where':>18}"
    print(header + "   first seen saying")
    print("-" * (100 if args.validate else 78))
    for callsign, record in likely:
        row = f"{callsign:10} {len(record['sessions']):5d} {record['mentions']:6d}"
        if args.validate:
            info = checked.get(callsign, {})
            where = info.get("state", "")
            distance = km_apart(home, info["grid"]) if info.get("grid") and home else None
            if distance is not None:
                where = f"{where} {distance:.0f}km"
            row += f"  {info.get('status', '?'):11} {where[:18]:>18}"
        print(row + f"   {record['example'].strip()[:34]}")

    if not likely:
        print("  (nothing yet appears on enough separate nets)")
    print(
        f"\n{len(once)} candidate(s) seen on fewer than {args.min_sessions} nets, "
        "which is what transcription noise looks like:"
    )
    print("  " + ", ".join(c for c, _ in once[:24]) + ("..." if len(once) > 24 else ""))

    if args.out:
        lines = [
            "# Draft roster mined from recorded nets -- NOT verified.",
            f"# {len(sessions)} sessions. Delete what you do not recognise, add names",
            "# and positions, then save as roster.csv.",
            "callsign,name,position,sources",
        ]
        lines += [f"{c},,," for c, _ in likely]
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nWrote {args.out} -- {len(likely)} entries. Read it before using it.")

    if args.validate:
        print(
            "\n`not-issued` is the signal worth acting on: nobody holds that callsign,\n"
            "so nobody said it, and it can be struck without listening.\n\n"
            "Distance is context, not evidence. A popular net collects check-ins from\n"
            "all over through EchoLink and AllStar, so a valid call a thousand miles\n"
            "away is an ordinary participant rather than a suspect. It is worth showing\n"
            "only because for an *event* roster you do want to know where somebody is.\n"
            "`non-us` means the database cannot answer -- the VE7s across the border are\n"
            "regulars on a Puget Sound repeater, not noise."
        )
    print(
        "\nRecurrence cannot tell you a callsign is spelled right, only that it\n"
        "keeps coming back. A station Whisper mangles the same way every time\n"
        "will look just as convincing as a real one, so check these by ear or\n"
        "against a licence database before trusting them.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
