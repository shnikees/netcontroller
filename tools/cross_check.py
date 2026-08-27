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

"""Find fabricated callsigns by asking a second engine what it heard.

Biasing an engine towards the roster is the biggest lever on callsign recovery
and also the biggest source of invented callsigns, and from inside a single
transcript the two are indistinguishable: "KJ7RAB, KJ7JXM." is either a station
passing traffic or the prompt read back. `hallucination.py` catches the cases
where several callsigns arrive with no speech around them, which is most of
them, but it cannot catch one plausible fabrication in one plausible sentence.

This can, because it uses evidence from outside the transcript. An echo of the
prompt has no acoustic support, so an engine that was never told the roster will
not go near it. Run the same clips through an unbiased engine and ask, per clip,
whether it heard anything the matcher recognises as the callsign the biased
engine claimed.

    python tools/cross_check.py --suspect wh-prompt --against pk-txt wh-plain

Directories hold one `.txt` per clip, named identically across directories --
which is what `whisper-cli -otxt` and `parakeet-cli -otxt` produce. Anything
missing from a comparison directory is treated as no support, not skipped.

Measured on 1,525 clips from six PSRG nets on 2026-08-26: of 876 callsigns
`base` Whisper reported with the roster in its prompt, 110 were corroborated and
**766 were not**. See docs/HARDWARE.md.

Two things this is not:

- **Not proof of fabrication.** Two engines can miss the same faint callsign, so
  an unsupported callsign is unconfirmed rather than disproven. It is one-sided
  evidence, and the direction that matters: 87% unsupported says the biasing is
  mostly echo whatever the exact per-clip verdicts are.
- **Not a live filter.** It costs a second full transcription pass, so it
  belongs in analysis of recordings, not in the path of a running net.

Support is judged by `CallsignMatcher`, deliberately, rather than by a string
compare or a regex. Engines write callsigns differently -- Parakeet spells "Kilo
Juliet 7, Romeo Alpha Bravo" where Whisper writes "KJ7RAB" -- and recovering
that is precisely what the normalizer is for. A first version of this check used
a regex for already-collapsed callsigns and reported 4% support where the
matcher finds 13%, because it could not read the phonetic spellings at all.
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from callsign_match import CallsignMatcher, RosterEntry  # noqa: E402

CALLSIGN_SHAPE = re.compile(r"\b([A-Z]{1,2}\d[A-Z]{1,3})\b")
"""US callsign shape, used only to *harvest* candidates from the suspect
transcripts. Matching them in the comparison transcripts is the matcher's job."""

_matchers: dict[str, CallsignMatcher] = {}


def _supports(text: str, callsign: str) -> bool:
    """Whether the matcher can find `callsign` in `text`.

    A one-entry roster per callsign, so the question asked is exactly "could
    this transcript have been this station" with no interference from the
    matcher's ambiguity refusal, which exists to arbitrate between roster
    entries and has nothing to do here.
    """
    if callsign not in _matchers:
        _matchers[callsign] = CallsignMatcher(roster=[RosterEntry(callsign)])
    return any(result.matched for result in _matchers[callsign].match_all(text))


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except OSError:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--suspect", required=True,
        help="directory of transcripts to check, from the biased engine",
    )
    parser.add_argument(
        "--against", required=True, nargs="+",
        help="one or more directories of transcripts from unbiased engine(s)",
    )
    parser.add_argument(
        "--examples", type=int, default=6,
        help="unsupported clips to print in full, for reading (default 6)",
    )
    args = parser.parse_args()

    suspect = Path(args.suspect)
    if not suspect.is_dir():
        raise SystemExit(f"Not a directory: {suspect}")
    others = [Path(d) for d in args.against]
    for directory in others:
        if not directory.is_dir():
            raise SystemExit(f"Not a directory: {directory}")

    supported = unsupported = 0
    unsupported_counts: collections.Counter[str] = collections.Counter()
    examples: list[tuple[str, str, str, list[str]]] = []

    for path in sorted(suspect.glob("*.txt")):
        text = _read(path)
        claimed = set(CALLSIGN_SHAPE.findall(text.upper()))
        if not claimed:
            continue
        elsewhere = [_read(directory / path.name) for directory in others]
        combined = "\n".join(elsewhere)
        for callsign in sorted(claimed):
            if _supports(combined, callsign):
                supported += 1
                continue
            unsupported += 1
            unsupported_counts[callsign] += 1
            if len(examples) < args.examples:
                examples.append((path.name, callsign, text.strip(),
                                 [t.strip() for t in elsewhere]))

    total = supported + unsupported
    if not total:
        print(f"No callsign-shaped strings in {suspect}")
        return 0

    print(f"{suspect.name}: {total} callsign extraction(s) across "
          f"{len(list(suspect.glob('*.txt')))} clip(s)")
    print(f"  corroborated by {', '.join(d.name for d in others)}: "
          f"{supported:5d}  ({100 * supported / total:.0f}%)")
    print(f"  no support from any of them:{'':17}{unsupported:5d}  "
          f"({100 * unsupported / total:.0f}%)")

    if unsupported_counts:
        print("\nmost frequently unsupported:")
        for callsign, count in unsupported_counts.most_common(10):
            print(f"  {callsign:8} {count:5d}")

    if examples:
        print("\nunsupported clips, for reading:")
        for name, callsign, claim, elsewhere in examples:
            print(f"\n  {name} -- claims {callsign}")
            print(f"    {suspect.name:12}: {claim[:76]!r}")
            for directory, text in zip(others, elsewhere):
                print(f"    {directory.name:12}: {text[:76]!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
