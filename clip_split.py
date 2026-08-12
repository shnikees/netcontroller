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

"""Split one clip into several log lines when it caught two transmissions.

On a fast net, stations key up inside the VAD's silence window and land in a
single clip. Left alone, only the first callsign becomes a log line and the
other station is simply missing from the net report.

The hard part is not finding the second callsign -- the matcher already does.
It is telling these two apart:

    "W6ABC checking in"  ...pause...  "K7XYZ also checking in"   -> two stations
    "W6ABC here, I have traffic for K7XYZ"                       -> one station

The transcripts look the same. What separates them is the **pause**: two
transmissions have dead air between them where one operator stops and another
starts keying, and a sentence naming somebody else does not. So splitting is
decided on the word timings, never on the text alone.

The bias is deliberate and matches the rest of the app: when the evidence is
weak, keep it as one line. An over-eager split invents a check-in that never
happened, which nobody will catch reading the log -- the same reason the
matcher would rather say "unmatched" than guess a callsign.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class Segment:
    """One transmission's worth of a clip that turned out to hold several."""

    text: str
    start_offset_ms: int
    """Milliseconds into the clip where this transmission began."""
    duration_ms: int


def split_transmissions(
    text: str,
    words: list,
    occurrences: list,
    clip_duration_ms: int,
    *,
    min_gap_ms: int = 500,
    min_segment_ms: int = 400,
) -> list[Segment]:
    """Split `text` where two stations are separated by real dead air.

    words: Word objects from the STT worker (text, start, end, offset).
    occurrences: MatchResults carrying start/end offsets into `text`.

    Returns one Segment per transmission -- a single-element list when there is
    no good reason to believe this was more than one.
    """
    whole = [Segment(text=text, start_offset_ms=0, duration_ms=clip_duration_ms)]
    if len(occurrences) < 2 or not words:
        return whole

    cuts: list[tuple[int, float]] = []
    for previous, current in zip(occurrences, occurrences[1:]):
        cut = _find_pause(words, previous.end, current.start, min_gap_ms)
        if cut is None:
            # No dead air between these two callsigns, so the second one was
            # spoken *about*, not *by* -- the "traffic for K7XYZ" case.
            continue
        cuts.append(cut)

    if not cuts:
        return whole

    segments: list[Segment] = []
    boundaries = [(0, 0.0)] + cuts + [(len(text), clip_duration_ms / 1000)]
    for (start_char, start_s), (end_char, end_s) in zip(boundaries, boundaries[1:]):
        piece = text[start_char:end_char].strip()
        duration_ms = int(round((end_s - start_s) * 1000))
        if not piece or duration_ms < min_segment_ms:
            # A sliver is more likely a mis-timing than a transmission; folding
            # it back is safer than logging a fragment as a check-in.
            return whole
        segments.append(
            Segment(
                text=piece,
                start_offset_ms=int(round(start_s * 1000)),
                duration_ms=duration_ms,
            )
        )

    log.info(
        "Clip held %d transmissions (%s); splitting on the pauses between them",
        len(segments),
        ", ".join(o.callsign for o in occurrences),
    )
    return segments


def _find_pause(
    words: list, after_offset: int, before_offset: int, min_gap_ms: int
) -> tuple[int, float] | None:
    """Largest inter-word silence between two character offsets.

    Returns (character offset to cut at, time in seconds), or None when nothing
    in that stretch is a long enough pause to be a change of transmitter.
    """
    # Inclusive of the word at `before_offset`: that is the first word of the
    # second callsign, and the pause we are looking for is the one immediately
    # in front of it. Excluding it measures every gap except the only one that
    # matters.
    between = [w for w in words if after_offset <= w.offset <= before_offset]
    if len(between) < 2:
        # Nothing between the two callsigns to measure a gap in; the second may
        # follow immediately, which is not evidence of a second transmission.
        return None

    best_gap = 0.0
    best: tuple[int, float] | None = None
    for previous, current in zip(between, between[1:]):
        gap = current.start - previous.end
        if gap > best_gap:
            best_gap = gap
            # Cut at the start of the word after the pause, and time the new
            # transmission from where its audio actually begins.
            best = (current.offset, current.start)

    if best is None or best_gap * 1000 < min_gap_ms:
        return None
    return best
