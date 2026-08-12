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

"""Work out two thresholds from data the app has already collected.

Unlike the VAD and split thresholds, these two need no recording session and no
sweep. Both are calibrations on things running a net produces anyway:

**escalation.min_confidence** -- how unsure a line has to be before it is worth
re-transcribing with a bigger model. Every transcript line carries a confidence
and whether it matched the roster, so the threshold is wherever a line stops
being likely to be right.

**voice.min_similarity** -- how close two recordings of one operator look. The
enrolled voice samples give both distributions directly: same-station pairs and
different-station pairs. The threshold goes where they separate.

Every function here returns the evidence alongside the number. A threshold with
no separation behind it is a threshold nobody should apply, and the caller has
to be able to see that.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from voice_id import VoiceProfiles, similarity


# --------------------------------------------------------------------------
# Escalation
# --------------------------------------------------------------------------


@dataclass
class ConfidenceCalibration:
    """Where 'unsure' begins, measured from lines the app already logged."""

    threshold: float
    matched: list[float] = field(default_factory=list)
    unmatched: list[float] = field(default_factory=list)
    escalate_fraction: float = 0.0
    """Share of all lines that would be re-transcribed at this threshold."""
    separation: float = 0.0
    """Median confidence of matched lines minus that of unmatched ones."""
    note: str = ""

    @property
    def usable(self) -> bool:
        return self.note == ""


def calibrate_escalation(
    entries: list[dict], *, floor: float = 0.2, ceiling: float = 0.9
) -> ConfidenceCalibration:
    """Pick the confidence below which a line is worth a second pass.

    The signal is simple: unmatched lines are the ones the pipeline got wrong,
    so if they sit lower in confidence than matched lines, confidence is worth
    thresholding on. The threshold goes between the two groups -- specifically
    at the point that separates them best, which is what a decision boundary is.
    """
    matched = [
        float(e.get("confidence", 0.0))
        for e in entries
        if e.get("matched") and not e.get("corrected")
    ]
    unmatched = [
        float(e.get("confidence", 0.0)) for e in entries if not e.get("matched")
    ]

    if len(matched) < 5 or len(unmatched) < 3:
        return ConfidenceCalibration(
            threshold=0.55,
            matched=matched,
            unmatched=unmatched,
            note=(
                f"not enough data yet ({len(matched)} matched, "
                f"{len(unmatched)} unmatched); run another net"
            ),
        )

    separation = float(np.median(matched) - np.median(unmatched))
    if separation <= 0.05:
        return ConfidenceCalibration(
            threshold=0.55,
            matched=matched,
            unmatched=unmatched,
            separation=separation,
            note=(
                "confidence does not separate good lines from bad on this net, "
                "so thresholding on it would escalate at random -- escalate on "
                "unmatched lines alone (on_unmatched: true, min_confidence: 0)"
            ),
        )

    # Sweep candidate thresholds: most unmatched lines caught, fewest matched
    # lines disturbed.
    scored: list[tuple[float, float]] = []
    for candidate in np.arange(floor, ceiling, 0.01):
        caught = sum(1 for c in unmatched if c < candidate) / len(unmatched)
        disturbed = sum(1 for c in matched if c < candidate) / len(matched)
        scored.append((float(candidate), caught - disturbed))

    best_score = max(score for _, score in scored)
    winners = [value for value, score in scored if score >= best_score - 1e-9]
    # The *middle* of the winning range, not the first value in it. Picking the
    # edge puts the threshold exactly on top of a cluster of lines, where
    # rounding it for display changes which side of it they fall on -- the
    # calibration would then measure one thing and the config do another.
    best_threshold = winners[len(winners) // 2]

    everything = matched + unmatched
    fraction = sum(1 for c in everything if c < best_threshold) / len(everything)
    return ConfidenceCalibration(
        threshold=round(best_threshold, 2),
        matched=matched,
        unmatched=unmatched,
        escalate_fraction=fraction,
        separation=separation,
    )


# --------------------------------------------------------------------------
# Voice
# --------------------------------------------------------------------------


@dataclass
class VoiceCalibration:
    """How alike two recordings of one operator are, on this net's audio."""

    threshold: float
    margin: float
    same: list[float] = field(default_factory=list)
    """Similarity between two clips of the same station."""
    different: list[float] = field(default_factory=list)
    """Similarity between clips of different stations."""
    false_accepts: float = 0.0
    """Share of different-station pairs that would pass the threshold."""
    recall: float = 0.0
    """Share of same-station pairs that would be recognised."""
    note: str = ""

    @property
    def usable(self) -> bool:
        return self.note == ""


def calibrate_voice(
    profiles: VoiceProfiles, *, target_false_accepts: float = 0.01
) -> VoiceCalibration:
    """Choose a similarity threshold from enrolled samples.

    Set by the cost of being wrong, not by the middle of the two distributions.
    A false accept puts a station in the log who never spoke; a false reject
    just means no suggestion appears. So the threshold is placed to keep false
    accepts near zero and recall is whatever that leaves.
    """
    same: list[float] = []
    different: list[float] = []

    stations = [p for p in profiles.profiles.values() if len(p.samples) >= 2]
    for profile in stations:
        for i, first in enumerate(profile.samples):
            for second in profile.samples[i + 1 :]:
                same.append(similarity(first, second))

    every = [p for p in profiles.profiles.values() if p.samples]
    for i, one in enumerate(every):
        for other in every[i + 1 :]:
            for a in one.samples:
                for b in other.samples:
                    different.append(similarity(a, b))

    if len(same) < 5 or len(different) < 5:
        return VoiceCalibration(
            threshold=0.82,
            margin=0.06,
            same=same,
            different=different,
            note=(
                f"not enough enrolled voices yet ({len(stations)} station(s) "
                "with two or more clips); run another net with voice enabled"
            ),
        )

    # The threshold that holds false accepts at the target. Sorting the
    # different-station scores and reading off the tail is exactly that.
    ordered = sorted(different, reverse=True)
    index = min(len(ordered) - 1, int(len(ordered) * target_false_accepts))
    threshold = float(ordered[index]) + 0.005

    recall = sum(1 for s in same if s >= threshold) / len(same)
    false_accepts = sum(1 for d in different if d >= threshold) / len(different)

    # The margin keeps two similar voices from being a coin flip, so scale it
    # to how tightly the different-station scores are packed.
    margin = round(max(0.03, float(np.std(different))), 3)

    note = ""
    if recall < 0.15:
        note = (
            f"only {recall:.0%} of same-station pairs would be recognised at "
            "this threshold -- the embedding is not separating your operators "
            "well, so expect few suggestions rather than wrong ones"
        )
    return VoiceCalibration(
        threshold=round(min(0.98, max(0.5, threshold)), 3),
        margin=margin,
        same=same,
        different=different,
        false_accepts=false_accepts,
        recall=recall,
        note=note,
    )


# --------------------------------------------------------------------------
# Reading what the app left behind
# --------------------------------------------------------------------------


def load_entries(directory: str | Path) -> list[dict]:
    """Every transcript line from every session in a directory.

    Corrections are folded onto the line they correct, so a line an operator
    fixed counts once, in its final state.
    """
    directory = Path(directory)
    if not directory.exists():
        return []

    by_id: dict[tuple[str, int], dict] = {}
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a truncated final line from a power cut
            if record.get("type") not in ("entry", "correction"):
                continue
            key = (path.name, int(record.get("id", 0)))
            by_id[key] = record  # a later correction replaces the entry
    return list(by_id.values())
