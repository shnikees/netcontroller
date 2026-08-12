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

"""Tests for calibrating thresholds from collected data.

The important cases are the refusals. A calibration that confidently returns a
number from data with no signal in it is worse than one that says "not yet" --
the operator would apply it and never know it was noise.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from calibrate import calibrate_escalation, calibrate_voice, load_entries
from voice_id import Profile, VoiceProfiles


def entry(confidence: float, matched: bool, **kwargs) -> dict:
    return {"type": "entry", "confidence": confidence, "matched": matched, **kwargs}


# --------------------------------------------------------------------------
# Escalation
# --------------------------------------------------------------------------


def test_the_threshold_lands_between_the_two_groups() -> None:
    entries = [entry(0.9, True) for _ in range(20)]
    entries += [entry(0.3, False) for _ in range(10)]

    result = calibrate_escalation(entries)
    assert result.usable
    assert 0.3 < result.threshold < 0.9


def test_the_reported_threshold_behaves_as_measured() -> None:
    """The rounded number in the config must do what the calibration said.

    Picking the edge of the winning range put the threshold exactly on top of
    a cluster of lines, where rounding for display flipped which side of it
    they fell on.
    """
    entries = [entry(0.9, True) for _ in range(20)]
    entries += [entry(0.3, False) for _ in range(10)]

    result = calibrate_escalation(entries)
    caught = sum(1 for c in result.unmatched if c < result.threshold)
    assert caught == len(result.unmatched)
    disturbed = sum(1 for c in result.matched if c < result.threshold)
    assert disturbed == 0


def test_too_little_data_refuses_to_answer() -> None:
    result = calibrate_escalation([entry(0.9, True), entry(0.2, False)])
    assert not result.usable
    assert "not enough data" in result.note


def test_confidence_that_does_not_separate_is_reported_as_useless() -> None:
    # Matched and unmatched lines at the same confidence: thresholding on it
    # would escalate at random, and saying so beats returning a number.
    rng = np.random.default_rng(0)
    entries = [entry(float(c), True) for c in rng.uniform(0.4, 0.8, 120)]
    entries += [entry(float(c), False) for c in rng.uniform(0.4, 0.8, 80)]

    result = calibrate_escalation(entries)
    assert not result.usable
    assert "does not separate" in result.note


def test_the_cost_of_the_threshold_is_reported() -> None:
    entries = [entry(0.9, True) for _ in range(15)]
    entries += [entry(0.2, False) for _ in range(5)]
    result = calibrate_escalation(entries)
    # A quarter of the lines are the unmatched ones; escalating them is the
    # cost the operator is agreeing to.
    assert 0.1 < result.escalate_fraction < 0.5


def test_corrected_lines_do_not_count_as_clean_matches() -> None:
    # A corrected line was wrong when the machine produced it, so counting its
    # confidence among the good ones would bias the threshold down.
    entries = [entry(0.9, True) for _ in range(10)]
    entries += [entry(0.1, True, corrected=True) for _ in range(10)]
    entries += [entry(0.3, False) for _ in range(5)]
    result = calibrate_escalation(entries)
    assert all(c == 0.9 for c in result.matched)


# --------------------------------------------------------------------------
# Voice
# --------------------------------------------------------------------------


def vectors(base: np.ndarray, count: int, spread: float, seed: int) -> list:
    """Clips of one speaker: the same voice, wobbling a little."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        v = base + rng.normal(0, spread, base.shape)
        out.append((v / np.linalg.norm(v)).astype(np.float32))
    return out


def profiles_with(spread: float = 0.05, seed: int = 0) -> VoiceProfiles:
    rng = np.random.default_rng(seed)
    profiles = VoiceProfiles()
    for index, callsign in enumerate(("W6ABC", "K7XYZ", "N5DEF")):
        base = rng.normal(0, 1, 24)
        base = base / np.linalg.norm(base)
        samples = vectors(base, 5, spread, seed + index)
        profiles.profiles[callsign] = Profile(
            callsign=callsign, centroid=samples[0], count=len(samples), samples=samples
        )
    return profiles


def test_the_threshold_separates_speakers() -> None:
    result = calibrate_voice(profiles_with(spread=0.05))
    assert result.usable
    # Above the different-station scores, below the same-station ones.
    assert max(result.different) <= result.threshold + 0.01
    assert result.recall > 0.5


def test_false_accepts_are_held_near_zero() -> None:
    # The asymmetry that matters: a false accept puts a station in the log who
    # never spoke, while a false reject just means no suggestion appears.
    result = calibrate_voice(profiles_with(spread=0.05))
    assert result.false_accepts <= 0.02


def test_no_enrolled_voices_refuses_to_answer() -> None:
    result = calibrate_voice(VoiceProfiles())
    assert not result.usable
    assert "not enough enrolled voices" in result.note


def test_one_clip_per_station_is_not_enough() -> None:
    profiles = VoiceProfiles()
    profiles.profiles["W6ABC"] = Profile(
        callsign="W6ABC", centroid=np.ones(24), count=1, samples=[np.ones(24)]
    )
    assert not calibrate_voice(profiles).usable


def test_indistinguishable_voices_are_flagged_not_hidden() -> None:
    # Speakers whose clips vary more than they differ: the honest answer is
    # "expect few suggestions", not a threshold that looks authoritative.
    result = calibrate_voice(profiles_with(spread=1.5, seed=7))
    assert result.recall < 0.5
    assert result.note


# --------------------------------------------------------------------------
# Reading sessions back
# --------------------------------------------------------------------------


def test_entries_are_read_from_session_files(tmp_path) -> None:
    path = tmp_path / "net-1.jsonl"
    path.write_text(
        json.dumps({"type": "session", "started_at": "2026-04-01T19:00:00"}) + "\n"
        + json.dumps({"type": "entry", "id": 1, "confidence": 0.9, "matched": True}) + "\n"
        + json.dumps({"type": "entry", "id": 2, "confidence": 0.3, "matched": False}) + "\n",
        encoding="utf-8",
    )
    entries = load_entries(tmp_path)
    assert len(entries) == 2


def test_a_correction_replaces_the_line_it_corrects(tmp_path) -> None:
    path = tmp_path / "net-1.jsonl"
    path.write_text(
        json.dumps({"type": "entry", "id": 1, "confidence": 0.3, "matched": False}) + "\n"
        + json.dumps({"type": "correction", "id": 1, "confidence": 0.3,
                      "matched": True, "corrected": True}) + "\n",
        encoding="utf-8",
    )
    entries = load_entries(tmp_path)
    assert len(entries) == 1
    assert entries[0]["corrected"] is True


def test_lines_from_several_nets_are_pooled(tmp_path) -> None:
    for name in ("net-1.jsonl", "net-2.jsonl"):
        (tmp_path / name).write_text(
            json.dumps({"type": "entry", "id": 1, "confidence": 0.8, "matched": True}) + "\n",
            encoding="utf-8",
        )
    # Same id in two sessions must not collapse into one line.
    assert len(load_entries(tmp_path)) == 2


def test_a_truncated_line_is_skipped(tmp_path) -> None:
    path = tmp_path / "net-1.jsonl"
    path.write_text(
        json.dumps({"type": "entry", "id": 1, "confidence": 0.8, "matched": True}) + "\n"
        + '{"type": "entry", "id": 2, "conf',
        encoding="utf-8",
    )
    assert len(load_entries(tmp_path)) == 1


def test_a_missing_directory_is_not_an_error(tmp_path) -> None:
    assert load_entries(tmp_path / "nope") == []
