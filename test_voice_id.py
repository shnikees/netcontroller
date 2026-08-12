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

"""Tests for identifying a station by voice.

Synthetic speakers: a glottal buzz shaped by fixed formants, which is what
actually distinguishes one voice from another. Real operators over FM are
harder than this, so these tests prove the mechanism rather than the accuracy
-- the threshold is the one number that has to be set against a real net.

The refusals matter more than the identifications. A suggestion is offered to
an operator; a wrong one that gets clicked through becomes a phantom check-in.
"""

from __future__ import annotations

import numpy as np
import pytest

from voice_id import VoiceProfiles, embed, similarity

RATE = 16_000


def voice(pitch: float, formants, seconds: float = 2.0, seed: int = 0) -> np.ndarray:
    """A crude synthetic speaker."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(RATE * seconds)) / RATE
    buzz = sum(np.sin(2 * np.pi * pitch * k * t) / k for k in range(1, 25))
    out = np.zeros_like(buzz)
    for freq, gain in formants:
        out += gain * np.sin(2 * np.pi * freq * t) * buzz
    out *= 0.6 + 0.4 * np.sin(2 * np.pi * 3.5 * t + rng.uniform(0, 3))
    out += rng.normal(0, 0.01, len(out))
    return (out / np.abs(out).max() * 0.8).astype(np.float32)


def alice(seed: int = 1) -> np.ndarray:
    return voice(210, [(800, 1.0), (1600, 0.7), (2900, 0.4)], seed=seed)


def bob(seed: int = 2) -> np.ndarray:
    return voice(110, [(500, 1.0), (1100, 0.8), (2400, 0.3)], seed=seed)


def carol(seed: int = 3) -> np.ndarray:
    return voice(175, [(650, 1.0), (2000, 0.6), (3300, 0.5)], seed=seed)


@pytest.fixture
def profiles() -> VoiceProfiles:
    return VoiceProfiles(min_enrolments=2)


def enrol_all(profiles: VoiceProfiles, callsign: str, maker, seeds=(1, 2, 3)) -> None:
    for seed in seeds:
        assert profiles.enrol(callsign, maker(seed))


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------


def test_the_same_speaker_scores_higher_than_a_different_one() -> None:
    assert similarity(embed(alice(1)), embed(alice(9))) > similarity(
        embed(alice(1)), embed(bob(9))
    )


def test_a_clip_too_short_to_characterise_returns_nothing() -> None:
    # Half a second of "roger" says nothing about who is speaking.
    assert embed(np.zeros(int(RATE * 0.3), dtype=np.float32)) is None


def test_silence_produces_no_embedding() -> None:
    assert embed(np.zeros(RATE * 2, dtype=np.float32)) is None


def test_embeddings_are_unit_length() -> None:
    assert float(np.linalg.norm(embed(alice()))) == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------
# Identifying
# --------------------------------------------------------------------------


def test_a_known_voice_is_suggested(profiles: VoiceProfiles) -> None:
    enrol_all(profiles, "W6ABC", alice)
    enrol_all(profiles, "K7XYZ", bob)

    suggestion = profiles.identify(alice(42))
    assert suggestion is not None
    assert suggestion.callsign == "W6ABC"


def test_an_unknown_voice_is_not_suggested(profiles: VoiceProfiles) -> None:
    # A visiting station nobody has enrolled must produce nothing, not the
    # nearest thing on file.
    enrol_all(profiles, "W6ABC", alice)
    enrol_all(profiles, "K7XYZ", bob)
    profiles.min_similarity = 0.99

    assert profiles.identify(carol(42)) is None


def test_nothing_is_suggested_before_enough_enrolments(
    profiles: VoiceProfiles,
) -> None:
    # One clip is one moment of one net.
    profiles.enrol("W6ABC", alice(1))
    assert profiles.identify(alice(2)) is None

    profiles.enrol("W6ABC", alice(3))
    assert profiles.identify(alice(2)) is not None


def test_two_similar_voices_produce_no_suggestion(profiles: VoiceProfiles) -> None:
    """A coin flip between two operators is worse than saying nothing."""
    enrol_all(profiles, "W6ABC", alice)
    # A near-twin of Alice, enrolled under a different callsign.
    for seed in (1, 2, 3):
        profiles.enrol("K7XYZ", voice(212, [(810, 1.0), (1610, 0.7), (2910, 0.4)], seed=seed))
    profiles.margin = 0.5  # nothing can clear this

    assert profiles.identify(alice(42)) is None


def test_an_empty_profile_store_suggests_nothing(profiles: VoiceProfiles) -> None:
    assert profiles.identify(alice()) is None


def test_a_clip_too_short_is_never_identified(profiles: VoiceProfiles) -> None:
    enrol_all(profiles, "W6ABC", alice)
    assert profiles.identify(np.zeros(int(RATE * 0.2), dtype=np.float32)) is None


# --------------------------------------------------------------------------
# Learning and persistence
# --------------------------------------------------------------------------


def test_enrolling_averages_over_clips(profiles: VoiceProfiles) -> None:
    enrol_all(profiles, "W6ABC", alice, seeds=(1, 2, 3, 4))
    assert profiles.profiles["W6ABC"].count == 4


def test_profiles_survive_a_restart(tmp_path) -> None:
    path = tmp_path / "voices.json"
    first = VoiceProfiles(path=path, min_enrolments=2)
    enrol_all(first, "W6ABC", alice)
    assert first.save()

    # Next week's net starts already knowing this voice.
    second = VoiceProfiles(path=path, min_enrolments=2)
    assert second.load() == 1
    suggestion = second.identify(alice(42))
    assert suggestion is not None and suggestion.callsign == "W6ABC"


def test_a_missing_or_corrupt_profile_file_is_not_fatal(tmp_path) -> None:
    assert VoiceProfiles(path=tmp_path / "nope.json").load() == 0

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert VoiceProfiles(path=broken).load() == 0


def test_forgetting_a_station_removes_its_voice(profiles: VoiceProfiles) -> None:
    # For when a profile was built from mis-matched lines.
    enrol_all(profiles, "W6ABC", alice)
    profiles.forget("W6ABC")
    assert profiles.identify(alice(42)) is None


def test_known_lists_only_usable_profiles(profiles: VoiceProfiles) -> None:
    enrol_all(profiles, "W6ABC", alice)
    profiles.enrol("K7XYZ", bob(1))  # only one clip
    assert profiles.known == ["W6ABC"]
