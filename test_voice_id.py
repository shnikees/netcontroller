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

from voice_id import EnrolmentAudio, VoiceProfiles, embed, similarity

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


# --------------------------------------------------------------------------
# Keeping the audio profiles were built from
#
# The point of this is a future embedder swap: vectors from two models mean
# nothing to each other, so without the clips, changing the model throws away
# every profile. It cannot be added retroactively, which is why it defaults on.
# --------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path) -> EnrolmentAudio:
    return EnrolmentAudio(tmp_path / "voice_audio", per_station=3, max_seconds=2.0)


def test_enrolling_keeps_the_clip(store: EnrolmentAudio, tmp_path) -> None:
    profiles = VoiceProfiles(audio=store, min_enrolments=1)
    profiles.enrol("W6ABC", alice(1))

    assert store.stations() == ["W6ABC"]
    assert len(store.clips("W6ABC")) == 1


def test_kept_audio_round_trips(store: EnrolmentAudio) -> None:
    original = alice(1)[: 16_000 * 2]
    store.save("W6ABC", original)

    recovered = store.clips("W6ABC")[0]
    assert len(recovered) == len(original)
    # 16-bit quantisation, so close rather than identical.
    assert float(np.abs(recovered - original).max()) < 0.001


def test_only_the_most_recent_clips_are_kept(store: EnrolmentAudio) -> None:
    # A voice heard last week is more use than one from six months ago.
    for seed in range(6):
        store.save("W6ABC", alice(seed))
    assert len(store.clips("W6ABC")) == 3


def test_long_clips_are_trimmed(store: EnrolmentAudio) -> None:
    # This lives on an SD card; a rag-chew must not cost 40 MB.
    store.save("W6ABC", voice(210, [(800, 1.0)], seconds=30))
    assert len(store.clips("W6ABC")[0]) <= 16_000 * 2


def test_a_callsign_cannot_escape_the_directory(store: EnrolmentAudio, tmp_path) -> None:
    # Never trust a roster file with a path.
    store.save("../../etc/W6ABC", alice(1))
    assert not (tmp_path / "etc").exists()
    assert store.stations()


def test_retention_off_writes_nothing(tmp_path) -> None:
    profiles = VoiceProfiles(audio=None, min_enrolments=1)
    assert profiles.enrol("W6ABC", alice(1))
    assert not (tmp_path / "voice_audio").exists()


def test_an_unwritable_directory_does_not_stop_enrolment(tmp_path) -> None:
    blocker = tmp_path / "in-the-way"
    blocker.write_text("not a directory")
    profiles = VoiceProfiles(
        audio=EnrolmentAudio(blocker / "voice_audio"), min_enrolments=1
    )
    # The profile is still learned; only the keeping failed.
    assert profiles.enrol("W6ABC", alice(1))
    assert "W6ABC" in profiles.profiles


# --------------------------------------------------------------------------
# Rebuilding -- what the retention is for
# --------------------------------------------------------------------------


def test_rebuilding_reproduces_the_profiles(store: EnrolmentAudio) -> None:
    profiles = VoiceProfiles(audio=store, min_enrolments=1)
    for seed in (1, 2):
        profiles.enrol("W6ABC", alice(seed))
        profiles.enrol("K7XYZ", bob(seed))

    before = {c: p.centroid.copy() for c, p in profiles.profiles.items()}
    profiles.profiles.clear()
    stations, clips = profiles.rebuild()

    assert stations == 2 and clips == 4
    for callsign, centroid in before.items():
        assert similarity(profiles.profiles[callsign].centroid, centroid) > 0.999


def test_rebuilding_still_identifies_the_right_station(store: EnrolmentAudio) -> None:
    profiles = VoiceProfiles(audio=store, min_enrolments=1)
    for seed in (1, 2, 3):
        profiles.enrol("W6ABC", alice(seed))
        profiles.enrol("K7XYZ", bob(seed))

    profiles.profiles.clear()
    profiles.rebuild()

    suggestion = profiles.identify(alice(42))
    assert suggestion is not None and suggestion.callsign == "W6ABC"


def test_rebuilding_survives_an_embedder_change(store: EnrolmentAudio, monkeypatch) -> None:
    """The scenario this whole feature exists for.

    A new embedder returns vectors of a different size and meaning. Every
    stored profile is void -- but the audio is not, so the profiles come back
    in one pass instead of weeks of re-enrolment.
    """
    profiles = VoiceProfiles(audio=store, min_enrolments=1)
    for seed in (1, 2):
        profiles.enrol("W6ABC", alice(seed))
        profiles.enrol("K7XYZ", bob(seed))
    assert profiles.profiles["W6ABC"].centroid.shape == (24,)

    import voice_id

    def different_embedder(audio, rate=16_000):
        # Stands in for ECAPA: a different size, different meaning, same job.
        base = np.asarray(audio, dtype=np.float32)
        if len(base) < 16_000 // 2:
            return None
        vector = np.array(
            [float(np.std(c)) for c in np.array_split(base, 192)], dtype=np.float32
        )
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 1e-9 else None

    monkeypatch.setattr(voice_id, "embed", different_embedder)
    stations, clips = profiles.rebuild()

    assert stations == 2 and clips == 4
    assert profiles.profiles["W6ABC"].centroid.shape == (192,)


def test_rebuilding_without_kept_audio_does_nothing() -> None:
    profiles = VoiceProfiles(min_enrolments=1)
    assert profiles.rebuild() == (0, 0)


def test_deleting_a_stations_clips_removes_it_on_rebuild(store: EnrolmentAudio) -> None:
    # How a profile poisoned by a wrong match gets fixed.
    profiles = VoiceProfiles(audio=store, min_enrolments=1)
    profiles.enrol("W6ABC", alice(1))
    profiles.enrol("K7XYZ", bob(1))

    for path in (store.directory / "K7XYZ").glob("*.wav"):
        path.unlink()
    profiles.rebuild()

    assert "W6ABC" in profiles.profiles
    assert "K7XYZ" not in profiles.profiles
