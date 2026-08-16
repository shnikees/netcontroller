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

"""Tests for the VAD state machine.

webrtcvad's own speech/non-speech judgement is not what is under test here --
the clip boundaries are. So these tests drive the segmenter with a scripted
speech pattern ("S" = voiced frame, "." = silence) and assert where the clips
land. Tuning the real thresholds against actual net audio is a separate,
manual step: see `python app.py --file recording.wav` and the README.
"""

from __future__ import annotations

import numpy as np
import pytest

from vad_segmenter import FRAME_BYTES_PER_MS, VadSegmenter


class ScriptedVad:
    """Stand-in for webrtcvad.Vad that replays a fixed speech pattern."""

    def __init__(self, pattern: str) -> None:
        self.pattern = pattern
        self.index = 0

    def is_speech(self, frame: bytes, rate: int) -> bool:
        speech = self.pattern[self.index] == "S"
        self.index += 1
        return speech


def run(pattern: str, **kwargs) -> list:
    """Segment one frame per pattern character; frames are silence-filled."""
    segmenter = VadSegmenter(**kwargs)
    segmenter._vad = ScriptedVad(pattern)
    frame = b"\x00\x00" * (segmenter.frame_ms * FRAME_BYTES_PER_MS // 2)
    return list(segmenter.segment(frame for _ in pattern))


# 30 ms frames throughout: 10 frames = 300 ms.
TUNING = dict(
    frame_ms=30, preroll_ms=300, silence_ms=300, min_clip_ms=200, trigger_ratio=0.7
)


def test_single_transmission_becomes_one_clip() -> None:
    clips = run("." * 10 + "S" * 30 + "." * 20, **TUNING)
    assert len(clips) == 1


def test_two_transmissions_split_on_silence() -> None:
    clips = run("." * 10 + "S" * 20 + "." * 20 + "S" * 20 + "." * 20, **TUNING)
    assert len(clips) == 2


def test_short_pause_does_not_split_a_transmission() -> None:
    # A 150 ms gap -- the sort of pause between "whiskey" and "six" -- must
    # stay inside one clip when silence_ms is 300.
    clips = run("." * 10 + "S" * 20 + "." * 5 + "S" * 20 + "." * 20, **TUNING)
    assert len(clips) == 1


def test_squelch_tail_below_min_length_is_dropped() -> None:
    # Long enough to trigger, too short to keep.
    clips = run(
        "." * 10 + "S" * 10 + "." * 20,
        frame_ms=30,
        preroll_ms=300,
        silence_ms=300,
        min_clip_ms=800,
        trigger_ratio=0.7,
    )
    assert clips == []


def test_isolated_noise_burst_does_not_trigger() -> None:
    # One voiced frame in a 300 ms window is under the 0.7 trigger ratio.
    clips = run("." * 10 + "S" + "." * 20, **TUNING)
    assert clips == []


def test_clip_includes_preroll_and_excludes_hangover() -> None:
    clips = run("." * 10 + "S" * 20 + "." * 20, **TUNING)
    clip = clips[0]
    # 20 voiced frames + up to a 10-frame preroll window, minus the trailing
    # silence that closed the clip.
    assert 600 <= clip.duration_ms <= 900
    assert clip.duration_ms == len(clip.audio) * 1000 // 16_000


def test_max_clip_length_forces_a_cut() -> None:
    clips = run(
        "." * 10 + "S" * 100,
        frame_ms=30,
        preroll_ms=300,
        silence_ms=300,
        min_clip_ms=200,
        max_clip_ms=900,
        trigger_ratio=0.7,
    )
    assert len(clips) >= 3
    assert all(c.duration_ms <= 900 for c in clips)


def test_open_transmission_is_flushed_at_end_of_stream() -> None:
    # Stream ends mid-transmission (recording ran out) -- do not lose the clip.
    clips = run("." * 10 + "S" * 30, **TUNING)
    assert len(clips) == 1


def test_audio_is_float32_in_unit_range() -> None:
    clip = run("." * 10 + "S" * 30 + "." * 20, **TUNING)[0]
    assert clip.audio.dtype.name == "float32"
    assert clip.audio.min() >= -1.0 and clip.audio.max() <= 1.0


def test_timestamps_advance_with_the_stream() -> None:
    clips = run("." * 10 + "S" * 20 + "." * 20 + "S" * 20 + "." * 20, **TUNING)
    assert clips[0].start_offset_ms < clips[1].start_offset_ms
    # Second clip starts after the first transmission and its trailing silence.
    assert clips[1].start_offset_ms >= 1_000


def test_rejects_invalid_frame_size() -> None:
    with pytest.raises(ValueError):
        VadSegmenter(frame_ms=25)

# --------------------------------------------------------------------------
# The noise gate
#
# webrtcvad decides speech from spectral shape, which falls apart on a feed
# that never goes quiet -- a streamed repeater, an open squelch, a scanner.
# Measured against a real 75-minute repeater stream it called 74% of the
# recording speech at every aggressiveness from 0 to 3, so the knob was no help.
#
# The bed below is deliberately *speech-shaped* rather than white noise. White
# noise is the easy case -- webrtcvad rejects it on its own, and an earlier
# version of these tests passed for that reason while proving nothing. What
# defeats it is quiet audio with the spectrum of a voice, which is exactly what
# an AGC'd receiver puts between overs.
# --------------------------------------------------------------------------

UNGATED = dict(gate_margin=1.0, gate_min_floor=10**9)


def _voice(ms: int, amplitude: int, rate: int = 16_000) -> np.ndarray:
    t = np.arange(int(rate * ms / 1000)) / rate
    wave = (
        np.sin(2 * np.pi * 180 * t)
        + 0.5 * np.sin(2 * np.pi * 400 * t)
        + 0.3 * np.sin(2 * np.pi * 900 * t)
    )
    return (wave / 1.8 * amplitude).astype(np.int16)


def _frames(audio: np.ndarray, frame_ms: int = 30) -> list[bytes]:
    step = int(16_000 * frame_ms / 1000) * 2
    raw = audio.astype("<i2").tobytes()
    return [raw[i : i + step] for i in range(0, len(raw) - step + 1, step)]


def _noisy_net() -> np.ndarray:
    """Two overs separated by a bed that is quiet but not silent."""
    return np.concatenate([
        _voice(2000, 1400),                        # bed, to establish a floor
        _voice(2000, 14000), _voice(1500, 1400),   # over, gap, over
        _voice(2000, 14000), _voice(1000, 1400),
    ])


def test_the_bed_defeats_webrtcvad_on_its_own() -> None:
    """The premise. If this ever fails, the tests below prove nothing."""
    ungated = VadSegmenter(**UNGATED)
    clips = list(ungated.segment(_frames(_noisy_net())))
    assert len(clips) == 1, "the bed should run the two overs together"


def test_the_gate_recovers_the_gap_between_two_overs() -> None:
    clips = list(VadSegmenter().segment(_frames(_noisy_net())))
    assert len(clips) == 2, f"expected two transmissions, got {len(clips)}"


def test_a_squelched_feed_is_untouched() -> None:
    """The property that lets this default to on.

    Real silence puts the floor near zero, the threshold lands near zero, and
    every frame passes exactly as it did before the gate existed.
    """
    audio = np.concatenate([
        np.zeros(16_000, dtype=np.int16), _voice(1200, 12000),
        np.zeros(16_000, dtype=np.int16), _voice(1200, 12000),
        np.zeros(8_000, dtype=np.int16),
    ])
    gated = list(VadSegmenter().segment(_frames(audio)))
    ungated = list(VadSegmenter(**UNGATED).segment(_frames(audio)))
    assert [c.duration_ms for c in gated] == [c.duration_ms for c in ungated]
    assert len(gated) == 2


def test_the_floor_does_not_chase_a_long_over() -> None:
    """The bug that sank the first design.

    A social net runs two-minute overs. A decaying-average floor climbs toward
    the speaker's own level and eventually gates out the person talking; a low
    percentile cannot, because the pauses between words keep exposing the real
    floor.
    """
    bursty = np.concatenate(
        [_voice(120, 14000) if i % 5 else _voice(120, 1400) for i in range(1000)]
    )  # 120 s of speech with word-length gaps
    seg = VadSegmenter()
    clips = list(seg.segment(_frames(np.concatenate([_voice(2000, 1400), bursty]))))
    assert seg.noise_floor < 4000, f"floor chased the speaker to {seg.noise_floor:.0f}"
    assert clips, "the gate closed on the only speaker"


def test_a_quiet_feed_never_engages_the_gate() -> None:
    seg = VadSegmenter()
    list(seg.segment(_frames(np.concatenate([_voice(2000, 60), _voice(1000, 9000)]))))
    assert seg.noise_floor < seg.gate_min_floor
