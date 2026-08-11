"""Tests for the VAD state machine.

webrtcvad's own speech/non-speech judgement is not what is under test here --
the clip boundaries are. So these tests drive the segmenter with a scripted
speech pattern ("S" = voiced frame, "." = silence) and assert where the clips
land. Tuning the real thresholds against actual net audio is a separate,
manual step: see `python app.py --file recording.wav` and the README.
"""

from __future__ import annotations

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
