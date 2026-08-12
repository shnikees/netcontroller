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

"""Tests for conditioning a clip before Whisper sees it."""

from __future__ import annotations

import numpy as np
import pytest

from audio_prep import high_pass, peak_normalize, prepare

RATE = 16_000


def tone(freq: float, seconds: float = 1.0, amplitude: float = 0.2) -> np.ndarray:
    t = np.arange(int(RATE * seconds)) / RATE
    return (np.sin(2 * np.pi * freq * t) * amplitude).astype(np.float32)


def test_quiet_speech_is_brought_up() -> None:
    # The case that matters: a radio's line output into a mic input, far too
    # quiet for a model trained on normalised audio.
    quiet = tone(1000, amplitude=0.02)
    assert np.abs(prepare(quiet)).max() > 0.9


def test_loud_audio_is_brought_down_rather_than_clipped() -> None:
    loud = tone(1000, amplitude=0.99)
    out = prepare(loud)
    assert np.abs(out).max() <= 1.0
    assert np.abs(out).max() == pytest.approx(0.95, abs=0.02)


def test_hiss_is_not_amplified_into_signal() -> None:
    # Amplifying dead air would manufacture something for the VAD and the model
    # to find, which is worse than leaving it quiet.
    rng = np.random.default_rng(0)
    hiss = (rng.normal(0, 0.001, RATE)).astype(np.float32)
    assert np.abs(prepare(hiss)).max() < 0.1


def test_dc_offset_is_removed() -> None:
    offset = tone(1000) + 0.3
    assert abs(float(np.mean(prepare(offset)))) < 0.01


def test_the_scipy_free_fallback_also_removes_rumble(monkeypatch) -> None:
    """The Pi case, and the reason CI runs a job without scipy at all.

    A fallback that is never exercised is a fallback that is broken -- this one
    was, until the minimal-dependency job said so.
    """
    import audio_prep

    monkeypatch.setattr(audio_prep, "HAVE_SCIPY", False)
    rumble = tone(40, amplitude=0.5)
    speech = tone(1000, amplitude=0.5)
    assert np.abs(audio_prep.high_pass(rumble)).max() < np.abs(rumble).max() * 0.5
    assert np.abs(audio_prep.high_pass(speech)).max() > np.abs(speech).max() * 0.8


def test_rumble_is_attenuated_and_speech_is_not() -> None:
    rumble = tone(40, amplitude=0.5)      # below the voice band
    speech = tone(1000, amplitude=0.5)    # in it
    assert np.abs(high_pass(rumble)).max() < np.abs(rumble).max() * 0.5
    assert np.abs(high_pass(speech)).max() > np.abs(speech).max() * 0.8


def test_the_waveform_is_otherwise_unchanged() -> None:
    # Normalising must scale, not distort: the frequency has to survive.
    out = prepare(tone(1000, amplitude=0.05))
    spectrum = np.abs(np.fft.rfft(out.astype(np.float64) * np.hanning(len(out))))
    peak_hz = np.fft.rfftfreq(len(out), 1 / RATE)[int(np.argmax(spectrum))]
    assert peak_hz == pytest.approx(1000, abs=5)


def test_empty_and_tiny_clips_are_safe() -> None:
    assert prepare(np.zeros(0, dtype=np.float32)).size == 0
    assert prepare(np.zeros(3, dtype=np.float32)).size == 3


def test_conditioning_can_be_turned_off() -> None:
    quiet = tone(1000, amplitude=0.02)
    untouched = prepare(quiet, highpass=False, normalize=False)
    assert np.allclose(untouched, quiet)


def test_output_stays_in_range_and_float32() -> None:
    out = prepare(tone(1000, amplitude=0.9) + 0.2)
    assert out.dtype == np.float32
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_normalising_is_idempotent() -> None:
    once = peak_normalize(tone(1000, amplitude=0.1))
    assert np.allclose(peak_normalize(once), once)
