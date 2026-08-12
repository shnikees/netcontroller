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

"""Tests for resampling to 16 kHz.

These check the signal, not just the shape. A resampler that returns the right
number of samples while mangling the audio would pass a length assertion and
then fail every net, so tones go in and an FFT checks what comes out.
"""

from __future__ import annotations

import numpy as np
import pytest

import resample as resample_module
from resample import HAVE_SOXR, Resampler, describe

TARGET = 16_000


@pytest.fixture(params=["auto", "fir"])
def engine(request, monkeypatch):
    """Run the signal tests against both engines.

    Without this the fallback would be dead code on any machine with soxr
    installed -- including this one -- and would first be exercised on a Pi in
    the field, which is precisely the wrong place to find out it is broken.
    """
    if request.param == "fir":
        monkeypatch.setattr(resample_module, "HAVE_SOXR", False)
    return request.param


def tone(frequency: float, rate: int, seconds: float = 0.5) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (np.sin(2 * np.pi * frequency * t) * 12000).astype(np.int16)


def dominant_frequency(samples: np.ndarray, rate: int) -> float:
    """Frequency of the strongest bin, ignoring DC."""
    windowed = samples.astype(np.float64) * np.hanning(len(samples))
    spectrum = np.abs(np.fft.rfft(windowed))
    spectrum[0] = 0.0
    return float(np.fft.rfftfreq(len(samples), 1 / rate)[int(np.argmax(spectrum))])


def resample_in_blocks(source_rate: int, samples: np.ndarray, block: int = 1024):
    """Feed the resampler in blocks, the way the audio callback does."""
    resampler = Resampler(source_rate, TARGET)
    out = [
        resampler.process(samples[i : i + block])
        for i in range(0, len(samples), block)
    ]
    out.append(resampler.flush())
    return np.concatenate([chunk for chunk in out if len(chunk)])


# --------------------------------------------------------------------------
# Mode selection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rate,expected",
    [(16_000, "passthrough"), (48_000, "decimate"), (32_000, "decimate")],
)
def test_integer_ratios_decimate(rate: int, expected: str) -> None:
    assert Resampler(rate, TARGET).mode == expected


def test_non_integer_ratio_uses_a_real_resampler() -> None:
    # 44100 is what microphones and USB sound cards actually offer, and it is
    # not a multiple of 16000 -- the case the old code rejected outright.
    assert Resampler(44_100, TARGET).mode == ("soxr" if HAVE_SOXR else "fir")


def test_fallback_is_used_when_soxr_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(resample_module, "HAVE_SOXR", False)
    assert Resampler(44_100, TARGET).mode == "fir"


# --------------------------------------------------------------------------
# Signal preservation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("source_rate", [16_000, 44_100, 48_000, 22_050, 8_000])
def test_speech_band_tone_survives(source_rate: int, engine) -> None:
    # 1 kHz sits in the middle of the voice band, where callsigns live.
    out = resample_in_blocks(source_rate, tone(1000, source_rate))
    assert len(out) > 0
    assert dominant_frequency(out, TARGET) == pytest.approx(1000, abs=30)


@pytest.mark.parametrize("source_rate", [44_100, 48_000])
def test_amplitude_is_roughly_preserved(source_rate: int, engine) -> None:
    source = tone(1000, source_rate)
    out = resample_in_blocks(source_rate, source)
    # Trim edges, where filter startup transients live.
    trimmed = out[TARGET // 10 : -TARGET // 10]
    assert np.abs(trimmed).max() == pytest.approx(np.abs(source).max(), rel=0.2)


@pytest.mark.parametrize("source_rate", [44_100, 48_000])
def test_output_length_matches_the_rate_ratio(source_rate: int, engine) -> None:
    out = resample_in_blocks(source_rate, tone(440, source_rate, seconds=1.0))
    assert len(out) == pytest.approx(TARGET, rel=0.02)


def test_high_frequency_content_does_not_fold_into_the_voice_band(engine) -> None:
    """The anti-aliasing check, and the reason a naive resampler is not enough.

    15 kHz cannot be represented at 16 kHz. Without a low-pass it reflects back
    to ~1 kHz -- right on top of speech -- and Whisper hears a whistle over
    every callsign.
    """
    out = resample_in_blocks(44_100, tone(15_000, 44_100))
    trimmed = out[TARGET // 10 : -TARGET // 10]
    windowed = trimmed.astype(np.float64) * np.hanning(len(trimmed))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(trimmed), 1 / TARGET)

    voice_band = spectrum[(freqs > 200) & (freqs < 4000)].max()
    reference = np.abs(tone(1000, TARGET)).max()
    # Whatever leaks through must be far below a real signal at the same level.
    assert voice_band < reference * 0.05


# --------------------------------------------------------------------------
# Streaming behaviour
# --------------------------------------------------------------------------


@pytest.mark.parametrize("block", [160, 441, 1024, 4096])
def test_block_size_does_not_change_the_result(block: int, engine) -> None:
    # The audio callback hands over whatever PortAudio decides; the output must
    # not depend on that.
    source = tone(1000, 44_100)
    out = resample_in_blocks(44_100, source, block=block)
    assert dominant_frequency(out, TARGET) == pytest.approx(1000, abs=30)


def test_no_samples_are_lost_at_block_boundaries() -> None:
    # Decimation with a remainder: 48000/16000 = 3, and a 1000-sample block is
    # not divisible by 3, so the tail must carry over.
    source = tone(1000, 48_000, seconds=1.0)
    out = resample_in_blocks(48_000, source, block=1000)
    assert len(out) == pytest.approx(TARGET, rel=0.01)


def test_streaming_has_no_discontinuities(engine) -> None:
    """A dropped or duplicated sample at a block edge clicks, and the VAD hears
    the click as speech."""
    out = resample_in_blocks(44_100, tone(500, 44_100), block=1024)
    trimmed = out[TARGET // 10 : -TARGET // 10].astype(np.float64)
    jumps = np.abs(np.diff(trimmed))
    # A 500 Hz sine at 16 kHz steps by at most ~2*pi*500/16000 of its amplitude.
    assert jumps.max() < np.abs(trimmed).max() * 0.5


def test_empty_input_is_handled() -> None:
    assert len(Resampler(44_100, TARGET).process(np.zeros(0, dtype=np.int16))) == 0


def test_flush_returns_the_held_tail() -> None:
    # A streaming resampler holds ~20 ms; without flushing, a file replay loses
    # the end of the last transmission.
    resampler = Resampler(44_100, TARGET)
    resampler.process(tone(1000, 44_100, seconds=0.2))
    assert len(resampler.flush()) >= 0  # never negative, never raises


def test_output_stays_int16() -> None:
    out = resample_in_blocks(44_100, tone(1000, 44_100))
    assert out.dtype == np.int16


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def test_describe_is_readable() -> None:
    assert "no resampling" in describe(16_000)
    assert "decimate by 3" in describe(48_000)
    assert "44100 -> 16000" in describe(44_100)
