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

"""Resampling to 16 kHz, for input devices that will not give it directly.

A loopback sink can be told to run at 16 or 48 kHz. A microphone or a USB sound
card taking line-in from a radio's speaker jack usually cannot: they run at
44.1 kHz, which is not an integer multiple of 16 kHz, so decimation does not
apply and real resampling is needed.

Three paths, picked automatically:

- Same rate: passthrough.
- Integer ratios (48000 -> 16000) decimate with a boxcar average. Cheap, and
  the content above 8 kHz on a narrowband voice channel is noise anyway.
- Everything else (44100 -> 16000) uses `soxr`, falling back to a windowed-sinc
  low-pass plus linear interpolation if it is not installed.

All three are streaming-safe: filter state carries across calls, so block
boundaries do not click and the VAD does not hear the click as speech.

Anti-aliasing matters here rather than being a nicety. Fold 44.1 kHz content
down onto the voice band and it sits on top of the speech, which costs
transcription accuracy on exactly the fast, run-together delivery that is
already the hardest thing to get right.
"""

from __future__ import annotations

import logging
import numpy as np

log = logging.getLogger(__name__)

try:  # pragma: no cover - depends on what is installed
    import soxr

    HAVE_SOXR = True
except ImportError:  # pragma: no cover
    soxr = None
    HAVE_SOXR = False


class Resampler:
    """Streaming resampler from `source_rate` to `target_rate`.

    Feed it int16 blocks of any length; get back int16 at the target rate. The
    number of samples out per call varies by a sample or two, which is why the
    caller re-chunks into fixed VAD frames rather than assuming a fixed size.
    """

    def __init__(self, source_rate: int, target_rate: int) -> None:
        self.source_rate = int(source_rate)
        self.target_rate = int(target_rate)
        self._mode = self._pick_mode()
        self._soxr = None
        self._tail = np.zeros(0, dtype=np.float32)

        if self._mode == "decimate":
            self._ratio = self.source_rate // self.target_rate
        elif self._mode == "soxr":
            self._soxr = soxr.ResampleStream(
                self.source_rate, self.target_rate, 1, dtype="float32", quality="HQ"
            )
        elif self._mode == "fir":
            # Low-pass at the target Nyquist *before* interpolating, so nothing
            # above 8 kHz folds down onto the voice band.
            self._taps = _lowpass(self.target_rate * 0.45 / self.source_rate)
            self._history = np.zeros(len(self._taps) - 1, dtype=np.float32)
            self._carry = np.zeros(0, dtype=np.float32)
            self._pos = 0.0
            self._step = self.source_rate / self.target_rate

    def _pick_mode(self) -> str:
        if self.source_rate == self.target_rate:
            return "passthrough"
        if self.source_rate % self.target_rate == 0:
            return "decimate"
        if HAVE_SOXR:
            return "soxr"
        return "fir"

    @property
    def mode(self) -> str:
        return self._mode

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Resample one block of int16 mono samples."""
        if self._mode == "passthrough":
            return samples

        if self._mode == "decimate":
            # Carry the remainder so no sample is dropped at block boundaries.
            data = np.concatenate([self._tail, samples.astype(np.float32)])
            usable = len(data) - (len(data) % self._ratio)
            self._tail = data[usable:]
            if usable == 0:
                return np.zeros(0, dtype=np.int16)
            return _to_int16(data[:usable].reshape(-1, self._ratio).mean(axis=1))

        if self._mode == "soxr":
            out = self._soxr.resample_chunk(samples.astype(np.float32) / 32768.0)
            return _to_int16(np.asarray(out).reshape(-1) * 32768.0)

        return self._fir_interpolate(samples)

    def flush(self) -> np.ndarray:
        """Emit whatever the resampler is still holding, at end of stream.

        A streaming resampler keeps a filter's worth of samples in hand -- about
        20 ms for soxr. Live capture never notices; a file replay would lose the
        tail of the last transmission without this.
        """
        if self._mode == "soxr":
            out = self._soxr.resample_chunk(np.zeros(0, dtype=np.float32), last=True)
            return _to_int16(np.asarray(out).reshape(-1) * 32768.0)
        return np.zeros(0, dtype=np.int16)

    def _fir_interpolate(self, samples: np.ndarray) -> np.ndarray:
        """Fallback path: FIR low-pass, then linear interpolation.

        Used only when soxr is missing. Cheaper than a full polyphase filter
        and good enough for narrowband voice, because the low-pass -- not the
        interpolation -- is what keeps aliasing out.
        """
        data = np.concatenate([self._history, samples.astype(np.float32)])
        if len(data) < len(self._taps):
            self._history = data
            return np.zeros(0, dtype=np.int16)
        filtered = np.convolve(data, self._taps, mode="valid")
        self._history = data[-(len(self._taps) - 1) :]

        buffer = np.concatenate([self._carry, filtered])
        # Linear interpolation needs the sample after the one it lands on, so
        # the last input sample is always held back for the next block.
        usable = len(buffer) - 1
        if usable < 1 or self._pos > usable:
            self._carry = buffer
            return np.zeros(0, dtype=np.int16)

        count = int((usable - self._pos) / self._step) + 1
        positions = self._pos + self._step * np.arange(count)
        # Each output sample interpolates between buffer[left] and buffer[left+1],
        # so a position landing exactly on the last sample has no partner and
        # must wait for the next block. Bites when upsampling (step < 1), where
        # positions land on integers exactly.
        positions = positions[positions < usable]
        if len(positions) == 0:
            self._carry = buffer
            return np.zeros(0, dtype=np.int16)
        count = len(positions)
        left = positions.astype(np.int64)
        frac = positions - left
        out = buffer[left] * (1.0 - frac) + buffer[left + 1] * frac

        next_pos = self._pos + count * self._step
        consumed = int(next_pos)
        self._carry = buffer[consumed:]
        self._pos = next_pos - consumed
        return _to_int16(out)


def _lowpass(cutoff: float, length: int = 127) -> np.ndarray:
    """Windowed-sinc low-pass. `cutoff` is a fraction of the source rate.

    127 taps with a Blackman window puts the stopband far enough down that
    15 kHz hiss does not reappear as a whistle over somebody's callsign, and is
    still cheap enough to run per audio block on a Pi.
    """
    n = np.arange(length) - (length - 1) / 2
    taps = 2 * cutoff * np.sinc(2 * cutoff * n)
    taps *= np.blackman(length)
    return (taps / taps.sum()).astype(np.float32)


def _to_int16(samples: np.ndarray) -> np.ndarray:
    return np.clip(samples, -32768, 32767).astype(np.int16)


def describe(source_rate: int, target_rate: int = 16_000) -> str:
    """One-line summary of what will happen, for the startup log."""
    if source_rate == target_rate:
        return f"{source_rate} Hz (no resampling)"
    if source_rate % target_rate == 0:
        return f"{source_rate} -> {target_rate} Hz (decimate by {source_rate // target_rate})"
    engine = "soxr" if HAVE_SOXR else "FIR + interpolation"
    return f"{source_rate} -> {target_rate} Hz ({engine})"
