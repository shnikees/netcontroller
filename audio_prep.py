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

"""Condition a clip just before Whisper sees it.

The cheapest accuracy available: under a millisecond per clip against a
transcription measured in seconds, and it matters most on exactly the inputs
that are hardest to control. A loopback sink hands over a consistent level; a radio's
speaker output into a USB sound card does not, and neither does a microphone
across the room. Whisper was trained on normalised audio and transcribes quiet
material noticeably worse.

Two steps, in order:

1. **High-pass at 80 Hz.** Mains hum, motor rumble and DC offset carry no speech
   and do nothing but eat headroom the normaliser is about to spend.
2. **Peak normalise**, with a floor so a clip of pure hiss is not amplified into
   something that sounds like speech to the model.

Deliberately *not* here: noise reduction. Spectral gating helps a clean office
recording and does unpredictable things to a squelched FM tail, and a
transcript ruined by artefacts is worse than one ruined by hiss -- at least
hiss is what the operator heard.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - depends on what is installed
    from scipy.signal import butter, lfilter

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    HAVE_SCIPY = False

TARGET_PEAK = 0.95
"""Leave a little headroom; clipping is its own kind of distortion."""

SILENCE_FLOOR = 0.01
"""Below this peak the clip is hiss or a dead carrier. Amplifying it would
manufacture signal that was never there."""

MAX_GAIN = 50.0
"""Ceiling on the boost. A line output into a mic input can land 30 dB down and
still be perfectly good speech, so the ceiling is generous; SILENCE_FLOOR, not
this, is what stops dead air being amplified into something model-shaped."""


def prepare(audio: np.ndarray, *, highpass: bool = True, normalize: bool = True) -> np.ndarray:
    """Return a conditioned copy of a float32 clip in [-1, 1]."""
    if audio.size == 0:
        return audio
    out = audio.astype(np.float32, copy=True)
    if highpass:
        out = high_pass(out)
    if normalize:
        out = peak_normalize(out)
    return out


def high_pass(audio: np.ndarray, cutoff_hz: float = 80.0, rate: int = 16_000) -> np.ndarray:
    """Remove rumble, hum and DC offset below the voice band.

    An IIR recursion cannot be vectorised in numpy, and a Python loop over
    80,000 samples costs more than everything else here put together -- so this
    uses scipy's C implementation when it is available and falls back to plain
    DC removal when it is not. The fallback is weaker but still worth having:
    a DC offset eats the headroom the normaliser is about to spend.
    """
    if audio.size < 8:
        return audio
    if not HAVE_SCIPY:
        return (audio - float(np.mean(audio))).astype(np.float32)
    b, a = butter(2, cutoff_hz / (rate / 2), btype="highpass")
    return lfilter(b, a, audio).astype(np.float32)


def peak_normalize(
    audio: np.ndarray,
    target: float = TARGET_PEAK,
    floor: float = SILENCE_FLOOR,
    max_gain: float = MAX_GAIN,
) -> np.ndarray:
    """Scale so the loudest sample sits near `target`, within reason."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < floor:
        return audio  # hiss or dead air; leave it alone
    gain = min(target / peak, max_gain)
    if abs(gain - 1.0) < 0.01:
        return audio
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)
