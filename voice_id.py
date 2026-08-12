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

"""Recognise a station by voice, to help when the callsign is not usable.

Every other part of this app works on what was *said*. This one works on *who
said it*, and it exists for the case nothing else can reach: a transmission
carrying no usable callsign at all. "Back to you, net control" is unmatched
forever no matter how good the transcription gets -- but it is still Frank's
voice, and Frank checked in ten minutes ago.

The enrolment data is already being produced. Every confidently matched
check-in, and every line an operator corrects by hand, is a labelled pair of
(audio, callsign). Profiles accumulate across nets, so the system knows more
voices every week without anyone doing anything extra.

**Suggestions only, and only on unmatched lines.** A voice match never
overrides a callsign that was actually heard and never silently fills one in.
It offers a name for the operator to accept with the click that already exists,
because the failure modes here are real: FM narrowband flattens the features
that distinguish speakers, two operators share one radio, and a relayed
transmission is somebody else's voice entirely. A wrong callsign nobody
notices is the outcome this whole app is built to avoid.

The embedding is deliberately dependency-free -- log-mel cepstral statistics in
numpy -- so it runs on a Pi without dragging in a deep-learning runtime. It is
weaker than a trained speaker-embedding network, which is the honest trade for
staying installable; the threshold is set high and the result is a suggestion,
so a weak model costs recall rather than correctness.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
FRAME_MS = 25
HOP_MS = 10
MEL_FILTERS = 26
CEPSTRA = 13
MIN_SPEECH_S = 0.6
"""Shorter than this carries too little voice to characterise a speaker."""


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------


def embed(audio: np.ndarray, rate: int = SAMPLE_RATE) -> np.ndarray | None:
    """Turn a clip into a fixed-length voice vector, or None if too short.

    Mean and standard deviation of log-mel cepstra across the clip: the mean
    carries the average timbre of the voice, the deviation carries how much it
    moves, and pooling over the whole clip makes the result independent of what
    was actually said.
    """
    if audio is None or len(audio) < int(rate * MIN_SPEECH_S):
        return None

    frames = _frame(_pre_emphasis(np.asarray(audio, dtype=np.float32)), rate)
    if frames.shape[0] < 4:
        return None

    spectrum = np.abs(np.fft.rfft(frames * np.hamming(frames.shape[1]), axis=1)) ** 2
    mel = spectrum @ _mel_filterbank(frames.shape[1], rate).T
    log_mel = np.log(np.maximum(mel, 1e-10))
    cepstra = _dct(log_mel)[:, 1:CEPSTRA]  # drop c0: it is loudness, not voice

    vector = np.concatenate([cepstra.mean(axis=0), cepstra.std(axis=0)])
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        return None
    return (vector / norm).astype(np.float32)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two unit vectors, clamped to 0-1."""
    return float(max(0.0, min(1.0, float(np.dot(a, b)))))


def _pre_emphasis(audio: np.ndarray, coefficient: float = 0.97) -> np.ndarray:
    out = np.empty_like(audio)
    out[0] = audio[0]
    np.subtract(audio[1:], coefficient * audio[:-1], out=out[1:])
    return out


def _frame(audio: np.ndarray, rate: int) -> np.ndarray:
    size = int(rate * FRAME_MS / 1000)
    hop = int(rate * HOP_MS / 1000)
    if len(audio) < size:
        return np.zeros((0, size), dtype=np.float32)
    count = 1 + (len(audio) - size) // hop
    indices = np.arange(size)[None, :] + hop * np.arange(count)[:, None]
    return audio[indices]


_FILTERBANK_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _mel_filterbank(frame_size: int, rate: int) -> np.ndarray:
    key = (frame_size, rate)
    if key in _FILTERBANK_CACHE:
        return _FILTERBANK_CACHE[key]

    bins = frame_size // 2 + 1
    # 80 Hz to 4 kHz: the band an FM voice channel actually carries, so the
    # filters are spent where there is signal rather than on empty spectrum.
    low, high = _to_mel(80.0), _to_mel(4000.0)
    points = _from_mel(np.linspace(low, high, MEL_FILTERS + 2))
    positions = np.floor((frame_size + 1) * points / rate).astype(int)
    positions = np.clip(positions, 0, bins - 1)

    bank = np.zeros((MEL_FILTERS, bins), dtype=np.float32)
    for i in range(MEL_FILTERS):
        left, centre, right = positions[i], positions[i + 1], positions[i + 2]
        if centre > left:
            bank[i, left:centre] = np.linspace(0, 1, centre - left, endpoint=False)
        if right > centre:
            bank[i, centre:right] = np.linspace(1, 0, right - centre, endpoint=False)
    _FILTERBANK_CACHE[key] = bank
    return bank


def _to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _from_mel(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _dct(values: np.ndarray) -> np.ndarray:
    """DCT-II along the last axis, written out to avoid a scipy dependency."""
    n = values.shape[1]
    basis = np.cos(
        np.pi / n * (np.arange(n)[None, :] + 0.5) * np.arange(n)[:, None]
    )
    return values @ basis.T


# --------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------


@dataclass
class Profile:
    """What one station sounds like, averaged over their enrolled clips."""

    callsign: str
    centroid: np.ndarray
    count: int = 0

    def add(self, vector: np.ndarray) -> None:
        """Fold in another clip. A running mean, so old nets still count."""
        self.centroid = (self.centroid * self.count + vector) / (self.count + 1)
        norm = np.linalg.norm(self.centroid)
        if norm > 1e-9:
            self.centroid = self.centroid / norm
        self.count += 1


@dataclass
class Suggestion:
    callsign: str
    score: float
    runner_up: str | None = None
    runner_up_score: float = 0.0


@dataclass
class VoiceProfiles:
    """Voice profiles for the roster, learned from confirmed check-ins.

    min_similarity: cosine score before a voice is worth suggesting. High on
        purpose -- a suggestion the operator has to think about is worse than
        no suggestion.
    margin: the best match must beat the runner-up by this much. Two operators
        who sound alike should produce no suggestion rather than a coin flip.
    min_enrolments: clips needed before a profile is used at all. One clip is
        one moment of one net, and mistaking noise for a voice is how phantom
        identifications start.
    """

    path: str | Path | None = None
    min_similarity: float = 0.82
    margin: float = 0.06
    min_enrolments: int = 2
    profiles: dict[str, Profile] = field(default_factory=dict)

    # -- learning ----------------------------------------------------------

    def enrol(self, callsign: str, audio: np.ndarray) -> bool:
        """Learn from a clip whose callsign is known. Returns whether it took."""
        vector = embed(audio)
        if vector is None:
            return False
        profile = self.profiles.get(callsign)
        if profile is None:
            profile = Profile(callsign=callsign, centroid=np.zeros_like(vector))
            self.profiles[callsign] = profile
        profile.add(vector)
        return True

    # -- using -------------------------------------------------------------

    def identify(self, audio: np.ndarray) -> Suggestion | None:
        """Best voice match for a clip, or None when nothing is confident."""
        vector = embed(audio)
        if vector is None:
            return None

        scored = sorted(
            (
                (similarity(vector, p.centroid), p.callsign)
                for p in self.profiles.values()
                if p.count >= self.min_enrolments
            ),
            reverse=True,
        )
        if not scored:
            return None

        best_score, best = scored[0]
        runner_up, runner_up_score = (None, 0.0)
        if len(scored) > 1:
            runner_up_score, runner_up = scored[1]

        if best_score < self.min_similarity:
            return None
        if runner_up is not None and best_score - runner_up_score < self.margin:
            # Two voices this close is not an identification, it is a guess.
            return None
        return Suggestion(
            callsign=best,
            score=best_score,
            runner_up=runner_up,
            runner_up_score=runner_up_score,
        )

    # -- persistence -------------------------------------------------------

    def load(self) -> int:
        """Read profiles from disk. Returns how many were loaded."""
        if not self.path or not Path(self.path).exists():
            return 0
        try:
            data = json.loads(Path(self.path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Could not read voice profiles from %s: %s", self.path, exc)
            return 0

        for callsign, entry in (data.get("profiles") or {}).items():
            vector = np.asarray(entry.get("centroid", []), dtype=np.float32)
            if vector.size:
                self.profiles[callsign] = Profile(
                    callsign=callsign,
                    centroid=vector,
                    count=int(entry.get("count", 1)),
                )
        return len(self.profiles)

    def save(self) -> bool:
        """Write profiles out so next week's net starts knowing these voices."""
        if not self.path:
            return False
        try:
            payload = {
                "profiles": {
                    callsign: {
                        "centroid": [round(float(x), 6) for x in p.centroid],
                        "count": p.count,
                    }
                    for callsign, p in self.profiles.items()
                }
            }
            target = Path(self.path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload), encoding="utf-8")
            return True
        except OSError as exc:
            log.error("Could not save voice profiles to %s: %s", self.path, exc)
            return False

    def forget(self, callsign: str) -> None:
        """Drop a profile -- for when a station's entries were mis-enrolled."""
        self.profiles.pop(callsign, None)

    @property
    def known(self) -> list[str]:
        return [c for c, p in self.profiles.items() if p.count >= self.min_enrolments]
