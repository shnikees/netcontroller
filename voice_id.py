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
import re
import wave
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


_BACKEND = None
"""A trained model, when one is configured. None means the built-in embedder."""


def set_backend(backend) -> None:
    """Install a trained embedder, or pass None to go back to the built-in one.

    Vectors from two backends mean nothing to each other, so switching invalidates
    every stored profile -- which is what the kept enrolment audio and
    `tools/rebuild_voices.py` are for.
    """
    global _BACKEND
    _BACKEND = backend


def backend_name() -> str:
    return "built-in" if _BACKEND is None else getattr(
        _BACKEND, "path", type(_BACKEND).__name__
    ).__str__()


def embed(audio: np.ndarray, rate: int = SAMPLE_RATE) -> np.ndarray | None:
    """Turn a clip into a fixed-length voice vector, or None if too short."""
    if audio is None or len(audio) < int(rate * MIN_SPEECH_S):
        return None
    if _BACKEND is not None:
        return _BACKEND(np.asarray(audio, dtype=np.float32), rate)
    return embed_builtin(audio, rate)


def embed_builtin(audio: np.ndarray, rate: int = SAMPLE_RATE) -> np.ndarray | None:
    """The dependency-free embedder: log-mel cepstral statistics.

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


_FILTERBANK_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def _mel_filterbank(frame_size: int, rate: int, mels: int = MEL_FILTERS) -> np.ndarray:
    key = (frame_size, rate, mels)
    if key in _FILTERBANK_CACHE:
        return _FILTERBANK_CACHE[key]

    bins = frame_size // 2 + 1
    # 80 Hz to 4 kHz: the band an FM voice channel actually carries, so the
    # filters are spent where there is signal rather than on empty spectrum.
    low, high = _to_mel(80.0), _to_mel(4000.0)
    points = _from_mel(np.linspace(low, high, mels + 2))
    positions = np.floor((frame_size + 1) * points / rate).astype(int)
    positions = np.clip(positions, 0, bins - 1)

    bank = np.zeros((mels, bins), dtype=np.float32)
    for i in range(mels):
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


MAX_SAMPLES = 12
"""Individual embeddings kept per station, on top of the centroid.

The centroid is what identification uses. The samples are what *calibration*
uses: you cannot work out how close two recordings of one operator are from an
average of them. A dozen is enough to characterise the spread and small enough
that the profile file stays readable."""


@dataclass
class Profile:
    """What one station sounds like, averaged over their enrolled clips."""

    callsign: str
    centroid: np.ndarray
    count: int = 0
    samples: list = field(default_factory=list)
    """Recent individual embeddings, kept so thresholds can be calibrated."""

    def add(self, vector: np.ndarray) -> None:
        """Fold in another clip. A running mean, so old nets still count."""
        self.centroid = (self.centroid * self.count + vector) / (self.count + 1)
        norm = np.linalg.norm(self.centroid)
        if norm > 1e-9:
            self.centroid = self.centroid / norm
        self.count += 1
        self.samples.append(vector)
        if len(self.samples) > MAX_SAMPLES:
            # Keep the most recent: a voice heard last week is more use than
            # one from a net six months ago on a different radio.
            self.samples.pop(0)


@dataclass
class EnrolmentAudio:
    """The clips each profile was built from, kept so it can be rebuilt.

    Embeddings from two different models are not comparable, so the day the
    embedder changes every profile becomes void. With the audio still here that
    is a re-embed pass measured in minutes; without it, enrolment starts from
    nothing and the next few nets are spent getting back to where you were.

    It also makes an honest comparison possible: two embedders scored on the
    *same* clips, rather than on two different months of traffic.

    Size is bounded per station and per clip, because this lives on an SD card:
    the defaults work out around a megabyte per station, so a hundred-station
    roster costs roughly a hundred megabytes.
    """

    directory: str | Path
    per_station: int = 6
    max_seconds: float = 5.0

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)

    def _folder(self, callsign: str) -> Path:
        # Callsigns are alphanumeric, but never trust a roster file with a
        # path: one stray slash would write outside the directory.
        safe = re.sub(r"[^A-Z0-9_-]", "_", callsign.upper())
        return Path(self.directory) / safe

    def save(self, callsign: str, audio: np.ndarray) -> Path | None:
        """Keep one clip for this station, dropping the oldest past the limit."""
        if audio is None or not len(audio):
            return None
        try:
            folder = self._folder(callsign)
            folder.mkdir(parents=True, exist_ok=True)

            limit = int(SAMPLE_RATE * self.max_seconds)
            clip = np.asarray(audio, dtype=np.float32)[:limit]
            pcm = np.clip(clip * 32768.0, -32768, 32767).astype(np.int16)

            existing = sorted(folder.glob("*.wav"))
            index = 0
            if existing:
                index = int(existing[-1].stem.split("-")[-1]) + 1
            path = folder / f"clip-{index:04d}.wav"
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(SAMPLE_RATE)
                handle.writeframes(pcm.tobytes())

            # Keep the most recent: a voice heard last week is more use than
            # one from a net six months ago on a different radio.
            for stale in sorted(folder.glob("*.wav"))[: -self.per_station]:
                stale.unlink(missing_ok=True)
            return path
        except OSError as exc:
            log.warning("Could not keep enrolment audio for %s: %s", callsign, exc)
            return None

    def clips(self, callsign: str) -> list[np.ndarray]:
        folder = self._folder(callsign)
        if not folder.exists():
            return []
        out: list[np.ndarray] = []
        for path in sorted(folder.glob("*.wav")):
            try:
                with wave.open(str(path), "rb") as handle:
                    raw = handle.readframes(handle.getnframes())
                out.append(np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0)
            except (OSError, wave.Error) as exc:
                log.warning("Skipping unreadable enrolment clip %s: %s", path, exc)
        return out

    def stations(self) -> list[str]:
        directory = Path(self.directory)
        if not directory.exists():
            return []
        return sorted(p.name for p in directory.iterdir() if p.is_dir())

    def total_clips(self) -> int:
        return sum(len(self.clips(name)) for name in self.stations())


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
    audio: EnrolmentAudio | None = None
    """Where the clips behind these profiles are kept, if they are kept."""

    # -- learning ----------------------------------------------------------

    def enrol(self, callsign: str, audio: np.ndarray) -> bool:
        """Learn from a clip whose callsign is known. Returns whether it took."""
        vector = embed(audio)
        if vector is None:
            return False
        if self.audio is not None:
            self.audio.save(callsign, audio)
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
                    samples=[
                        np.asarray(s, dtype=np.float32)
                        for s in entry.get("samples", [])
                    ],
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
                        "samples": [
                            [round(float(x), 6) for x in sample]
                            for sample in p.samples
                        ],
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

    def rebuild(self, store: EnrolmentAudio | None = None) -> tuple[int, int]:
        """Re-embed every kept clip and rebuild the profiles from scratch.

        What the retained audio is for. Run it after changing the embedder --
        the old vectors are meaningless under a new model, and this replaces
        them in minutes rather than waiting weeks for re-enrolment.

        Returns (stations rebuilt, clips used).
        """
        store = store or self.audio
        if store is None:
            return (0, 0)

        rebuilt: dict[str, Profile] = {}
        clips_used = 0
        for callsign in store.stations():
            for clip in store.clips(callsign):
                vector = embed(clip)
                if vector is None:
                    continue
                profile = rebuilt.get(callsign)
                if profile is None:
                    profile = Profile(callsign=callsign, centroid=np.zeros_like(vector))
                    rebuilt[callsign] = profile
                profile.add(vector)
                clips_used += 1
        self.profiles = rebuilt
        return (len(rebuilt), clips_used)

    def forget(self, callsign: str) -> None:
        """Drop a profile -- for when a station's entries were mis-enrolled."""
        self.profiles.pop(callsign, None)

    @property
    def known(self) -> list[str]:
        return [c for c, p in self.profiles.items() if p.count >= self.min_enrolments]
