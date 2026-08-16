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

"""Cut a continuous frame stream into one clip per transmission.

Nets are half-duplex, so "one run of speech between silences" is a good proxy
for "one transmission". The tuning that matters is `silence_ms`: it has to be
long enough to survive the pauses inside a phonetically spelled callsign
("whiskey ... six ... alpha") without splitting one check-in into three clips.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Iterable, Iterator

import numpy as np
import webrtcvad

FRAME_BYTES_PER_MS = 32  # 16 kHz * 2 bytes per sample / 1000 ms


@dataclass
class Clip:
    """One transmission's worth of audio."""

    audio: np.ndarray
    """float32 samples in [-1, 1], 16 kHz mono -- what faster-whisper wants."""
    start_offset_ms: int
    """Milliseconds from the start of the capture session."""
    duration_ms: int
    source: str = ""
    """Which receiver heard this. Empty when only one source is configured."""
    sequence: int = 0
    """Monotonic per session; orders the disk backlog."""
    priority: int = 0
    """From the source. Higher is transcribed first when there is a backlog."""


@dataclass
class VadSegmenter:
    """webrtcvad state machine with pre-roll and hangover.

    aggressiveness: webrtcvad 0-3; 3 is the most aggressive at calling audio
        non-speech, which suits a squelched receiver feed.
    silence_ms: trailing silence that closes a clip.
    min_clip_ms: clips shorter than this are dropped as squelch tails/noise.
    max_clip_ms: hard cap so one stuck transmitter cannot buffer forever.
    preroll_ms: audio kept from *before* the trigger, so the clip does not start
        mid-syllable.
    trigger_ratio: fraction of the preroll window that must be voiced to open a
        clip -- debounces single-frame noise bursts.
    """

    frame_ms: int = 30
    aggressiveness: int = 3
    silence_ms: int = 800
    min_clip_ms: int = 400
    max_clip_ms: int = 120_000
    preroll_ms: int = 300
    trigger_ratio: float = 0.7
    gate_margin: float = 1.5
    """How far above the tracked noise floor a frame must sit to count as
    speech, on top of what webrtcvad says. See `_gate_open`."""
    gate_min_floor: float = 300.0
    """Below this floor the gate never engages, so a squelched feed -- which
    has real silence between transmissions -- behaves exactly as before."""
    gate_window_ms: int = 10_000
    """How much recent audio the floor is measured over. Long enough to span a
    gap between overs, short enough to follow a receiver whose noise changes."""
    gate_percentile: float = 10.0
    _vad: webrtcvad.Vad = field(init=False, repr=False)
    _floor: float = field(default=0.0, init=False, repr=False)
    _frames_seen: int = field(default=0, init=False, repr=False)
    _recent: collections.deque = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.frame_ms not in (10, 20, 30):
            raise ValueError("frame_ms must be 10, 20, or 30 (webrtcvad constraint)")
        self._vad = webrtcvad.Vad(self.aggressiveness)
        self._recent = collections.deque(
            maxlen=max(1, self.gate_window_ms // self.frame_ms)
        )

    def segment(self, frames: Iterable[bytes]) -> Iterator[Clip]:
        """Consume raw 16 kHz int16 frames, yielding one Clip per transmission."""
        preroll_len = max(1, self.preroll_ms // self.frame_ms)
        silence_frames = max(1, self.silence_ms // self.frame_ms)
        max_frames = max(1, self.max_clip_ms // self.frame_ms)

        ring: collections.deque[tuple[bytes, bool]] = collections.deque(
            maxlen=preroll_len
        )
        voiced: list[bytes] = []
        triggered = False
        trailing_silence = 0
        elapsed_ms = 0
        clip_start_ms = 0

        for frame in frames:
            expected = self.frame_ms * FRAME_BYTES_PER_MS
            if len(frame) != expected:
                # A short frame means the stream is closing; ignore the remnant.
                continue
            # The gate is evaluated first and unconditionally: it has to see
            # every frame to estimate a noise floor. Folding it into an `and`
            # after webrtcvad short-circuits it away on exactly the quiet
            # frames the floor is made of, leaving it measuring only speech.
            gate_open = self._gate_open(frame)
            is_speech = gate_open and self._vad.is_speech(frame, 16_000)
            elapsed_ms += self.frame_ms

            if not triggered:
                ring.append((frame, is_speech))
                speech_count = sum(1 for _, s in ring if s)
                if len(ring) == ring.maxlen and speech_count > self.trigger_ratio * len(
                    ring
                ):
                    triggered = True
                    trailing_silence = 0
                    voiced = [f for f, _ in ring]
                    clip_start_ms = elapsed_ms - len(ring) * self.frame_ms
                    ring.clear()
                continue

            voiced.append(frame)
            trailing_silence = 0 if is_speech else trailing_silence + 1

            if trailing_silence >= silence_frames or len(voiced) >= max_frames:
                clip = self._close(voiced, clip_start_ms, trailing_silence)
                triggered = False
                voiced = []
                ring.clear()
                if clip is not None:
                    yield clip

        if triggered and voiced:
            clip = self._close(voiced, clip_start_ms, trailing_silence)
            if clip is not None:
                yield clip

    # -- the noise gate ----------------------------------------------------

    GATE_WARMUP_MS = 300
    """Enough frames for a usable percentile, short enough that a noisy feed
    cannot open a junk clip before the gate starts working."""

    def _gate_open(self, frame: bytes) -> bool:
        """Whether this frame is loud enough, relative to the noise floor.

        webrtcvad decides speech from spectral shape, which works on a
        squelched receiver -- the gaps are digital silence and nothing is
        ambiguous. It does *not* work on a feed that never goes quiet. Measured
        against 75 minutes of a repeater's public stream, the quiet stretches
        between overs sat at about RMS 1400 rather than zero, and webrtcvad
        called 74% of the recording speech at *every* aggressiveness from 0 to
        3 -- the setting made no difference at all, because the hiss is well
        inside what its model considers voice.

        The gaps were there: 118 of them over 800 ms once measured against the
        noise floor instead of against silence. So the fix is not a better
        spectral model, it is a relative threshold. Track the floor, and
        require speech to stand above it.

        Self-disabling on purpose. A squelched radio has a floor near zero, so
        the threshold lands near zero and every frame passes -- identical
        behaviour to before this existed. It only engages where it is needed:
        an open squelch, a scanner feed, a streamed repeater.
        """
        samples = np.frombuffer(frame, dtype=np.int16)
        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))) if samples.size else 0.0

        self._recent.append(rms)
        self._frames_seen += 1
        if self._frames_seen * self.frame_ms < self.GATE_WARMUP_MS:
            return True

        # A low percentile of the recent past, rather than a decaying average.
        # An average chases whatever it is measuring: on a two-minute over --
        # ordinary on a social net -- an EMA floor climbs to most of the
        # speaker's own level and the gate shuts on the person it is listening
        # to. A percentile cannot, because speech is bursty: even mid-sentence
        # there are pauses between words, and those pauses are what expose the
        # real floor.
        self._floor = float(np.percentile(self._recent, self.gate_percentile))
        if self._floor < self.gate_min_floor:
            return True
        return rms >= self._floor * self.gate_margin

    @property
    def noise_floor(self) -> float:
        """Current floor estimate, for logging and the health strip."""
        return self._floor

    def _close(
        self, voiced: list[bytes], start_ms: int, trailing_silence: int
    ) -> Clip | None:
        # Trim the hangover silence back off; it is dead weight for the STT pass.
        keep = voiced[: len(voiced) - trailing_silence] if trailing_silence else voiced
        duration_ms = len(keep) * self.frame_ms
        if duration_ms < self.min_clip_ms:
            return None
        pcm = np.frombuffer(b"".join(keep), dtype=np.int16)
        return Clip(
            audio=pcm.astype(np.float32) / 32768.0,
            start_offset_ms=start_ms,
            duration_ms=duration_ms,
        )
