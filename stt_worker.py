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

"""faster-whisper wrapper: one clip in, text plus a confidence estimate out.

Two things here exist to buy accuracy without spending latency.

**A prompt that fits.** Whisper's prompt window is 224 tokens and anything
past it is silently dropped, so a roster of 50+ stations cannot simply be
listed -- most of it would be discarded at an arbitrary point. `build_prompt`
counts real tokens and stops at the budget, which is why the caller hands over
terms already ordered by how likely each station is to speak next.

**Conditioned audio.** Clips are high-passed and normalised before decoding;
under a millisecond against a transcription measured in seconds.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from audio_prep import prepare

log = logging.getLogger(__name__)

LEAD_IN = "Amateur radio net check-ins."


@dataclass
class Word:
    """One word with its timing, used to find the pause between two stations."""

    text: str
    start: float
    end: float
    offset: int
    """Character offset into Transcription.text, so a callsign found in the
    text can be located in time."""


@dataclass
class Transcription:
    text: str
    confidence: float
    """0-1, derived from Whisper's avg_logprob. Rough, but good for a UI cue."""
    language: str = ""
    no_speech_prob: float = 0.0
    words: list[Word] = field(default_factory=list)


@dataclass
class SttWorker:
    """Lazily-loaded faster-whisper model.

    model_size: tiny/base/small/medium/large-v3. See the README for the
        latency/accuracy tradeoff; `base` is the default because it keeps up on
        a laptop CPU while a net is running.
    device: "cpu", "cuda", or "auto" to use CUDA when it is available.
    initial_prompt: roster-derived hotwords, from CallsignMatcher.hotwords().
    """

    model_size: str = "base"
    device: str = "auto"
    compute_type: str | None = None
    initial_prompt: str = ""
    beam_size: int = 5
    language: str | None = "en"
    word_timestamps: bool = True
    """Needed to split a clip that caught two stations: the pause between them
    is only visible in the timings."""
    prompt_token_budget: int = 200
    """Under Whisper's 224-token window, leaving room for the lead-in."""
    condition_audio: bool = True
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    """Whisper invents text on clips that are only noise; these two are what
    keep a squelch tail from becoming a check-in."""
    active_device: str = field(default="", init=False)
    active_compute_type: str = field(default="", init=False)
    _model: object | None = field(default=None, init=False, repr=False)
    _tokenizer_cache: object | None = field(default=None, init=False, repr=False)
    prompt_terms_used: int = field(default=0, init=False)
    prompt_terms_offered: int = field(default=0, init=False)

    def reload(self, model_size: str | None = None) -> None:
        """Drop the current model and load again, optionally a different size.

        Called between clips on the STT thread, never underneath one. The
        buffering exists precisely so the seconds this takes cost latency
        rather than audio.
        """
        if model_size:
            self.model_size = model_size
        self._model = None
        self._tokenizer_cache = None
        self.load()

    def load(self) -> None:
        """Load the model. Called eagerly at startup so the first check-in of
        the net is not the thing that pays the download/init cost."""
        from faster_whisper import WhisperModel

        device = self._resolve_device()
        compute_type = self.compute_type or (
            "float16" if device == "cuda" else "int8"
        )
        log.info(
            "Loading faster-whisper %s on %s (%s)",
            self.model_size,
            device,
            compute_type,
        )
        self._model = WhisperModel(
            self.model_size, device=device, compute_type=compute_type
        )
        # Recorded so the dashboard can show what inference is *actually*
        # running on: `device: auto` quietly choosing the CPU on a machine with
        # a GPU is a thing to find out before an event, not during one.
        self.active_device = device
        self.active_compute_type = compute_type

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda"
        except Exception:  # pragma: no cover - depends on host hardware
            log.debug("CUDA probe failed; falling back to CPU", exc_info=True)
        return "cpu"

    def build_prompt(self, terms: list[str], lead_in: str = LEAD_IN) -> str:
        """Pack as many bias terms as the token window allows, in order.

        Counting real tokens rather than guessing is the point: the difference
        between a prompt that biases 48 stations and one that is truncated
        mid-word is invisible until somebody checks.
        """
        if self._model is None:
            self.load()
        tokenizer = self._tokenizer()
        if tokenizer is None:  # pragma: no cover - only if the API changes
            return lead_in + " " + ", ".join(terms[:40]) + "."

        used = len(tokenizer.encode(lead_in))
        kept: list[str] = []
        for term in terms:
            cost = len(tokenizer.encode(f" {term},"))
            if used + cost > self.prompt_token_budget:
                break
            kept.append(term)
            used += cost
        self.prompt_terms_used = len(kept)
        self.prompt_terms_offered = len(terms)
        return f"{lead_in} " + ", ".join(kept) + "."

    def _tokenizer(self):
        if self._tokenizer_cache is None and self._model is not None:
            try:
                from faster_whisper.tokenizer import Tokenizer

                self._tokenizer_cache = Tokenizer(
                    self._model.hf_tokenizer,  # type: ignore[union-attr]
                    multilingual=True,
                    task="transcribe",
                    language=self.language or "en",
                )
            except Exception:  # pragma: no cover - depends on the library
                log.debug("No tokenizer available; prompt will be estimated")
        return self._tokenizer_cache

    def transcribe(self, audio: np.ndarray, prompt: str | None = None) -> Transcription:
        if self._model is None:
            self.load()
        if self.condition_audio:
            audio = prepare(audio)
        segments, info = self._model.transcribe(  # type: ignore[union-attr]
            audio,
            language=self.language,
            beam_size=self.beam_size,
            initial_prompt=(prompt if prompt is not None else self.initial_prompt) or None,
            vad_filter=False,  # the segmenter already did this
            condition_on_previous_text=False,  # transmissions are independent
            word_timestamps=self.word_timestamps,
            log_prob_threshold=self.log_prob_threshold,
            no_speech_threshold=self.no_speech_threshold,
        )
        segments = list(segments)
        text = " ".join(s.text.strip() for s in segments).strip()
        confidence = _confidence(segments)
        return Transcription(
            text=text,
            words=_words(segments, text),
            confidence=confidence,
            language=getattr(info, "language", "") or "",
            no_speech_prob=(
                sum(s.no_speech_prob for s in segments) / len(segments)
                if segments
                else 1.0
            ),
        )


def _words(segments: list, text: str) -> list[Word]:
    """Flatten Whisper's per-word timings and locate each word in `text`.

    The offsets are found by scanning forward through the joined text, which
    keeps them correct even though joining the segments changed the spacing.
    """
    words: list[Word] = []
    cursor = 0
    for segment in segments:
        for word in getattr(segment, "words", None) or []:
            stripped = word.word.strip()
            if not stripped:
                continue
            found = text.find(stripped, cursor)
            if found < 0:  # shouldn't happen, but never guess an offset
                continue
            cursor = found + len(stripped)
            words.append(
                Word(text=stripped, start=word.start, end=word.end, offset=found)
            )
    return words


def _confidence(segments: list) -> float:
    """Duration-weighted mean of exp(avg_logprob), clamped to 0-1.

    Whisper does not emit a calibrated confidence; this is a monotonic proxy
    that is fine for colouring a cell in the dashboard and nothing more.
    """
    if not segments:
        return 0.0
    total_weight = 0.0
    total = 0.0
    for segment in segments:
        weight = max(segment.end - segment.start, 1e-3)
        total += math.exp(segment.avg_logprob) * weight
        total_weight += weight
    return max(0.0, min(1.0, total / total_weight))
