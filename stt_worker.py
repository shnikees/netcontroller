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

"""faster-whisper wrapper: one clip in, text plus a confidence estimate out."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Transcription:
    text: str
    confidence: float
    """0-1, derived from Whisper's avg_logprob. Rough, but good for a UI cue."""
    language: str = ""
    no_speech_prob: float = 0.0


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
    _model: object | None = field(default=None, init=False, repr=False)

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

    def transcribe(self, audio: np.ndarray) -> Transcription:
        if self._model is None:
            self.load()
        segments, info = self._model.transcribe(  # type: ignore[union-attr]
            audio,
            language=self.language,
            beam_size=self.beam_size,
            initial_prompt=self.initial_prompt or None,
            vad_filter=False,  # the segmenter already did this
            condition_on_previous_text=False,  # transmissions are independent
        )
        segments = list(segments)
        text = " ".join(s.text.strip() for s in segments).strip()
        confidence = _confidence(segments)
        return Transcription(
            text=text,
            confidence=confidence,
            language=getattr(info, "language", "") or "",
            no_speech_prob=(
                sum(s.no_speech_prob for s in segments) / len(segments)
                if segments
                else 1.0
            ),
        )


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
