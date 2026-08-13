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

"""Run a trained speaker-embedding model through ONNX Runtime.

The numpy embedder in `voice_id` was chosen so the app installs on a Pi with no
deep-learning stack. It is 24 hand-picked statistics that were never trained to
tell one speaker from another, and it conflates the voice with the channel --
a profile is really "Frank on his HT", and breaks when Frank checks in mobile.

A trained network (ECAPA-TDNN, TitaNet, and others) is discriminatively trained
on thousands of speakers and augmented specifically for channel robustness. The
route that keeps the install small is an **ONNX export under `onnxruntime`** --
an 18 MB wheel and a model of about the same, against hundreds of megabytes for
PyTorch or NeMo.

**Deliberately on the CPU.** A speaker embedding is a small model over a few
seconds of audio: milliseconds either way. Any GPU present belongs to Whisper,
and spending it here would trade something that matters for something that
does not.

The awkward part is that exported speaker models do not agree on their inputs.
Some take a raw waveform, some take log-mel features, some want a length
alongside, and the feature layout is transposed between families. Rather than
hard-coding one vendor's convention, the session is *inspected* and the input
built to match -- so a model somebody downloads later has a fair chance of
working without a code change.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

try:  # pragma: no cover - depends on what is installed
    import onnxruntime

    HAVE_ONNXRUNTIME = True
except ImportError:  # pragma: no cover
    HAVE_ONNXRUNTIME = False

SAMPLE_RATE = 16_000
DEFAULT_MELS = 80
"""What speaker models overwhelmingly expect when they take features."""


class OnnxEmbedder:
    """A trained speaker-embedding model, adapted to whatever it asks for."""

    def __init__(self, path: str | Path, mels: int = DEFAULT_MELS) -> None:
        if not HAVE_ONNXRUNTIME:
            raise RuntimeError(
                "onnxruntime is not installed -- pip install onnxruntime"
            )
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"No speaker model at {self.path}")

        self.mels = mels
        # CPU on purpose: see the module docstring. A GPU provider here would
        # compete with transcription for the thing that actually needs it.
        self.session = onnxruntime.InferenceSession(
            str(self.path), providers=["CPUExecutionProvider"]
        )
        self._plan = self._inspect()
        log.info(
            "Speaker model %s: %s input, %d dimensions",
            self.path.name,
            self._plan["kind"],
            self.dimensions or 0,
        )

    # -- what does this model want? ----------------------------------------

    def _inspect(self) -> dict:
        """Work out how to feed this model from its declared inputs."""
        inputs = self.session.get_inputs()
        main = inputs[0]
        shape = list(main.shape)

        plan: dict = {
            "name": main.name,
            "shape": shape,
            "length_input": None,
            "transposed": False,
        }

        # A second small integer input is a length: TitaNet and friends take
        # (audio_signal, length) so the model can mask padding.
        for extra in inputs[1:]:
            if len(extra.shape) <= 1:
                plan["length_input"] = extra.name
                break

        if len(shape) <= 2:
            plan["kind"] = "waveform"
            return plan

        # Rank 3 means features. Which axis is which is the part that differs
        # between families, so it is read off whichever dimension is fixed.
        plan["kind"] = "features"
        fixed = [(i, d) for i, d in enumerate(shape) if isinstance(d, int) and d > 1]
        if fixed:
            axis, size = fixed[0]
            self.mels = size
            # [batch, mels, frames] is the NeMo layout; [batch, frames, mels]
            # the other. Axis 1 fixed means mels come first.
            plan["transposed"] = axis == 1
        return plan

    @property
    def dimensions(self) -> int | None:
        shape = self.session.get_outputs()[0].shape
        last = shape[-1] if shape else None
        return last if isinstance(last, int) else None

    @property
    def kind(self) -> str:
        return self._plan["kind"]

    # -- embedding ---------------------------------------------------------

    def __call__(self, audio: np.ndarray, rate: int = SAMPLE_RATE) -> np.ndarray | None:
        samples = np.asarray(audio, dtype=np.float32)
        if samples.size < rate // 2:
            return None

        if self._plan["kind"] == "waveform":
            main = samples[None, :]
        else:
            features = log_mel(samples, rate, self.mels)
            if features.shape[0] < 4:
                return None
            # features come out [frames, mels]
            main = (features.T if self._plan["transposed"] else features)[None, :, :]

        feed = {self._plan["name"]: main.astype(np.float32)}
        if self._plan["length_input"]:
            feed[self._plan["length_input"]] = np.array(
                [main.shape[-1] if self._plan["transposed"] else main.shape[1]],
                dtype=np.int64,
            )

        try:
            outputs = self.session.run(None, feed)
        except Exception as exc:  # pragma: no cover - depends on the model
            log.warning("Speaker model failed on a clip: %s", exc)
            return None

        # Some exports return (logits, embedding); the embedding is the one
        # that is not the size of a speaker inventory, so prefer the last.
        vector = np.asarray(outputs[-1], dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-9:
            return None
        return (vector / norm).astype(np.float32)


def log_mel(audio: np.ndarray, rate: int, mels: int) -> np.ndarray:
    """Log-mel filterbank features, [frames, mels].

    Shares the framing and filterbank in `voice_id`, so both embedders hear
    the same thing and a comparison between them is about the model rather
    than the front end.
    """
    from voice_id import _frame, _mel_filterbank, _pre_emphasis

    frames = _frame(_pre_emphasis(audio), rate)
    if frames.shape[0] == 0:
        return np.zeros((0, mels), dtype=np.float32)

    spectrum = np.abs(np.fft.rfft(frames * np.hamming(frames.shape[1]), axis=1)) ** 2
    bank = _mel_filterbank(frames.shape[1], rate, mels)
    return np.log(np.maximum(spectrum @ bank.T, 1e-10)).astype(np.float32)


def load(path: str | Path | None, mels: int = DEFAULT_MELS) -> OnnxEmbedder | None:
    """Load a model, or explain in one line why the numpy embedder is being
    used instead. A missing model must never stop a net starting."""
    if not path:
        log.warning("voice.backend is onnx but no voice.model_path is set")
        return None
    if not HAVE_ONNXRUNTIME:
        log.warning(
            "voice.backend is onnx but onnxruntime is not installed; "
            "falling back to the built-in embedder"
        )
        return None
    try:
        return OnnxEmbedder(path, mels=mels)
    except Exception as exc:
        log.warning(
            "Could not load speaker model %s (%s); falling back to the "
            "built-in embedder",
            path,
            exc,
        )
        return None
