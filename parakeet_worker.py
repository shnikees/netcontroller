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

"""Parakeet TDT 0.6b behind the same interface as `SttWorker`.

Measured 2026-08-26 over 1,525 clips and 307.8 minutes of real net audio, this
recovered **31 roster callsigns against unbiased `base` Whisper's 19**, at the
same wall clock, and fabricated nothing detectable. It also stayed silent on 60
clips where Whisper wrote text for dead carrier. See docs/HARDWARE.md.

**It takes no bias terms, and that is the feature.** There is no prompt and no
hotwords option -- `build_bias` returns an empty string. On the same corpus, 87%
of the callsigns prompted Whisper reported had no acoustic support from any
unprompted engine (`tools/cross_check.py`), so the roster biasing that
`stt_worker.py` spends its token budget on was buying mostly echo. An engine
that cannot be told the roster cannot recite it back.

**How it runs, and the one real cost.** whisper.cpp builds a `parakeet-cli`
binary, and that is the whole integration: there is no Python binding, no server
mode and no way to feed it audio on a pipe, so each clip is one subprocess over
a temporary wav. Measured on an M-series laptop:

    one clip, cold spawn      1.35 s
    20 clips in one spawn     3.18 s

so roughly **1.25 s of model loading per clip** and 0.1 s of actual inference.
That is wasteful and it is still fast enough: a 10-second transmission costs
about 1.35 s end to end, which is comparable to `base` Whisper on the same
machine and far inside realtime. Fixing it properly means either a server mode
upstream or driving `libparakeet` through ctypes, and neither is worth doing
before the engine has run a net.

**What this loses against `SttWorker`, and it is worth knowing before switching:**

- **Confidence is on a different scale.** Whisper's number is
  `exp(avg_logprob)`; this one is the mean per-token probability the decoder
  reports. Both are 0-1 and monotonic, and they are *not* comparable. Anything
  calibrated against Whisper's scale -- the escalation thresholds, whatever
  `tools/calibrate.py` last wrote -- is invalid after switching engines and has
  to be re-derived.
- **`no_speech_prob` is binary, not a probability.** Parakeet either returns
  tokens or returns nothing, so this is 0.0 or 1.0 with nothing in between. It
  is honest about the two cases it can distinguish rather than inventing a
  gradient.
- **`model_size` is not a size.** Reported as a fixed name so the dashboard has
  something to show; changing model size from the dashboard is meaningless here
  and `reload` treats a requested size as a no-op rather than pretending.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from audio_prep import prepare
from stt_worker import Transcription, Word

log = logging.getLogger(__name__)

MODEL_NAME = "parakeet-tdt-0.6b"

TOKEN_RE = re.compile(
    r"""p=(?P<p>[\d.]+).*?
        t0=\s*(?P<t0>\d+)\s+
        t1=\s*(?P<t1>\d+)\s+
        word_start=(?P<start>true|false)\s+
        "(?P<text>.*)"$""",
    re.VERBOSE,
)
"""One line of `-ps` token detail, which arrives on stderr:

    [ 0] id= 346 frame= 1 ... p=0.8869 plog=-10.4921 t0= 8 t1= 16 word_start=true "|A"

`plog` is deliberately not read: it is not the log of `p` -- for p=0.9918 it
reports -14.06 -- so it is some pre-softmax quantity and guessing at its meaning
would put a wrong number on the dashboard.
"""

SUCCESS_MARKER = "Segments ("
"""How to tell a working run from a broken one, because the exit code will not.
`parakeet-cli` returns 0 for a file that does not exist. It prints
`Segments (0):` for a clip it judged silent and `Segments (1):` for one it
transcribed, so the marker is present either way and absent only on real
failure. Without this check a missing binary or an unreadable clip would look
exactly like silence, and a net would log nothing while appearing healthy."""

WORD_MARK = "▁"
"""The sentencepiece word-start mark, which prefixes the first token of a word."""

TIMEOUT_SECONDS = 120.0
"""Generous: the measurement is 1.35 s for a clip. This exists so a wedged
subprocess cannot stall the transcription thread for the rest of the net."""


@dataclass
class ParakeetWorker:
    """Duck-typed replacement for `SttWorker`, driving `parakeet-cli`.

    binary: path to `parakeet-cli` from a whisper.cpp build.
    model: path to the converted ggml model. Produced by whisper.cpp's
        `models/convert-parakeet-to-ggml.py` from the 2.5 GB `.nemo` release,
        which yields a 1.2 GB ggml file.
    """

    binary: str = "parakeet-cli"
    model: str = ""
    cpu_threads: int = 0
    """Threads for `-t`. 0 leaves the binary's own default of 4."""
    condition_audio: bool = True
    use_gpu: bool = True
    """Metal, CUDA or Vulkan depending on how whisper.cpp was built. Passing
    `-ng` disables it. Unlike CTranslate2 this is not CUDA-only, which is the
    reason the hardware options in docs/HARDWARE.md reopen at all."""

    # -- accepted for interface compatibility, and unused ------------------
    # app.py builds one worker for the live path and one for escalation from
    # the same config block. These are the fields it passes that mean nothing
    # to a transducer with no prompt window; taking them and ignoring them
    # keeps the call sites identical.
    bias_mode: str = "none"
    beam_size: int = 0
    prompt_token_budget: int = 0
    initial_prompt: str = ""
    language: str | None = "en"
    word_timestamps: bool = True
    device: str = "auto"
    compute_type: str | None = None
    model_size: str = MODEL_NAME

    active_device: str = field(default="", init=False)
    active_compute_type: str = field(default="", init=False)
    prompt_terms_used: int = field(default=0, init=False)
    prompt_terms_offered: int = field(default=0, init=False)
    _ready: bool = field(default=False, init=False, repr=False)

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        """Check the binary and model are really there.

        Called eagerly at startup for the same reason `SttWorker.load` is: a
        wrong path should stop the app before a net rather than surface as an
        empty log during one. There is nothing to keep in memory between clips,
        so this validates rather than loads.
        """
        resolved = shutil.which(self.binary) or self.binary
        if not Path(resolved).exists():
            raise FileNotFoundError(
                f"parakeet-cli not found at {self.binary!r}. Build whisper.cpp "
                "and point whisper.parakeet_binary at build/bin/parakeet-cli."
            )
        if not self.model or not Path(self.model).exists():
            raise FileNotFoundError(
                f"Parakeet model not found at {self.model!r}. Convert the .nemo "
                "release with whisper.cpp's models/convert-parakeet-to-ggml.py "
                "and point whisper.parakeet_model at the result."
            )
        self.binary = resolved
        self.active_device = "gpu" if self.use_gpu else "cpu"
        self.active_compute_type = "ggml"
        self._ready = True
        log.info("Parakeet ready: %s (%s)", Path(self.model).name, self.active_device)

    def reload(self, model_size: str | None = None) -> None:
        """No-op beyond re-validating.

        Model size is not a dial on this engine, so a size requested from the
        dashboard is logged and ignored rather than silently appearing to work.
        """
        if model_size and model_size != MODEL_NAME:
            log.warning(
                "Ignoring model size %r: Parakeet ships one model. Switch "
                "whisper.engine back to faster-whisper to change sizes.",
                model_size,
            )
        self._ready = False
        self.load()

    # -- biasing, of which there is none -----------------------------------

    def build_bias(self, terms: list[str]) -> str:
        """Nothing. Recorded as offered-but-unused so the dashboard is honest.

        The roster still does all the work it ever did -- it is what
        `CallsignMatcher` matches *against*, downstream of here. What is gone is
        telling the decoder in advance, which measured as 87% echo.
        """
        self.prompt_terms_offered = len(terms)
        self.prompt_terms_used = 0
        return ""

    def build_prompt(self, terms: list[str], lead_in: str = "") -> str:
        """Alias, for callers that predate `build_bias`."""
        return self.build_bias(terms)

    # -- the work ----------------------------------------------------------

    def transcribe(self, audio: np.ndarray, prompt: str | None = None) -> Transcription:
        """One clip in, text plus timings out. `prompt` is accepted and ignored."""
        if not self._ready:
            self.load()
        if audio.size == 0:
            return Transcription(text="", confidence=0.0, no_speech_prob=1.0)
        if self.condition_audio:
            audio = prepare(audio)

        with tempfile.TemporaryDirectory(prefix="netcontroller-pk-") as scratch:
            clip = Path(scratch) / "clip.wav"
            _write_wav(clip, audio)
            command = [self.binary, "-m", self.model, "-ps", str(clip)]
            if self.cpu_threads:
                command += ["-t", str(self.cpu_threads)]
            if not self.use_gpu:
                command.append("-ng")
            finished = subprocess.run(
                command, capture_output=True, text=True, timeout=TIMEOUT_SECONDS
            )

        # The exit code is not evidence; see SUCCESS_MARKER.
        if SUCCESS_MARKER not in finished.stderr:
            tail = (finished.stderr or "").strip().splitlines()
            raise RuntimeError(
                "parakeet-cli produced no segment report: "
                + (tail[-1][:160] if tail else "no output at all")
            )

        text = finished.stdout.strip()
        words = _parse_words(finished.stderr, text)
        return Transcription(
            text=text,
            words=words,
            confidence=_confidence(finished.stderr),
            language=self.language or "",
            # Binary by construction: the engine either returned tokens or did
            # not. Silence on an unintelligible clip is the behaviour that made
            # it worth switching to, so it must reach the caller as such.
            no_speech_prob=0.0 if text else 1.0,
        )


# --------------------------------------------------------------------------
# Parsing `-ps` output
# --------------------------------------------------------------------------


def _tokens(stderr: str) -> list[tuple[float, int, int, bool, str]]:
    """Every token line, as (p, t0_cs, t1_cs, starts_word, text)."""
    found = []
    for line in stderr.splitlines():
        match = TOKEN_RE.search(line)
        if match:
            found.append((
                float(match["p"]),
                int(match["t0"]),
                int(match["t1"]),
                match["start"] == "true",
                match["text"].replace(WORD_MARK, ""),
            ))
    return found


def _parse_words(stderr: str, text: str) -> list[Word]:
    """Group tokens into words, and locate each word in `text`.

    Word timestamps exist here for one reason: splitting a clip that caught two
    stations, which is only visible in the pause between them. Tokens are
    sub-word pieces ("Go" + "od"), so they are glued back together on the
    `word_start` flag.

    Timestamps are centiseconds. Confirmed against a 54.36 s clip whose last
    token ended at t1=5408; the `frame` field is 80 ms units and is not used.

    Offsets are found by scanning forward, and a word that cannot be located is
    dropped rather than given a guessed offset -- the same rule as
    `stt_worker._words`, because a wrong offset moves a callsign to the wrong
    place in time, which is worse than not knowing where it was.
    """
    grouped: list[tuple[str, int, int]] = []
    for probability, t0, t1, starts_word, piece in _tokens(stderr):
        if starts_word or not grouped:
            grouped.append((piece, t0, t1))
        else:
            previous, start, _ = grouped[-1]
            grouped[-1] = (previous + piece, start, t1)

    words: list[Word] = []
    cursor = 0
    for piece, t0, t1 in grouped:
        stripped = piece.strip()
        if not stripped:
            continue
        found = text.find(stripped, cursor)
        if found < 0:
            continue
        cursor = found + len(stripped)
        words.append(
            Word(text=stripped, start=t0 / 100.0, end=t1 / 100.0, offset=found)
        )
    return words


def _confidence(stderr: str) -> float:
    """Mean per-token probability, clamped to 0-1.

    Not comparable to `stt_worker._confidence`, which is a duration-weighted
    `exp(avg_logprob)`. Both are monotonic proxies fit for colouring a cell;
    neither is calibrated, and thresholds tuned against one are wrong for the
    other. The mean is arithmetic on purpose -- a geometric mean lets one
    low-probability comma drag a confident sentence down.
    """
    probabilities = [token[0] for token in _tokens(stderr)]
    if not probabilities:
        return 0.0
    return max(0.0, min(1.0, sum(probabilities) / len(probabilities)))


def _write_wav(path: Path, audio: np.ndarray) -> None:
    """16-bit mono at 16 kHz, which is what the pipeline carries throughout."""
    clipped = np.clip(audio, -1.0, 1.0)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes((clipped * 32767.0).astype("<i2").tobytes())
