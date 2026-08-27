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

"""Tests for the Parakeet adapter.

The binary is not a test dependency -- these run on the recorded output of real
`parakeet-cli` invocations, which is the part that can silently change shape.
The fixtures below are trimmed from actual runs over PSRG clips on 2026-08-26.
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import parakeet_worker
from parakeet_worker import (
    ParakeetWorker,
    _confidence,
    _parse_words,
    _tokens,
    _write_wav,
)

# Real stderr from `parakeet-cli -ps` on clip c0016, whose transcript is
# "Around the". Four sub-word tokens across two words.
STDERR_AROUND = """\
read_audio_data: reading audio data from 'c0016.wav' ...
read_audio_data: trying to decode with miniaudio
Segments (1):
Segment 0: [0 -> 8] "Around the"
Tokens [4]:
  [ 0] id=  346 frame=  1 dur_idx= 1 dur_val= 1 p=0.8869 plog=-10.4921 t0=   8 t1=  16 word_start=true "▁A"
  [ 1] id=  320 frame=  2 dur_idx= 1 dur_val= 1 p=0.9975 plog=-14.0659 t0=  16 t1=  24 word_start=false "ro"
  [ 2] id=  816 frame=  3 dur_idx= 1 dur_val= 1 p=1.0000 plog=-10.8861 t0=  24 t1=  32 word_start=false "und"
  [ 3] id=  506 frame=  4 dur_idx= 4 dur_val= 4 p=0.9894 plog=-8.3364 t0=  32 t1=  64 word_start=true "▁the"
"""

# A clip the engine judged silent. Note there is no `Tokens [` line at all,
# which is why the success marker is `Segments (`.
STDERR_SILENT = """\
read_audio_data: reading audio data from 'c0002.wav' ...
read_audio_data: trying to decode with miniaudio
Segments (0):
"""


def _fake_run(stdout: str, stderr: str, calls: list | None = None):
    """Stand in for subprocess.run, recording the command it was given."""

    def run(command, **kwargs):
        if calls is not None:
            calls.append(command)
        return SimpleNamespace(returncode=0, stdout=stdout, stderr=stderr)

    return run


def _worker(tmp_path: Path, **kwargs) -> ParakeetWorker:
    binary = tmp_path / "parakeet-cli"
    binary.write_text("#!/bin/sh\n")
    model = tmp_path / "ggml-model.bin"
    model.write_bytes(b"not really a model")
    worker = ParakeetWorker(binary=str(binary), model=str(model), **kwargs)
    worker.load()
    return worker


# -- parsing ---------------------------------------------------------------


def test_tokens_are_read_off_the_stderr_report() -> None:
    found = _tokens(STDERR_AROUND)
    assert len(found) == 4
    probability, t0, t1, starts_word, text = found[0]
    assert (probability, t0, t1, starts_word, text) == (0.8869, 8, 16, True, "A")


def test_subword_tokens_are_glued_back_into_words() -> None:
    """"A" + "ro" + "und" is one word, and its timing spans all three.

    Word timings exist to find the pause between two stations sharing a clip,
    so a word whose end came from the wrong token would split in the wrong
    place.
    """
    words = _parse_words(STDERR_AROUND, "Around the")
    assert [w.text for w in words] == ["Around", "the"]
    assert words[0].start == pytest.approx(0.08)
    assert words[0].end == pytest.approx(0.32)
    assert words[1].start == pytest.approx(0.32)
    assert words[1].end == pytest.approx(0.64)


def test_word_offsets_locate_the_word_in_the_text() -> None:
    words = _parse_words(STDERR_AROUND, "Around the")
    assert [w.offset for w in words] == [0, 7]


def test_a_word_not_present_in_the_text_is_dropped_not_guessed() -> None:
    """A wrong offset moves a callsign to the wrong point in time, which is
    worse than not knowing where it was. Same rule as stt_worker._words."""
    words = _parse_words(STDERR_AROUND, "completely different text")
    assert words == []


def test_confidence_is_the_mean_token_probability() -> None:
    expected = (0.8869 + 0.9975 + 1.0 + 0.9894) / 4
    assert _confidence(STDERR_AROUND) == pytest.approx(expected)


def test_confidence_of_a_silent_clip_is_zero() -> None:
    assert _confidence(STDERR_SILENT) == 0.0


def test_timestamps_are_centiseconds() -> None:
    """Pinned against a real 54.36 s clip whose last token ended at t1=5408.

    Getting this wrong by the 80 ms `frame` unit instead would stretch every
    clip eightfold and make every split decision nonsense.
    """
    stderr = STDERR_SILENT.replace(
        "Segments (0):",
        'Segments (1):\nTokens [1]:\n  [ 0] id= 1 frame=676 dur_idx= 2 dur_val= 2 '
        'p=0.7193 plog=-7.1246 t0=5392 t1=5408 word_start=true "▁Pat"',
    )
    words = _parse_words(stderr, "Pat")
    assert words[0].end == pytest.approx(54.08)


# -- transcription ---------------------------------------------------------


def test_transcribe_returns_text_words_and_confidence(tmp_path, monkeypatch) -> None:
    worker = _worker(tmp_path)
    monkeypatch.setattr(
        subprocess, "run", _fake_run("Around the\n", STDERR_AROUND)
    )
    result = worker.transcribe(np.zeros(16_000, dtype=np.float32))
    assert result.text == "Around the"
    assert [w.text for w in result.words] == ["Around", "the"]
    assert result.confidence > 0.9
    assert result.no_speech_prob == 0.0


def test_a_silent_clip_reports_no_speech(tmp_path, monkeypatch) -> None:
    """The behaviour that made this engine worth switching to.

    Whisper writes text for dead carrier -- 60 clips of 1,525 came back empty
    from Parakeet and zero from Whisper. That silence has to reach the caller
    as silence, not as a failure and not as an empty-but-confident line.
    """
    worker = _worker(tmp_path)
    monkeypatch.setattr(subprocess, "run", _fake_run("", STDERR_SILENT))
    result = worker.transcribe(np.zeros(16_000, dtype=np.float32))
    assert result.text == ""
    assert result.words == []
    assert result.no_speech_prob == 1.0


def test_a_run_with_no_segment_report_is_an_error(tmp_path, monkeypatch) -> None:
    """parakeet-cli exits 0 for a file that does not exist.

    Without the marker check, a wrong path or a broken build would look exactly
    like a net where nobody spoke -- the log would fill with nothing and the
    dashboard would look healthy.
    """
    worker = _worker(tmp_path)
    monkeypatch.setattr(
        subprocess, "run", _fake_run("", "error: failed to open 'clip.wav'\n")
    )
    with pytest.raises(RuntimeError, match="no segment report"):
        worker.transcribe(np.zeros(16_000, dtype=np.float32))


def test_an_empty_clip_is_not_handed_to_the_binary(tmp_path, monkeypatch) -> None:
    worker = _worker(tmp_path)
    calls: list = []
    monkeypatch.setattr(subprocess, "run", _fake_run("", "", calls))
    result = worker.transcribe(np.array([], dtype=np.float32))
    assert result.no_speech_prob == 1.0
    assert calls == []


def test_threads_and_gpu_flags_reach_the_command(tmp_path, monkeypatch) -> None:
    worker = _worker(tmp_path, cpu_threads=8, use_gpu=False)
    calls: list = []
    monkeypatch.setattr(subprocess, "run", _fake_run("hi", STDERR_AROUND, calls))
    worker.transcribe(np.zeros(16_000, dtype=np.float32))
    assert "-t" in calls[0] and "8" in calls[0]
    assert "-ng" in calls[0]
    assert "-ps" in calls[0]


def test_no_gpu_flag_is_absent_when_gpu_is_wanted(tmp_path, monkeypatch) -> None:
    worker = _worker(tmp_path, use_gpu=True)
    calls: list = []
    monkeypatch.setattr(subprocess, "run", _fake_run("hi", STDERR_AROUND, calls))
    worker.transcribe(np.zeros(16_000, dtype=np.float32))
    assert "-ng" not in calls[0]


def test_a_prompt_is_accepted_and_ignored(tmp_path, monkeypatch) -> None:
    """app.py passes a prompt unconditionally; there is nowhere to put it."""
    worker = _worker(tmp_path)
    calls: list = []
    monkeypatch.setattr(subprocess, "run", _fake_run("hi", STDERR_AROUND, calls))
    worker.transcribe(np.zeros(16_000, dtype=np.float32), prompt="KJ7RAB, KJ7JXM")
    assert not any("KJ7RAB" in str(part) for part in calls[0])


# -- biasing, of which there is none ---------------------------------------


def test_build_bias_returns_nothing_but_records_what_was_offered() -> None:
    worker = ParakeetWorker(model="unused")
    assert worker.build_bias(["KJ7RAB", "KJ7JXM", "net control"]) == ""
    assert worker.prompt_terms_offered == 3
    assert worker.prompt_terms_used == 0


def test_build_prompt_is_an_alias_for_callers_that_predate_build_bias() -> None:
    worker = ParakeetWorker(model="unused")
    assert worker.build_prompt(["KJ7RAB"]) == ""


# -- lifecycle -------------------------------------------------------------


def test_a_missing_binary_fails_at_load_not_mid_net(tmp_path) -> None:
    model = tmp_path / "m.bin"
    model.write_bytes(b"x")
    worker = ParakeetWorker(binary=str(tmp_path / "nope"), model=str(model))
    with pytest.raises(FileNotFoundError, match="parakeet-cli"):
        worker.load()


def test_a_missing_model_fails_at_load(tmp_path) -> None:
    binary = tmp_path / "parakeet-cli"
    binary.write_text("#!/bin/sh\n")
    worker = ParakeetWorker(binary=str(binary), model=str(tmp_path / "nope.bin"))
    with pytest.raises(FileNotFoundError, match="model not found"):
        worker.load()


def test_reload_declines_a_model_size_rather_than_pretending(tmp_path, caplog) -> None:
    """The dashboard can ask for `small`. There is one Parakeet model.

    Reporting success would leave the dashboard naming a model that is not
    loaded, which is the kind of quiet wrongness this project keeps finding.
    """
    worker = _worker(tmp_path)
    with caplog.at_level("WARNING"):
        worker.reload("small")
    assert "one model" in caplog.text
    assert worker.model_size == parakeet_worker.MODEL_NAME


def test_wav_round_trips_as_16_bit_mono_at_16k(tmp_path) -> None:
    audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    path = tmp_path / "clip.wav"
    _write_wav(path, audio)
    with wave.open(str(path)) as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 16_000
        back = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    assert back.tolist() == [0, 16383, -16383, 32767, -32767]


def test_out_of_range_samples_are_clipped_not_wrapped(tmp_path) -> None:
    """A sample past 1.0 wrapping to a large negative is an audible click, and
    on a normalised clip it would only ever happen to the loudest syllable."""
    path = tmp_path / "clip.wav"
    _write_wav(path, np.array([2.0, -2.0], dtype=np.float32))
    with wave.open(str(path)) as handle:
        back = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    assert back.tolist() == [32767, -32767]


# -- engine selection in app.py --------------------------------------------


def test_the_config_selects_the_engine() -> None:
    import app
    from config import load_config

    config = load_config(None)
    config.whisper.engine = "parakeet"
    config.whisper.parakeet_model = "/nonexistent/model.bin"
    worker = app.build_engine(config)
    assert isinstance(worker, ParakeetWorker)
    # Built but not loaded: a bad path must fail at start(), not at import.
    assert worker.model == "/nonexistent/model.bin"


def test_the_default_engine_is_still_faster_whisper() -> None:
    import app
    from config import load_config
    from stt_worker import SttWorker

    config = load_config(None)
    assert config.whisper.engine == "faster-whisper"
    assert isinstance(app.build_engine(config), SttWorker)


def test_an_unknown_engine_is_refused_loudly() -> None:
    """Rather than silently falling back to Whisper, which would make a
    misspelled config look like it worked."""
    import app
    from config import load_config

    config = load_config(None)
    config.whisper.engine = "parakeat"
    with pytest.raises(SystemExit, match="Unknown whisper.engine"):
        app.build_engine(config)


def test_device_cpu_disables_the_gpu() -> None:
    import app
    from config import load_config

    config = load_config(None)
    config.whisper.engine = "parakeet"
    config.whisper.device = "cpu"
    assert app.build_engine(config).use_gpu is False
