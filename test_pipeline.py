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

"""End-to-end pipeline tests with a stubbed transcriber.

The claim under test is the one that motivated splitting VAD from STT: a
transcriber slower than real time must make transcripts arrive *late*, never
leave them *missing*. So these run a real capture -> VAD -> queue -> spill
chain over a synthetic recording, with Whisper replaced by something slow and
deterministic.
"""

from __future__ import annotations

import asyncio
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import app as app_module
from callsign_match import CallsignMatcher, RosterEntry
from config import Config, SourceConfig
from health import HealthFleet
from server import Broadcaster
from transcript_store import TranscriptStore

RATE = 16_000
ROSTER = [RosterEntry("W6ABC", "Alice")]


class SlowStub:
    """Stands in for SttWorker: slow, and reports which clips it saw."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.seen = 0
        self.model_size = "stub"
        self.active_device = "cpu"
        self.active_compute_type = "stub"
        self._lock = threading.Lock()

    def load(self) -> None:
        pass

    def build_prompt(self, terms, lead_in: str = "") -> str:
        self.prompt_terms_used = len(terms)
        self.prompt_terms_offered = len(terms)
        return ", ".join(terms[:10])

    def transcribe(self, audio: np.ndarray, prompt: str | None = None):
        if self.delay:
            time.sleep(self.delay)
        with self._lock:
            self.seen += 1
            index = self.seen
        return _Transcription(f"transmission {index}")


class _Transcription:
    def __init__(self, text: str) -> None:
        self.text = text
        self.confidence = 0.9
        self.language = "en"
        self.no_speech_prob = 0.0
        self.words = []  # no timings, so nothing is ever split


def write_net_wav(path: Path, transmissions: int, seconds: float = 1.0) -> None:
    """Speech-shaped noise bursts separated by silence the VAD will split on."""
    rng = np.random.default_rng(4)
    gap = np.zeros(int(RATE * 1.2), dtype=np.int16)
    parts = [gap]
    for _ in range(transmissions):
        noise = rng.normal(0, 6000, int(RATE * seconds))
        # Amplitude-modulated noise reads as speech to webrtcvad far more
        # reliably than a pure tone does.
        envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 4 * np.arange(len(noise)) / RATE)
        parts.append(np.clip(noise * envelope, -32768, 32767).astype(np.int16))
        parts.append(gap)
    audio = np.concatenate(parts)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(audio.tobytes())


@pytest.fixture
def loop():
    """A real event loop on its own thread, as app.py has at runtime."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


def build(tmp_path, loop, stub: SlowStub, **buffering) -> tuple:
    config = Config()
    config.buffering.spill_dir = str(tmp_path / "spill")
    config.buffering.clip_queue_max = buffering.get("clip_queue_max", 32)
    config.buffering.spill_enabled = buffering.get("spill_enabled", True)
    config.vad.min_clip_ms = 300
    config.vad.silence_ms = 500

    if buffering.get("sources"):
        config.sources = buffering["sources"]

    store = TranscriptStore()
    pipeline = app_module.Pipeline(
        config=config,
        store=store,
        matcher=CallsignMatcher(roster=ROSTER),
        broadcaster=Broadcaster(),
        loop=loop,
        fleet=HealthFleet(),
        wav_path=buffering.get("wav"),
    )
    pipeline.stt = stub
    return pipeline, store


def run_to_completion(pipeline, store, expected: int, timeout: float = 60.0) -> None:
    pipeline.start()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and len(store.entries) < expected:
        time.sleep(0.1)
    pipeline.drain(timeout=10)
    pipeline.stop()


def test_all_transmissions_are_transcribed(tmp_path, loop) -> None:
    wav = tmp_path / "net.wav"
    write_net_wav(wav, transmissions=4)
    stub = SlowStub()
    pipeline, store = build(tmp_path, loop, stub, wav=str(wav))

    run_to_completion(pipeline, store, expected=4)

    assert len(store.entries) == 4


def test_slow_transcriber_loses_nothing(tmp_path, loop) -> None:
    """The regression this whole design exists for.

    With VAD and Whisper on one thread, a transcription slower than the audio
    meant the next transmission was dropped mid-word. Now it is merely late.
    """
    wav = tmp_path / "net.wav"
    write_net_wav(wav, transmissions=6)
    stub = SlowStub(delay=0.4)  # slower than the clips arrive
    pipeline, store = build(tmp_path, loop, stub, wav=str(wav), clip_queue_max=2)

    run_to_completion(pipeline, store, expected=6)

    assert len(store.entries) == 6, "audio was lost while the transcriber was behind"


def test_overflow_spills_to_disk_and_is_recovered(tmp_path, loop) -> None:
    wav = tmp_path / "net.wav"
    write_net_wav(wav, transmissions=6)
    stub = SlowStub(delay=0.4)
    pipeline, store = build(tmp_path, loop, stub, wav=str(wav), clip_queue_max=1)

    run_to_completion(pipeline, store, expected=6)

    assert pipeline.spill.spilled > 0, "queue never overflowed; test proves nothing"
    assert pipeline.spill.recovered == pipeline.spill.spilled
    assert len(store.entries) == 6
    # Everything that spilled is flagged, so the operator knows those lines
    # appeared after the fact.
    assert any(entry.late for entry in store.entries)


def test_log_stays_in_transmission_order_despite_late_arrivals(
    tmp_path, loop
) -> None:
    wav = tmp_path / "net.wav"
    write_net_wav(wav, transmissions=6)
    stub = SlowStub(delay=0.4)
    pipeline, store = build(tmp_path, loop, stub, wav=str(wav), clip_queue_max=1)

    run_to_completion(pipeline, store, expected=6)

    timestamps = [entry.timestamp for entry in store.entries]
    assert timestamps == sorted(timestamps), "a recovered clip landed out of order"


def test_spilling_disabled_drops_instead_of_writing(tmp_path, loop) -> None:
    wav = tmp_path / "net.wav"
    write_net_wav(wav, transmissions=6)
    stub = SlowStub(delay=0.4)
    pipeline, store = build(
        tmp_path, loop, stub, wav=str(wav), clip_queue_max=1, spill_enabled=False
    )

    pipeline.start()
    time.sleep(6)
    pipeline.stop()

    # The documented behaviour of turning spilling off: audio is dropped, and
    # the drop is counted rather than silent.
    assert pipeline.spill.spilled == 0
    assert pipeline.fleet.snapshot()["errors"] > 0


# --------------------------------------------------------------------------
# Multiple receivers
# --------------------------------------------------------------------------


def test_two_sources_are_both_logged_and_tagged(tmp_path, loop) -> None:
    """The repeater and the staging frequency in one log.

    Each source captures independently; the transcriber is shared, so the two
    are serialised rather than competing for the model.
    """
    repeater = tmp_path / "repeater.wav"
    simplex = tmp_path / "simplex.wav"
    write_net_wav(repeater, transmissions=3)
    write_net_wav(simplex, transmissions=2)

    stub = SlowStub()
    pipeline, store = build(
        tmp_path,
        loop,
        stub,
        sources=[
            SourceConfig(name="Repeater", file=str(repeater)),
            SourceConfig(name="Simplex", file=str(simplex)),
        ],
    )

    run_to_completion(pipeline, store, expected=5)

    assert len(store.entries) == 5
    heard = {entry.source for entry in store.entries}
    assert heard == {"Repeater", "Simplex"}
    assert sum(e.source == "Repeater" for e in store.entries) == 3
    assert sum(e.source == "Simplex" for e in store.entries) == 2


def test_sources_are_interleaved_in_time_order(tmp_path, loop) -> None:
    repeater = tmp_path / "repeater.wav"
    simplex = tmp_path / "simplex.wav"
    write_net_wav(repeater, transmissions=3)
    write_net_wav(simplex, transmissions=3)

    pipeline, store = build(
        tmp_path,
        loop,
        SlowStub(),
        sources=[
            SourceConfig(name="Repeater", file=str(repeater)),
            SourceConfig(name="Simplex", file=str(simplex)),
        ],
    )
    run_to_completion(pipeline, store, expected=6)

    timestamps = [e.timestamp for e in store.entries]
    assert timestamps == sorted(timestamps)


def test_one_dead_source_does_not_stop_the_others(tmp_path, loop) -> None:
    """A failed receiver must not take the net down with it."""
    working = tmp_path / "working.wav"
    write_net_wav(working, transmissions=3)

    pipeline, store = build(
        tmp_path,
        loop,
        SlowStub(),
        sources=[
            SourceConfig(name="Broken", device="no-such-device-xyz"),
            SourceConfig(name="Working", file=str(working)),
        ],
    )
    config = pipeline.config
    config.health.restart_capture = False  # fail fast rather than retry forever

    run_to_completion(pipeline, store, expected=3)

    assert len(store.entries) == 3
    assert all(e.source == "Working" for e in store.entries)

    health = pipeline.fleet.snapshot()
    assert health["state"] == "error"
    # The banner has to name the broken one, or the operator does not know
    # which receiver to go and look at.
    assert any("Broken" in issue for issue in health["issues"])
    assert health["sources"]["Working"]["state"] == "ok"


def test_single_source_entries_are_not_tagged(tmp_path, loop) -> None:
    # With one receiver a source column would be noise on every line.
    wav = tmp_path / "net.wav"
    write_net_wav(wav, transmissions=2)
    pipeline, store = build(tmp_path, loop, SlowStub(), wav=str(wav))

    run_to_completion(pipeline, store, expected=2)

    assert all(entry.source == "" for entry in store.entries)


def test_priority_source_is_transcribed_first(tmp_path, loop) -> None:
    """When the transcriber is behind, the repeater must not queue behind a
    staging channel -- the main log is the one people are waiting on."""
    repeater = tmp_path / "repeater.wav"
    simplex = tmp_path / "simplex.wav"
    write_net_wav(repeater, transmissions=3)
    write_net_wav(simplex, transmissions=3)

    stub = SlowStub(delay=0.3)
    pipeline, store = build(
        tmp_path,
        loop,
        stub,
        clip_queue_max=8,
        spill_enabled=False,
        sources=[
            SourceConfig(name="Simplex", file=str(simplex), priority=0),
            SourceConfig(name="Repeater", file=str(repeater), priority=10),
        ],
    )
    run_to_completion(pipeline, store, expected=6)

    order = [e.source for e in sorted(store.entries, key=lambda e: e.id)]
    # Both sources start together and the queue backs up, so once there is a
    # real backlog the higher-priority source should be served first.
    assert "Repeater" in order and "Simplex" in order
    first_repeater = order.index("Repeater")
    last_simplex = len(order) - 1 - order[::-1].index("Simplex")
    assert first_repeater < last_simplex, f"priority ignored: {order}"


def test_per_source_vad_settings_are_applied(tmp_path, loop) -> None:
    # A weak simplex signal needs a gentler VAD than a strong repeater; sharing
    # one global setting is what made per-source overrides necessary.
    pipeline, _ = build(
        tmp_path,
        loop,
        SlowStub(),
        sources=[
            SourceConfig(name="Repeater", aggressiveness=3, silence_ms=600),
            SourceConfig(name="Simplex", aggressiveness=1, silence_ms=1200),
        ],
    )
    by_name = {s.name: s for s in pipeline.sources}
    assert by_name["Repeater"].segmenter.aggressiveness == 3
    assert by_name["Repeater"].segmenter.silence_ms == 600
    assert by_name["Simplex"].segmenter.aggressiveness == 1
    assert by_name["Simplex"].segmenter.silence_ms == 1200


def test_sources_without_overrides_inherit_the_global_vad(tmp_path, loop) -> None:
    pipeline, _ = build(
        tmp_path, loop, SlowStub(), sources=[SourceConfig(name="Repeater")]
    )
    segmenter = pipeline.sources[0].segmenter
    assert segmenter.aggressiveness == pipeline.config.vad.aggressiveness
    assert segmenter.silence_ms == pipeline.config.vad.silence_ms


# --------------------------------------------------------------------------
# Escalation state
#
# The failure these guard against is a line stuck showing "waiting" for a
# second pass that already happened, failed, or was dropped -- which tells net
# control to hold off on a line that is never going to change.
# --------------------------------------------------------------------------


class FakeEscalator:
    """Stands in for the second-pass SttWorker."""

    def __init__(self, text: str = "", raises: bool = False) -> None:
        self.text, self.raises = text, raises

    def build_prompt(self, terms) -> str:
        return ""

    def transcribe(self, audio, prompt=""):
        if self.raises:
            raise RuntimeError("second pass exploded")
        return type("R", (), {"text": self.text, "confidence": 0.9})()


def escalating(tmp_path, loop):
    pipeline, store = build(tmp_path, loop, SlowStub())
    pipeline.config.escalation.enabled = True
    pipeline.config.escalation.on_unmatched = True
    return pipeline, store


def unsure_entry(store):
    from datetime import datetime

    return store.add(
        started_at=datetime(2026, 4, 1, 19, 0, 0),
        matched=False,
        matched_callsign=None,
        operator_name="",
        raw_text="something unclear",
        confidence=0.3,
        match_score=0.0,
        clip_duration=3.0,
        candidate="W6AB",
    )


def test_a_queued_line_is_marked_as_waiting(tmp_path, loop) -> None:
    pipeline, store = escalating(tmp_path, loop)
    entry = unsure_entry(store)
    pipeline._maybe_escalate(entry, np.zeros(16_000, dtype=np.float32))

    assert entry.escalation_pending is True
    assert entry.escalated is False  # not the same statement


def test_an_improved_line_stops_waiting_and_says_it_was_re_transcribed(
    tmp_path, loop
) -> None:
    pipeline, store = escalating(tmp_path, loop)
    entry = unsure_entry(store)
    pipeline._maybe_escalate(entry, np.zeros(16_000, dtype=np.float32))
    pipeline._escalator = FakeEscalator("whiskey six alpha bravo charlie")

    assert pipeline._escalate_one() is True
    assert entry.escalated is True
    assert entry.escalation_pending is False
    assert entry.matched_callsign == "W6ABC"


def test_a_second_pass_that_gains_nothing_still_clears_the_waiting_mark(
    tmp_path, loop
) -> None:
    pipeline, store = escalating(tmp_path, loop)
    entry = unsure_entry(store)
    pipeline._maybe_escalate(entry, np.zeros(16_000, dtype=np.float32))
    pipeline._escalator = FakeEscalator("still nothing useful here")

    pipeline._escalate_one()
    assert entry.escalation_pending is False
    # It genuinely was not re-transcribed into anything better, so it must not
    # claim to have been.
    assert entry.escalated is False


def test_a_failing_second_pass_does_not_strand_the_line(tmp_path, loop) -> None:
    pipeline, store = escalating(tmp_path, loop)
    entry = unsure_entry(store)
    pipeline._maybe_escalate(entry, np.zeros(16_000, dtype=np.float32))
    pipeline._escalator = FakeEscalator(raises=True)

    pipeline._escalate_one()
    assert entry.escalation_pending is False


def test_a_clip_dropped_from_a_full_queue_stops_claiming_to_be_waiting(
    tmp_path, loop
) -> None:
    pipeline, store = escalating(tmp_path, loop)
    pipeline._escalate = __import__("collections").deque(maxlen=2)
    audio = np.zeros(16_000, dtype=np.float32)

    first, second, third = (unsure_entry(store) for _ in range(3))
    for entry in (first, second, third):
        pipeline._maybe_escalate(entry, audio)

    # The oldest was pushed out to make room, so it will never be re-transcribed.
    assert first.escalation_pending is False
    assert second.escalation_pending is True
    assert third.escalation_pending is True


# --------------------------------------------------------------------------
# Prompt echo reaching the log
#
# hallucination.py was written, tested and documented a week before anything
# imported it, so the live pipeline went on logging fabricated callsigns while
# the measurements said not to. These cover the wiring rather than the rule.
# --------------------------------------------------------------------------


class EchoStub:
    """A transcriber that reads the prompt back, as Whisper does on a dead clip.

    Carries the attributes the pipeline reads off its engine -- the prompt
    counters among them -- because `Pipeline` reports them on the status strip
    and a stub without them fails inside the try/except that is there to keep
    one bad clip from killing the net, which turns a missing attribute into a
    silently empty log.
    """

    prompt_terms_used = 0
    prompt_terms_offered = 0
    active_device = "cpu"
    active_compute_type = "int8"
    model_size = "base"

    def __init__(self, text: str) -> None:
        self.text = text

    def transcribe(self, audio, prompt=""):
        return SimpleNamespace(text=self.text, confidence=0.9, words=[], language="en")

    def build_prompt(self, terms, lead_in=""):
        return ""

    def build_bias(self, terms):
        return ""


def _clip(seconds: float = 3.0):
    return SimpleNamespace(
        audio=np.zeros(int(16_000 * seconds), dtype=np.float32),
        start_offset_ms=0,
        duration_ms=int(seconds * 1000),
        sequence=1,
        source="",
    )


def echo_pipeline(tmp_path, loop, text):
    pipeline, store = build(tmp_path, loop, SlowStub())
    pipeline.matcher = CallsignMatcher(
        roster=[RosterEntry("W6ABC"), RosterEntry("K7XYZ"), RosterEntry("N5DEF")]
    )
    pipeline.stt = EchoStub(text)
    return pipeline, store


def test_a_recited_prompt_names_nobody(tmp_path, loop) -> None:
    pipeline, store = echo_pipeline(tmp_path, loop, "W6ABC, K7XYZ, N5DEF.")
    pipeline._handle_clip(_clip())

    assert len(store.entries) == 1, "the transcript should still be logged"
    entry = store.entries[0]
    assert entry.matched is False
    assert entry.matched_callsign is None
    assert "prompt echo" in entry.unmatched_reason


def test_the_transcript_survives_even_though_the_callsigns_do_not(tmp_path, loop) -> None:
    """Dropping the line entirely would hide it from the operator, who may
    recognise something in the text that the matcher cannot."""
    pipeline, store = echo_pipeline(tmp_path, loop, "W6ABC, K7XYZ, N5DEF.")
    pipeline._handle_clip(_clip())
    assert store.entries[0].raw_text == "W6ABC, K7XYZ, N5DEF."


def test_a_real_check_in_is_untouched(tmp_path, loop) -> None:
    pipeline, store = echo_pipeline(
        tmp_path, loop, "Net control, this is whiskey six alpha bravo charlie, no traffic"
    )
    pipeline._handle_clip(_clip())

    entry = store.entries[0]
    assert entry.matched is True
    assert entry.matched_callsign == "W6ABC"
    assert entry.unmatched_reason == ""


def test_an_echo_is_not_split_into_several_transmissions(tmp_path, loop) -> None:
    """Splitting on invented callsigns would turn one junk clip into three
    lines, each looking like its own station."""
    pipeline, store = echo_pipeline(tmp_path, loop, "W6ABC, K7XYZ, N5DEF.")
    pipeline.config.split.enabled = True
    pipeline._handle_clip(_clip())
    assert len(store.entries) == 1


# --------------------------------------------------------------------------
# How the roster reaches Whisper
# --------------------------------------------------------------------------


class RecordingModel:
    """Captures the keyword arguments faster-whisper would have received."""

    def __init__(self) -> None:
        self.calls = []

    def transcribe(self, audio, **kw):
        self.calls.append(kw)
        return iter(()), SimpleNamespace(language="en")


def _worker(mode):
    from stt_worker import SttWorker

    worker = SttWorker(bias_mode=mode, condition_audio=False)
    worker._model = RecordingModel()
    return worker


def test_hotwords_mode_does_not_also_send_a_prompt() -> None:
    """Sending both measured worse than either alone, so the two modes must be
    exclusive rather than additive."""
    worker = _worker("hotwords")
    worker.transcribe(np.zeros(16_000, dtype=np.float32), prompt="W6ABC, K7XYZ")
    call = worker._model.calls[0]
    assert call["hotwords"] == "W6ABC, K7XYZ"
    assert call["initial_prompt"] is None


def test_prompt_mode_is_the_old_behaviour_exactly() -> None:
    worker = _worker("prompt")
    worker.transcribe(np.zeros(16_000, dtype=np.float32), prompt="Net check-ins. W6ABC.")
    call = worker._model.calls[0]
    assert call["initial_prompt"] == "Net check-ins. W6ABC."
    assert call["hotwords"] is None


def test_hotwords_carry_callsigns_without_the_alphabet() -> None:
    """bias_terms offers the phonetic alphabet as well. It earns its place in a
    prompt and is untested as a hotword, so the measured configuration --
    callsigns only -- is what gets reproduced."""
    from stt_worker import SttWorker

    worker = SttWorker(bias_mode="hotwords")
    worker._model = RecordingModel()
    worker._tokenizer_cache = None
    built = worker.build_hotwords(["W6ABC", "alpha", "bravo", "K7XYZ", "niner"])
    assert "W6ABC" in built and "K7XYZ" in built
    assert "alpha" not in built and "niner" not in built
