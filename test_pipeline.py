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

import numpy as np
import pytest

import app as app_module
from callsign_match import CallsignMatcher, RosterEntry
from config import Config
from health import HealthMonitor
from server import Broadcaster
from transcript_store import TranscriptStore

RATE = 16_000
ROSTER = [RosterEntry("W6ABC", "Alice")]


class SlowStub:
    """Stands in for SttWorker: slow, and reports which clips it saw."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.seen = 0
        self._lock = threading.Lock()

    def load(self) -> None:
        pass

    def transcribe(self, audio: np.ndarray):
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

    store = TranscriptStore()
    pipeline = app_module.Pipeline(
        config=config,
        store=store,
        matcher=CallsignMatcher(roster=ROSTER),
        broadcaster=Broadcaster(),
        loop=loop,
        health=HealthMonitor(),
        wav_path=buffering["wav"],
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
    assert pipeline.health.snapshot().errors > 0
