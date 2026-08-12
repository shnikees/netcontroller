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

"""Tests for the ring buffer and the disk spill.

Between them these are what stop audio being lost when Whisper cannot keep up,
so the tests are about survival under backlog, not about happy paths.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from clip_spill import SpillStore
from ring_buffer import RingBuffer


def ramp(start: int, count: int) -> np.ndarray:
    """Distinguishable samples, so lost or reordered data is visible."""
    return np.arange(start, start + count, dtype=np.int16)


# --------------------------------------------------------------------------
# Ring buffer
# --------------------------------------------------------------------------


def test_write_then_read_round_trips() -> None:
    ring = RingBuffer(1000)
    ring.write(ramp(0, 100))
    assert np.array_equal(ring.read(100), ramp(0, 100))


def test_reads_in_order_across_many_writes() -> None:
    ring = RingBuffer(1000)
    for i in range(5):
        ring.write(ramp(i * 100, 100))
    out = np.concatenate([ring.read(100) for _ in range(5)])
    assert np.array_equal(out, ramp(0, 500))


def test_wraps_around_the_end_without_losing_samples() -> None:
    # Deliberately not a divisor of the capacity, so writes straddle the seam.
    ring = RingBuffer(256)
    written = []
    read = []
    for i in range(20):
        block = ramp(i * 100, 100)
        ring.write(block)
        written.append(block)
        read.append(ring.read(100))
    assert np.array_equal(np.concatenate(read), np.concatenate(written))
    assert ring.dropped == 0


def test_partial_reads_leave_the_rest() -> None:
    ring = RingBuffer(1000)
    ring.write(ramp(0, 100))
    assert np.array_equal(ring.read(40), ramp(0, 40))
    assert np.array_equal(ring.read(60), ramp(40, 60))


def test_overrun_drops_the_oldest_and_counts_it() -> None:
    ring = RingBuffer(100)
    ring.write(ramp(0, 80))
    ring.write(ramp(80, 60))  # 40 samples over capacity

    assert ring.dropped == 40
    # What survives is the newest audio: the old frames belong to a
    # transmission that was already truncated.
    assert np.array_equal(ring.read(100), ramp(40, 100))


def test_a_block_larger_than_the_buffer_keeps_its_tail() -> None:
    ring = RingBuffer(100)
    ring.write(ramp(0, 250))
    assert np.array_equal(ring.read(100), ramp(150, 100))


def test_read_blocks_until_data_arrives() -> None:
    ring = RingBuffer(1000)
    result: list[np.ndarray | None] = []

    reader = threading.Thread(target=lambda: result.append(ring.read(50)))
    reader.start()
    ring.write(ramp(0, 50))
    reader.join(timeout=2)

    assert not reader.is_alive()
    assert np.array_equal(result[0], ramp(0, 50))


def test_close_releases_a_blocked_reader() -> None:
    # Without this, stopping the app would hang on the capture thread.
    ring = RingBuffer(1000)
    result: list[np.ndarray | None] = []

    reader = threading.Thread(target=lambda: result.append(ring.read(50)))
    reader.start()
    ring.close()
    reader.join(timeout=2)

    assert not reader.is_alive()
    assert result[0] is None


def test_concurrent_producer_and_consumer_lose_nothing() -> None:
    total = 200
    block = 128
    # Capacity exceeds everything written, so the consumer falling behind
    # cannot cause a drop -- this test is about ordering and integrity under
    # concurrency, not about the overrun policy (covered above).
    ring = RingBuffer(total * block * 2)
    received: list[np.ndarray] = []

    def consume() -> None:
        for _ in range(total):
            # A timeout rather than an unbounded wait: if the buffer ever
            # deadlocks, the test should fail rather than hang the suite.
            chunk = ring.read(block, timeout=5)
            if chunk is None:
                return
            received.append(chunk)

    reader = threading.Thread(target=consume)
    reader.start()
    for i in range(total):
        ring.write(ramp(i * block, block))
    reader.join(timeout=10)

    assert ring.dropped == 0
    assert np.array_equal(np.concatenate(received), ramp(0, total * block))


def test_fill_reports_the_backlog() -> None:
    ring = RingBuffer(1000)
    assert ring.fill == 0.0
    ring.write(ramp(0, 500))
    assert ring.fill == pytest.approx(0.5)


def test_writing_nothing_is_harmless() -> None:
    ring = RingBuffer(100)
    assert ring.write(np.zeros(0, dtype=np.int16)) == 0


# --------------------------------------------------------------------------
# Disk spill
# --------------------------------------------------------------------------


def clip_audio(value: float, samples: int = 1600) -> np.ndarray:
    return np.full(samples, value, dtype=np.float32)


@pytest.fixture
def spill(tmp_path) -> SpillStore:
    return SpillStore(tmp_path / "spill")


def test_spilled_clip_round_trips(spill: SpillStore) -> None:
    spill.write(clip_audio(0.5), start_offset_ms=1234, duration_ms=100, sequence=7)

    recovered = spill.read_oldest()
    assert recovered is not None
    assert recovered.sequence == 7
    assert recovered.start_offset_ms == 1234
    assert recovered.duration_ms == 100
    # 16-bit quantisation, so approximate rather than exact.
    assert recovered.audio.mean() == pytest.approx(0.5, abs=0.001)


def test_backlog_drains_oldest_first(spill: SpillStore) -> None:
    # A net log read out of order is worse than a late one.
    for sequence in (1, 2, 3):
        spill.write(clip_audio(0.1 * sequence), 0, 100, sequence)

    assert [spill.read_oldest().sequence for _ in range(3)] == [1, 2, 3]


def test_reading_removes_the_clip(spill: SpillStore) -> None:
    spill.write(clip_audio(0.5), 0, 100, 1)
    assert spill.pending() == 1
    spill.read_oldest()
    assert spill.pending() == 0
    assert spill.read_oldest() is None


def test_empty_spill_is_not_an_error(spill: SpillStore) -> None:
    assert spill.pending() == 0
    assert spill.read_oldest() is None


def test_unreadable_clip_is_discarded_not_fatal(spill: SpillStore, tmp_path) -> None:
    # A clip half-written when the power went out must not wedge the backlog.
    spill.write(clip_audio(0.5), 0, 100, 1)
    (tmp_path / "spill" / "clip-000002.wav").write_bytes(b"not a wav")

    assert spill.read_oldest().sequence == 1
    assert spill.read_oldest() is None
    assert spill.discarded == 1


def test_missing_metadata_falls_back_to_the_audio(spill: SpillStore, tmp_path) -> None:
    spill.write(clip_audio(0.5, samples=1600), 0, 100, 3)
    (tmp_path / "spill" / "clip-000003.json").unlink()

    recovered = spill.read_oldest()
    assert recovered is not None
    assert recovered.duration_ms == 100  # 1600 samples at 16 kHz


def test_disk_backlog_is_bounded(tmp_path) -> None:
    spill = SpillStore(tmp_path / "spill", max_clips=3)
    for sequence in range(1, 6):
        spill.write(clip_audio(0.1), 0, 100, sequence)

    assert spill.pending() <= 3
    assert spill.discarded == 2
    # The newest clips are the ones kept.
    assert spill.read_oldest().sequence == 3


def test_clear_removes_a_previous_session(spill: SpillStore) -> None:
    for sequence in (1, 2):
        spill.write(clip_audio(0.1), 0, 100, sequence)
    assert spill.clear() == 2
    assert spill.pending() == 0


def test_write_failure_is_reported_not_raised(tmp_path) -> None:
    # Read-only media: the clip is lost, but the net carries on.
    target = tmp_path / "file-in-the-way"
    target.write_text("not a directory")
    spill = SpillStore(target / "spill")
    assert spill.write(clip_audio(0.5), 0, 100, 1) is None
