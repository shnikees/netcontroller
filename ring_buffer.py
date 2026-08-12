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

"""Pre-allocated ring buffer between the audio callback and the VAD.

The audio callback runs on PortAudio's thread and must return quickly. The
previous design built a `bytes` object per frame and pushed it onto a Queue,
which allocates on every block -- 33 allocations a second, plus the garbage to
collect afterwards. On a loaded Pi that is a recipe for xruns, which arrive as
clicks the VAD then mistakes for speech.

Here the storage is allocated once and reused. The writer copies into it, the
reader copies out; neither grows the heap.

An honest caveat: this is Python, so the callback still takes the GIL and
nothing here is hard real-time. Removing the per-block allocation removes the
part that was ours to remove.

**Overrun policy: drop the oldest.** If the reader falls behind, the newest
audio is the audio worth keeping -- an old frame belongs to a transmission that
was already truncated, so overwriting it loses nothing that was salvageable.
Drops are counted, never silent.
"""

from __future__ import annotations

import threading

import numpy as np


class RingBuffer:
    """Single-producer, single-consumer ring buffer of int16 samples."""

    def __init__(self, capacity: int) -> None:
        if capacity < 2:
            raise ValueError("capacity must be at least 2 samples")
        self._buffer = np.zeros(capacity, dtype=np.int16)
        self._capacity = capacity
        self._write = 0
        self._count = 0
        self._condition = threading.Condition()
        self._closed = False
        self.dropped = 0
        """Samples overwritten before the reader got to them."""

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        with self._condition:
            return self._count

    @property
    def fill(self) -> float:
        """How full the buffer is, 0.0-1.0. The backlog gauge."""
        with self._condition:
            return self._count / self._capacity

    # -- producer (audio callback) -----------------------------------------

    def write(self, samples: np.ndarray) -> int:
        """Append samples, overwriting the oldest if full. Returns samples dropped."""
        count = len(samples)
        if count == 0:
            return 0

        # A block larger than the whole buffer can only keep its tail.
        if count >= self._capacity:
            samples = samples[-self._capacity :]
            count = len(samples)

        with self._condition:
            end = self._write + count
            if end <= self._capacity:
                self._buffer[self._write : end] = samples
            else:
                split = self._capacity - self._write
                self._buffer[self._write :] = samples[:split]
                self._buffer[: end - self._capacity] = samples[split:]
            self._write = end % self._capacity

            overflow = max(0, self._count + count - self._capacity)
            if overflow:
                self.dropped += overflow
            self._count = min(self._capacity, self._count + count)
            self._condition.notify_all()
        return overflow

    # -- consumer (VAD thread) ---------------------------------------------

    def read(self, count: int, timeout: float | None = None) -> np.ndarray | None:
        """Block until `count` samples are available and return them.

        Returns None once the buffer is closed and drained, which is the
        signal for the reader loop to stop.
        """
        with self._condition:
            while self._count < count and not self._closed:
                if not self._condition.wait(timeout=timeout):
                    return None  # timed out; caller decides what that means
            if self._count < count:
                return None  # closed and short

            start = (self._write - self._count) % self._capacity
            end = start + count
            if end <= self._capacity:
                out = self._buffer[start:end].copy()
            else:
                out = np.concatenate(
                    [self._buffer[start:], self._buffer[: end - self._capacity]]
                )
            self._count -= count
            return out

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def clear(self) -> None:
        """Discard buffered audio -- used when reopening a device mid-net."""
        with self._condition:
            self._count = 0
            self._write = 0
