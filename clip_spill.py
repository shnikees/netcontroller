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

"""Overflow store for clips waiting on a transcriber that has fallen behind.

The last line of defence. When transmissions arrive faster than Whisper can
process them -- an underpowered box, a busy net, or a model a size too big --
the in-memory queue fills. Rather than dropping audio, clips go to disk as WAVs
and are transcribed during the next lull, or after the net ends.

The trade is explicit: a spilled clip appears in the log **late**. Its timestamp
is still the moment it was transmitted, so the finished log reads in the right
order -- but net control will not see that line while the station is still
talking. Late beats missing for a log that gets filed afterwards; if it starts
happening every net, the model is too big for the hardware and the dashboard
says so.

Clips are written oldest-first and read back oldest-first, so a backlog drains
in transmission order.
"""

from __future__ import annotations

import json
import logging
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


@dataclass
class SpilledClip:
    """A clip recovered from disk, with the metadata needed to log it correctly."""

    audio: np.ndarray
    start_offset_ms: int
    duration_ms: int
    sequence: int
    path: Path
    source: str = ""


class SpillStore:
    """Append clips to disk and read them back oldest-first.

    max_clips bounds the damage when a machine is so far behind that it will
    never catch up: past that, the oldest spilled clip is discarded, and the
    count is surfaced rather than hidden.
    """

    def __init__(self, directory: str | Path, max_clips: int = 500) -> None:
        self.directory = Path(directory)
        self.max_clips = max_clips
        self.spilled = 0
        self.recovered = 0
        self.discarded = 0

    def _ensure_directory(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        audio: np.ndarray,
        start_offset_ms: int,
        duration_ms: int,
        sequence: int,
        source: str = "",
    ) -> Path | None:
        """Write one clip. Returns its path, or None if the write failed.

        A failure here (disk full, read-only media) must not take the pipeline
        down: the clip is lost, which is what would have happened anyway, and
        the net carries on.
        """
        try:
            self._ensure_directory()
            self._enforce_limit()
            path = self.directory / f"clip-{sequence:06d}.wav"
            pcm = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(SAMPLE_RATE)
                handle.writeframes(pcm.tobytes())
            path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "sequence": sequence,
                        "start_offset_ms": start_offset_ms,
                        "duration_ms": duration_ms,
                        "source": source,
                    }
                ),
                encoding="utf-8",
            )
            self.spilled += 1
            return path
        except OSError as exc:
            log.error("Could not spill clip %d to disk: %s", sequence, exc)
            return None

    def pending(self) -> int:
        if not self.directory.exists():
            return 0
        return len(list(self.directory.glob("clip-*.wav")))

    def _paths(self) -> list[Path]:
        if not self.directory.exists():
            return []
        return sorted(self.directory.glob("clip-*.wav"))

    def _enforce_limit(self) -> None:
        paths = self._paths()
        while len(paths) >= self.max_clips:
            oldest = paths.pop(0)
            self._remove(oldest)
            self.discarded += 1
            log.warning(
                "Spill directory full (%d clips); dropped the oldest, %s",
                self.max_clips,
                oldest.name,
            )

    def read_oldest(self) -> SpilledClip | None:
        """Pop the oldest spilled clip, or None when there is nothing waiting."""
        for path in self._paths():
            try:
                with wave.open(str(path), "rb") as handle:
                    frames = handle.readframes(handle.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                meta = self._read_meta(path)
                self._remove(path)
                self.recovered += 1
                return SpilledClip(
                    audio=audio,
                    start_offset_ms=int(meta.get("start_offset_ms", 0)),
                    duration_ms=int(
                        meta.get("duration_ms", len(audio) * 1000 // SAMPLE_RATE)
                    ),
                    sequence=int(meta.get("sequence", 0)),
                    path=path,
                    source=str(meta.get("source", "")),
                )
            except (OSError, wave.Error, ValueError) as exc:
                # A half-written clip (power cut mid-spill) costs that clip only.
                log.warning("Discarding unreadable spilled clip %s: %s", path, exc)
                self._remove(path)
                self.discarded += 1
        return None

    def _read_meta(self, path: Path) -> dict:
        sidecar = path.with_suffix(".json")
        if not sidecar.exists():
            return {}
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _remove(self, path: Path) -> None:
        for target in (path, path.with_suffix(".json")):
            try:
                target.unlink(missing_ok=True)
            except OSError:  # pragma: no cover - best effort
                log.debug("Could not remove %s", target, exc_info=True)

    def clear(self) -> int:
        """Remove everything. Called at startup: last net's backlog is not this
        net's problem, and transcribing it would interleave two sessions."""
        removed = 0
        for path in self._paths():
            self._remove(path)
            removed += 1
        return removed
