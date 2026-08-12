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

"""Health tracking for the capture pipeline.

The failure this exists to catch is the quiet one. A crash is obvious -- the
process is gone. What actually bites at a net control table is the pipeline
that stays up while producing nothing: the SDR app was closed, the loopback
sink got repointed, the squelch stayed shut, the machine fell behind. The
dashboard looks fine, and forty minutes of the net go unlogged.

So the monitor watches for *silence where there should be sound*, at three
levels:

    frames arriving   -> is the audio device still delivering?
    signal in frames  -> is anything actually on the channel?
    clips completing  -> is the VAD/STT chain still producing?

The capture thread reports events; the asyncio loop reads snapshots. Times come
from an injectable clock so the state machine can be tested without sleeping.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

OK = "ok"
WARNING = "warning"
ERROR = "error"

_SEVERITY = {OK: 0, WARNING: 1, ERROR: 2}


def _first_line(message: str) -> str:
    """First line of an exception message, tidied for a one-line banner.

    Capture errors carry multi-line remediation hints for the log; the banner
    gets the headline, without the colon left dangling where the list was.
    """
    if not message:
        return ""
    return message.splitlines()[0].strip().rstrip(":").strip()


@dataclass(frozen=True)
class Health:
    """A point-in-time view of the pipeline, safe to serialise to the dashboard."""

    state: str
    issues: list[str]
    """Operator-facing descriptions, worst first. Empty when state is ok."""
    capturing: bool
    uptime_s: float
    frames: int
    clips: int
    transcriptions: int
    errors: int
    overflows: int
    signal_rms: float
    """Recent audio level in int16 units. 0 means the device is delivering silence."""
    seconds_since_frame: float | None
    seconds_since_clip: float | None
    last_transcribe_s: float
    """How long the last transcription took; a proxy for keeping up."""
    backlog: int
    """Clips waiting in memory for the transcriber."""
    spilled: int
    """Clips written to disk this session because the queue was full."""
    spill_pending: int
    """Spilled clips still waiting to be transcribed."""
    last_error: str

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "issues": self.issues,
            "capturing": self.capturing,
            "uptime_s": round(self.uptime_s, 1),
            "frames": self.frames,
            "clips": self.clips,
            "transcriptions": self.transcriptions,
            "errors": self.errors,
            "overflows": self.overflows,
            "signal_rms": round(self.signal_rms, 1),
            "seconds_since_frame": (
                None
                if self.seconds_since_frame is None
                else round(self.seconds_since_frame, 1)
            ),
            "seconds_since_clip": (
                None
                if self.seconds_since_clip is None
                else round(self.seconds_since_clip, 1)
            ),
            "last_transcribe_s": round(self.last_transcribe_s, 2),
            "backlog": self.backlog,
            "spilled": self.spilled,
            "spill_pending": self.spill_pending,
            "last_error": self.last_error,
        }


@dataclass
class HealthMonitor:
    """Thread-safe pipeline health.

    stall_after_s: no audio frames for this long is an error -- the device
        stopped delivering, which on PulseAudio usually means the sink went
        away underneath us.
    silence_after_s: frames arriving but no signal in them for this long is a
        warning. Long enough not to fire during ordinary quiet stretches of a
        net; short enough to catch a closed squelch before the net ends.
    silence_rms: below this, a frame counts as dead air rather than quiet
        speech. Receiver hiss normally sits well above it.
    """

    stall_after_s: float = 5.0
    silence_after_s: float = 300.0
    silence_rms: float = 15.0
    clock: Callable[[], float] = time.monotonic

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        now = self.clock()
        self._started = now
        self._capturing = False
        self._finished = False
        self._last_frame: float | None = None
        self._last_clip: float | None = None
        self._last_signal: float | None = None
        self._frames = 0
        self._clips = 0
        self._transcriptions = 0
        self._errors = 0
        self._overflows = 0
        self._signal_rms = 0.0
        self._last_transcribe_s = 0.0
        self._backlog = 0
        self._spilled = 0
        self._spill_pending = 0
        self._last_error = ""

    # -- reported by the capture thread ------------------------------------

    def capture_started(self) -> None:
        with self._lock:
            self._capturing = True
            self._finished = False
            self._last_error = ""
            # Give the device a fresh grace period; otherwise a restart looks
            # stalled for as long as it takes the first frame to arrive.
            self._last_frame = self.clock()
            self._last_signal = self._last_frame

    def capture_stopped(self) -> None:
        with self._lock:
            self._capturing = False

    def capture_finished(self) -> None:
        """Capture ended because the source ran out, not because it broke.

        A finished file replay is a success; alarming about it would train the
        operator to ignore the banner during the tuning workflow that is
        supposed to precede going live.
        """
        with self._lock:
            self._capturing = False
            self._finished = True

    def capture_failed(self, message: str) -> None:
        with self._lock:
            self._capturing = False
            self._errors += 1
            self._last_error = _first_line(message) or "capture failed"

    def note_frame(self, rms: float) -> None:
        now = self.clock()
        with self._lock:
            self._frames += 1
            self._last_frame = now
            self._signal_rms = rms
            if rms >= self.silence_rms:
                self._last_signal = now

    def note_clip(self) -> None:
        with self._lock:
            self._clips += 1
            self._last_clip = self.clock()

    def note_transcription(self, seconds: float, backlog: int = 0) -> None:
        with self._lock:
            self._transcriptions += 1
            self._last_transcribe_s = seconds
            self._backlog = backlog

    def note_spill(self, spilled: int, pending: int) -> None:
        with self._lock:
            self._spilled = spilled
            self._spill_pending = pending

    def note_overflows(self, total: int) -> None:
        with self._lock:
            self._overflows = total

    def note_error(self, message: str) -> None:
        with self._lock:
            self._errors += 1
            self._last_error = _first_line(message) or "error"

    # -- read by the event loop --------------------------------------------

    def snapshot(self) -> Health:
        now = self.clock()
        with self._lock:
            since_frame = None if self._last_frame is None else now - self._last_frame
            since_clip = None if self._last_clip is None else now - self._last_clip
            since_signal = (
                None if self._last_signal is None else now - self._last_signal
            )
            issues: list[tuple[str, str]] = []

            if not self._capturing and not self._finished:
                issues.append(
                    (
                        ERROR,
                        f"Audio capture is not running: {self._last_error}"
                        if self._last_error
                        else "Audio capture is not running",
                    )
                )
            elif (
                self._capturing
                and since_frame is not None
                and since_frame > self.stall_after_s
            ):
                issues.append(
                    (
                        ERROR,
                        f"No audio from the device for {since_frame:.0f}s "
                        "-- check SDR++/GQRX and the loopback sink",
                    )
                )
            elif (
                self._capturing
                and since_signal is not None
                and since_signal > self.silence_after_s
            ):
                issues.append(
                    (
                        WARNING,
                        f"Audio is silent ({self._signal_rms:.0f} RMS) for "
                        f"{since_signal / 60:.0f} min -- check squelch and "
                        "the SDR app's output level",
                    )
                )

            if self._spill_pending:
                issues.append(
                    (
                        WARNING,
                        f"Transcriber is behind: {self._spill_pending} clip(s) "
                        "buffered to disk, catching up between transmissions "
                        "-- a smaller Whisper model would keep up live",
                    )
                )

            if self._overflows:
                issues.append(
                    (
                        WARNING,
                        f"Dropped {self._overflows} audio block(s) -- the "
                        "machine is behind; try a smaller Whisper model",
                    )
                )

            issues.sort(key=lambda pair: -_SEVERITY[pair[0]])
            state = issues[0][0] if issues else OK

            return Health(
                state=state,
                issues=[message for _, message in issues],
                capturing=self._capturing,
                uptime_s=now - self._started,
                frames=self._frames,
                clips=self._clips,
                transcriptions=self._transcriptions,
                errors=self._errors,
                overflows=self._overflows,
                signal_rms=self._signal_rms,
                seconds_since_frame=since_frame,
                seconds_since_clip=since_clip,
                last_transcribe_s=self._last_transcribe_s,
                backlog=self._backlog,
                spilled=self._spilled,
                spill_pending=self._spill_pending,
                last_error=self._last_error,
            )


@dataclass
class HealthFleet:
    """Per-source monitors plus one combined verdict for the dashboard.

    With two receivers, "the pipeline is unhealthy" is not actionable -- the
    operator needs to know *which* one to go and look at. So issues are
    prefixed with the source name whenever more than one source exists, and the
    overall state is the worst of them.
    """

    stall_after_s: float = 5.0
    silence_after_s: float = 300.0
    silence_rms: float = 15.0
    clock: Callable[[], float] = time.monotonic
    _monitors: dict[str, HealthMonitor] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def monitor(self, name: str) -> HealthMonitor:
        with self._lock:
            if name not in self._monitors:
                self._monitors[name] = HealthMonitor(
                    stall_after_s=self.stall_after_s,
                    silence_after_s=self.silence_after_s,
                    silence_rms=self.silence_rms,
                    clock=self.clock,
                )
            return self._monitors[name]

    @property
    def names(self) -> list[str]:
        with self._lock:
            return list(self._monitors)

    def note_spill(self, spilled: int, pending: int) -> None:
        for monitor in list(self._monitors.values()):
            monitor.note_spill(spilled, pending)

    def note_error(self, message: str) -> None:
        for monitor in list(self._monitors.values()):
            monitor.note_error(message)

    def snapshot(self) -> dict:
        with self._lock:
            monitors = dict(self._monitors)
        if not monitors:
            return {"state": ERROR, "issues": ["No audio sources configured"], "sources": {}}

        per_source = {name: m.snapshot() for name, m in monitors.items()}
        multi = len(per_source) > 1
        state = OK
        issues: list[str] = []
        for name, snapshot in per_source.items():
            if _SEVERITY[snapshot.state] > _SEVERITY[state]:
                state = snapshot.state
            for issue in snapshot.issues:
                issues.append(f"{name}: {issue}" if multi else issue)

        combined = {
            "state": state,
            "issues": issues,
            "sources": {name: s.to_dict() for name, s in per_source.items()},
        }
        # Totals across sources, so the header can show one set of numbers.
        for key in ("frames", "clips", "transcriptions", "errors", "overflows"):
            combined[key] = sum(getattr(s, key) for s in per_source.values())
        combined["spill_pending"] = max(
            (s.spill_pending for s in per_source.values()), default=0
        )
        combined["spilled"] = max((s.spilled for s in per_source.values()), default=0)
        combined["capturing"] = any(s.capturing for s in per_source.values())
        combined["signal_rms"] = max(
            (s.signal_rms for s in per_source.values()), default=0.0
        )
        return combined
