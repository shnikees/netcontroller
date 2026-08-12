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

"""Tests for the health state machine.

Time is injected, so a five-minute silence is tested in microseconds and the
suite never sleeps.
"""

from __future__ import annotations

import pytest

from health import ERROR, OK, WARNING, HealthMonitor


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def monitor(clock: FakeClock) -> HealthMonitor:
    return HealthMonitor(
        stall_after_s=5.0, silence_after_s=300.0, silence_rms=15.0, clock=clock
    )


LOUD = 500.0
QUIET = 2.0


# --------------------------------------------------------------------------
# Baseline
# --------------------------------------------------------------------------


def test_starts_unhealthy_until_capture_begins(monitor: HealthMonitor) -> None:
    # Nothing has opened the device yet, and saying "ok" then would be a lie.
    snapshot = monitor.snapshot()
    assert snapshot.state == ERROR
    assert not snapshot.capturing


def test_healthy_once_frames_flow(monitor: HealthMonitor) -> None:
    monitor.capture_started()
    monitor.note_frame(LOUD)
    assert monitor.snapshot().state == OK


# --------------------------------------------------------------------------
# The device stops delivering
# --------------------------------------------------------------------------


def test_stalled_audio_is_an_error(monitor: HealthMonitor, clock: FakeClock) -> None:
    monitor.capture_started()
    monitor.note_frame(LOUD)
    clock.advance(6.0)

    snapshot = monitor.snapshot()
    assert snapshot.state == ERROR
    assert "No audio from the device" in snapshot.issues[0]


def test_brief_gap_is_not_a_stall(monitor: HealthMonitor, clock: FakeClock) -> None:
    monitor.capture_started()
    monitor.note_frame(LOUD)
    clock.advance(2.0)
    assert monitor.snapshot().state == OK


def test_capture_failure_reports_the_reason(monitor: HealthMonitor) -> None:
    monitor.capture_started()
    monitor.capture_failed("Could not open audio input net_sink.monitor: no device\nhint")
    snapshot = monitor.snapshot()
    assert snapshot.state == ERROR
    # Only the first line: the banner is one line tall, the hints are in the log.
    assert "no device" in snapshot.issues[0]
    assert "hint" not in snapshot.issues[0]
    assert snapshot.errors == 1


def test_restart_clears_the_error(monitor: HealthMonitor) -> None:
    monitor.capture_failed("device vanished")
    assert monitor.snapshot().state == ERROR

    monitor.capture_started()
    monitor.note_frame(LOUD)
    assert monitor.snapshot().state == OK


def test_restart_grace_period_is_not_read_as_a_stall(
    monitor: HealthMonitor, clock: FakeClock
) -> None:
    monitor.capture_started()
    monitor.note_frame(LOUD)
    clock.advance(600.0)  # long outage
    monitor.capture_failed("device vanished")
    monitor.capture_started()  # reopened, first frame not in yet
    assert monitor.snapshot().state == OK


# --------------------------------------------------------------------------
# The device delivers, but there is nothing on it
# --------------------------------------------------------------------------


def test_prolonged_silence_is_a_warning(
    monitor: HealthMonitor, clock: FakeClock
) -> None:
    monitor.capture_started()
    for _ in range(10):
        clock.advance(40.0)
        monitor.note_frame(QUIET)

    snapshot = monitor.snapshot()
    assert snapshot.state == WARNING
    assert "silent" in snapshot.issues[0]
    # Frames are still arriving, so this must not read as a dead device.
    assert snapshot.capturing


def test_quiet_stretch_within_a_net_is_fine(
    monitor: HealthMonitor, clock: FakeClock
) -> None:
    monitor.capture_started()
    monitor.note_frame(LOUD)
    clock.advance(120.0)  # two minutes between check-ins is ordinary
    monitor.note_frame(QUIET)
    assert monitor.snapshot().state == OK


def test_signal_resets_the_silence_timer(
    monitor: HealthMonitor, clock: FakeClock
) -> None:
    monitor.capture_started()
    clock.advance(280.0)
    monitor.note_frame(LOUD)  # somebody keyed up
    clock.advance(100.0)
    monitor.note_frame(QUIET)
    assert monitor.snapshot().state == OK


def test_a_dead_device_outranks_silence(
    monitor: HealthMonitor, clock: FakeClock
) -> None:
    monitor.capture_started()
    monitor.note_frame(QUIET)
    clock.advance(600.0)
    snapshot = monitor.snapshot()
    # Both conditions hold; the operator needs the actionable one first.
    assert snapshot.state == ERROR
    assert "No audio from the device" in snapshot.issues[0]


def test_finished_replay_is_not_an_error(
    monitor: HealthMonitor, clock: FakeClock
) -> None:
    # A file replay reaching its end is success. Alarming about it would teach
    # the operator to ignore the banner during exactly the tuning workflow that
    # is meant to happen before going live.
    monitor.capture_started()
    monitor.note_frame(LOUD)
    monitor.capture_finished()
    assert monitor.snapshot().state == OK

    # And it stays fine as time passes: a finished source is not a stalled one,
    # which is what the stall and silence timers would otherwise conclude.
    clock.advance(600.0)
    assert monitor.snapshot().state == OK


def test_a_broken_capture_is_still_an_error_after_a_finish(
    monitor: HealthMonitor,
) -> None:
    monitor.capture_started()
    monitor.capture_finished()
    monitor.capture_started()
    monitor.capture_failed("device vanished")
    assert monitor.snapshot().state == ERROR


# --------------------------------------------------------------------------
# Falling behind
# --------------------------------------------------------------------------


def test_dropped_audio_is_a_warning(monitor: HealthMonitor) -> None:
    monitor.capture_started()
    monitor.note_frame(LOUD)
    monitor.note_overflows(12)

    snapshot = monitor.snapshot()
    assert snapshot.state == WARNING
    assert "Dropped 12" in snapshot.issues[0]
    assert "smaller Whisper model" in snapshot.issues[0]


def test_counters_and_timings_are_reported(monitor: HealthMonitor) -> None:
    monitor.capture_started()
    monitor.note_frame(LOUD)
    monitor.note_clip()
    monitor.note_transcription(0.42)

    snapshot = monitor.snapshot()
    assert snapshot.frames == 1
    assert snapshot.clips == 1
    assert snapshot.transcriptions == 1
    assert snapshot.last_transcribe_s == pytest.approx(0.42)


def test_transcription_failure_is_counted_without_killing_health(
    monitor: HealthMonitor,
) -> None:
    monitor.capture_started()
    monitor.note_frame(LOUD)
    monitor.note_error("transcription failed: boom")
    snapshot = monitor.snapshot()
    # One bad clip is not a reason to tell the operator the pipeline is down.
    assert snapshot.state == OK
    assert snapshot.errors == 1


def test_snapshot_serialises_for_the_dashboard(monitor: HealthMonitor) -> None:
    monitor.capture_started()
    monitor.note_frame(LOUD)
    data = monitor.snapshot().to_dict()
    assert data["state"] == OK
    assert data["issues"] == []
    assert isinstance(data["signal_rms"], float)
    assert data["seconds_since_frame"] is not None
