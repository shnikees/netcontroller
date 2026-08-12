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

"""Tests for writing the session to disk as it happens.

The scenario these exist for is the ugly one: the machine dies mid-net without
a chance to clean up. Whatever reached disk before that moment is what net
control has left, so these check the file *during* the session, never only
after a tidy close.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from session_writer import SessionWriter, latest_session, read_session
from transcript_store import TranscriptStore

START = datetime(2026, 4, 1, 19, 0, 0)


def add(store: TranscriptStore, callsign: str | None, text: str, minutes: int = 0):
    return store.add(
        started_at=START + timedelta(minutes=minutes),
        matched=callsign is not None,
        matched_callsign=callsign,
        operator_name="Alice" if callsign else "",
        raw_text=text,
        confidence=0.9,
        match_score=100.0,
        clip_duration=3.0,
        candidate=callsign,
    )


@pytest.fixture
def session(tmp_path):
    store = TranscriptStore()
    writer = SessionWriter(tmp_path / "transcripts", store, started_at=START)
    writer.start()
    return writer, store


def test_files_are_created_at_start(session) -> None:
    writer, _ = session
    assert writer.jsonl_path.exists()
    assert writer.jsonl_path.name == "net-20260401-190000.jsonl"


def test_each_entry_reaches_disk_immediately(session) -> None:
    """The whole point: no waiting for a clean shutdown."""
    writer, store = session
    writer.append(add(store, "W6ABC", "checking in"))

    # Read it back without closing anything -- this is the mid-net state.
    records = read_session(writer.jsonl_path)
    entries = [r for r in records if r["type"] == "entry"]
    assert len(entries) == 1
    assert entries[0]["matched_callsign"] == "W6ABC"


def test_readable_log_is_current_without_an_export(session) -> None:
    writer, store = session
    writer.append(add(store, "W6ABC", "checking in"))
    writer.append(add(store, "K7XYZ", "no traffic", minutes=1))

    text = writer.text_path.read_text(encoding="utf-8")
    assert "W6ABC (Alice): checking in" in text
    assert "K7XYZ" in text


def test_readable_log_stays_in_transmission_order(session) -> None:
    # A clip recovered from the spill arrives late but belongs earlier.
    writer, store = session
    writer.append(add(store, "K7XYZ", "second", minutes=5))
    writer.append(add(store, "W6ABC", "first", minutes=1))

    lines = [
        line
        for line in writer.text_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("[")
    ]
    assert "first" in lines[0]
    assert "second" in lines[1]


def test_corrections_are_appended_as_history(session) -> None:
    writer, store = session
    entry = add(store, None, "Fictor echo three zulu")
    writer.append(entry)

    store.correct(entry.id, "KJ6TUV", "Frank")
    writer.record_correction(entry)

    records = read_session(writer.jsonl_path)
    kinds = [r["type"] for r in records]
    # The original line is still there: that the machine got it wrong before a
    # human fixed it is part of the record, not something to overwrite.
    assert kinds == ["session", "entry", "correction"]
    assert records[1]["matched"] is False
    assert records[2]["matched_callsign"] == "KJ6TUV"


def test_a_truncated_final_line_costs_only_that_line(session) -> None:
    # The shape a power cut leaves behind.
    writer, store = session
    writer.append(add(store, "W6ABC", "checking in"))
    with open(writer.jsonl_path, "a", encoding="utf-8") as handle:
        handle.write('{"type": "entry", "raw_text": "half a li')

    records = read_session(writer.jsonl_path)
    assert [r["type"] for r in records] == ["session", "entry"]


def test_missing_file_reads_as_empty(tmp_path) -> None:
    assert read_session(tmp_path / "nope.jsonl") == []


def test_unwritable_directory_does_not_stop_the_net(tmp_path) -> None:
    """A full or read-only disk costs the transcript file, not the session."""
    blocker = tmp_path / "in-the-way"
    blocker.write_text("not a directory")
    store = TranscriptStore()
    writer = SessionWriter(blocker / "transcripts", store)

    assert writer.start() is None
    writer.append(add(store, "W6ABC", "checking in"))  # must not raise
    writer.close()


def test_write_failure_is_reported_once(tmp_path, caplog) -> None:
    # A full disk should produce one error, not one per transmission for the
    # rest of the net.
    blocker = tmp_path / "in-the-way"
    blocker.write_text("not a directory")
    store = TranscriptStore()
    writer = SessionWriter(blocker / "transcripts", store)
    writer.start()

    with caplog.at_level("ERROR"):
        for i in range(5):
            writer.append(add(store, "W6ABC", f"line {i}", minutes=i))

    assert sum("Session writing disabled" in r.message for r in caplog.records) <= 1


def test_close_leaves_a_complete_readable_log(session) -> None:
    writer, store = session
    writer.append(add(store, "W6ABC", "checking in"))
    writer.close()

    text = writer.text_path.read_text(encoding="utf-8")
    assert "1 transmissions" in text
    assert "Check-ins: W6ABC" in text


# --------------------------------------------------------------------------
# Resuming an interrupted net
#
# A crash mid-net should leave one log for the net, not two halves somebody
# has to staple together afterwards.
# --------------------------------------------------------------------------


def test_restoring_rebuilds_the_log(tmp_path) -> None:
    writer = SessionWriter(tmp_path / "t", TranscriptStore(), started_at=START)
    store = writer.store
    writer.start()
    writer.append(add(store, "W6ABC", "checking in"))
    writer.append(add(store, "K7XYZ", "no traffic", minutes=1))

    restarted = TranscriptStore()
    assert restarted.restore(read_session(writer.jsonl_path)) == 2
    assert [e.matched_callsign for e in restarted.entries] == ["W6ABC", "K7XYZ"]
    assert restarted.check_ins() == ["W6ABC", "K7XYZ"]


def test_numbering_continues_after_a_restore(tmp_path) -> None:
    # Reusing an id would make a correction hit the wrong line.
    writer = SessionWriter(tmp_path / "t", TranscriptStore(), started_at=START)
    writer.start()
    writer.append(add(writer.store, "W6ABC", "first"))
    writer.append(add(writer.store, "K7XYZ", "second", minutes=1))

    restarted = TranscriptStore()
    restarted.restore(read_session(writer.jsonl_path))
    fresh = add(restarted, "N5DEF", "after the restart", minutes=2)
    assert fresh.id == 3


def test_a_correction_survives_the_restart(tmp_path) -> None:
    writer = SessionWriter(tmp_path / "t", TranscriptStore(), started_at=START)
    writer.start()
    entry = add(writer.store, None, "unreadable")
    writer.append(entry)
    writer.store.correct(entry.id, "KJ6TUV", "Frank")
    writer.record_correction(entry)

    restarted = TranscriptStore()
    restarted.restore(read_session(writer.jsonl_path))
    # What is restored is the log as it stood, not a replay of how it got there.
    assert len(restarted.entries) == 1
    assert restarted.entries[0].matched_callsign == "KJ6TUV"
    assert restarted.entries[0].corrected is True


def test_resuming_appends_to_the_same_files(tmp_path) -> None:
    first = SessionWriter(tmp_path / "t", TranscriptStore(), started_at=START)
    first.start()
    first.append(add(first.store, "W6ABC", "before the crash"))

    restarted = TranscriptStore()
    restarted.restore(read_session(first.jsonl_path))
    second = SessionWriter(tmp_path / "t", restarted, resume=first.jsonl_path)
    assert second.start() == first.jsonl_path
    second.append(add(restarted, "K7XYZ", "after the restart", minutes=1))

    text = first.text_path.read_text(encoding="utf-8")
    assert "before the crash" in text and "after the restart" in text
    assert len(list((tmp_path / "t").glob("*.jsonl"))) == 1


def test_resuming_records_that_it_resumed(tmp_path) -> None:
    first = SessionWriter(tmp_path / "t", TranscriptStore(), started_at=START)
    first.start()
    SessionWriter(tmp_path / "t", TranscriptStore(), resume=first.jsonl_path).start()
    assert any(r["type"] == "resumed" for r in read_session(first.jsonl_path))


def test_latest_session_finds_the_newest(tmp_path) -> None:
    directory = tmp_path / "t"
    directory.mkdir()
    import os, time

    for index, name in enumerate(("net-1.jsonl", "net-2.jsonl")):
        path = directory / name
        path.write_text("", encoding="utf-8")
        os.utime(path, (time.time() + index, time.time() + index))
    assert latest_session(directory).name == "net-2.jsonl"


def test_nothing_to_resume_from_is_not_an_error(tmp_path) -> None:
    assert latest_session(tmp_path / "nope") is None
    assert TranscriptStore().restore([]) == 0
