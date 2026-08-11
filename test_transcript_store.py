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

"""Tests for the session log and its exports."""

from __future__ import annotations

import csv
from datetime import datetime

import pytest

from transcript_store import TranscriptStore


def add(store: TranscriptStore, callsign: str | None, text: str, **kwargs):
    return store.add(
        started_at=datetime(2026, 4, 1, 19, 0, 0),
        matched=callsign is not None,
        matched_callsign=callsign,
        operator_name=kwargs.pop("name", "Alice" if callsign else ""),
        raw_text=text,
        confidence=kwargs.pop("confidence", 0.9),
        match_score=kwargs.pop("match_score", 100.0),
        clip_duration=kwargs.pop("clip_duration", 3.0),
        **kwargs,
    )


@pytest.fixture
def store() -> TranscriptStore:
    store = TranscriptStore()
    add(store, "W6ABC", "checking in")
    add(store, None, "unreadable", unmatched_reason="below_threshold", candidate="W6ABD")
    add(store, "W6ABC", "no traffic")
    return store


def test_ids_increment(store: TranscriptStore) -> None:
    assert [e.id for e in store.entries] == [1, 2, 3]


def test_check_ins_are_unique_and_ordered() -> None:
    store = TranscriptStore()
    add(store, "K7XYZ", "first")
    add(store, "W6ABC", "second")
    add(store, "K7XYZ", "again")
    assert store.check_ins() == ["K7XYZ", "W6ABC"]


def test_unmatched_entries_are_not_check_ins(store: TranscriptStore) -> None:
    assert store.check_ins() == ["W6ABC"]


def test_export_csv_round_trips(store: TranscriptStore, tmp_path) -> None:
    path = store.export_csv(tmp_path / "log.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert len(rows) == 3
    assert rows[0]["matched_callsign"] == "W6ABC"
    assert rows[1]["unmatched_reason"] == "below_threshold"


def test_export_text_is_readable(store: TranscriptStore, tmp_path) -> None:
    text = store.export_text(tmp_path / "log.txt").read_text(encoding="utf-8")
    assert "W6ABC (Alice): checking in" in text
    assert "UNMATCHED: unreadable" in text
    assert "Check-ins: W6ABC" in text


def test_export_of_empty_session_still_writes(tmp_path) -> None:
    path = TranscriptStore().export_text(tmp_path / "log.txt")
    assert "0 transmissions" in path.read_text(encoding="utf-8")
