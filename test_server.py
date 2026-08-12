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

"""Tests for the HTTP API, mainly the correction endpoint."""

from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from callsign_match import CallsignMatcher, RosterEntry
from feedback import FeedbackLog
from server import Broadcaster, create_app
from transcript_store import TranscriptStore

ROSTER = [RosterEntry("W6ABC", "Alice"), RosterEntry("K7XYZ", "Bob")]


@pytest.fixture
def context(tmp_path):
    store = TranscriptStore()
    store.add(
        started_at=datetime(2026, 4, 1, 19, 0, 0),
        matched=False,
        matched_callsign=None,
        operator_name="",
        raw_text="Fictor echo three zulu quiddac Romeo, visiting station.",
        confidence=0.57,
        match_score=0.0,
        clip_duration=3.4,
        candidate="E3Z",
        unmatched_reason="below_threshold",
    )
    matcher = CallsignMatcher(roster=ROSTER)
    feedback = FeedbackLog(tmp_path / "feedback.jsonl")
    app = create_app(
        store,
        ROSTER,
        Broadcaster(),
        export_dir=str(tmp_path),
        matcher=matcher,
        feedback=feedback,
    )
    return TestClient(app), store, matcher, feedback


def test_history_includes_roster(context) -> None:
    client, *_ = context
    data = client.get("/api/history").json()
    assert len(data["entries"]) == 1
    assert data["roster"][0]["callsign"] == "W6ABC"


def test_correction_updates_entry_logs_and_learns(context) -> None:
    client, store, matcher, feedback = context

    res = client.post("/api/correct", json={"entry_id": 1, "callsign": "K7XYZ"})
    assert res.status_code == 200
    body = res.json()

    # 1. the log line is fixed, and remembers what it used to say
    assert body["entry"]["matched"] is True
    assert body["entry"]["matched_callsign"] == "K7XYZ"
    assert body["entry"]["operator_name"] == "Bob"
    assert body["entry"]["corrected"] is True
    assert body["entry"]["original_callsign"] is None  # it was unmatched

    # 2. the correction is on disk as training data
    assert [c.to_callsign for c in feedback.all()] == ["K7XYZ"]

    # 3. the matcher learned it, so the next transmission matches by itself
    assert body["learned"] is True
    assert matcher.aliases == {"E3Z": "K7XYZ"}
    assert store.check_ins() == ["K7XYZ"]


def test_correcting_a_wrong_match_keeps_the_original(context) -> None:
    client, store, _, _ = context
    store.correct(1, "W6ABC", "Alice")  # matcher had guessed wrong

    body = client.post(
        "/api/correct", json={"entry_id": 1, "callsign": "K7XYZ"}
    ).json()
    assert body["entry"]["matched_callsign"] == "K7XYZ"
    assert body["entry"]["original_callsign"] is None
    # Re-correcting must not overwrite the record of the machine's answer with
    # the operator's first attempt.
    assert store.get(1).corrected is True


def test_correction_to_an_off_roster_callsign_is_rejected(context) -> None:
    client, store, matcher, feedback = context
    res = client.post("/api/correct", json={"entry_id": 1, "callsign": "VE3ZQR"})
    assert res.status_code == 400
    assert store.get(1).matched is False
    assert feedback.all() == []
    assert matcher.aliases == {}


def test_correction_of_a_missing_entry_404s(context) -> None:
    client, *_ = context
    assert client.post("/api/correct", json={"entry_id": 99, "callsign": "W6ABC"}).status_code == 404


def test_callsign_is_normalized_to_upper(context) -> None:
    client, store, *_ = context
    client.post("/api/correct", json={"entry_id": 1, "callsign": " k7xyz "})
    assert store.get(1).matched_callsign == "K7XYZ"


def test_aliases_endpoint_reports_what_was_learned(context) -> None:
    client, *_ = context
    assert client.get("/api/aliases").json()["aliases"] == {}
    client.post("/api/correct", json={"entry_id": 1, "callsign": "K7XYZ"})
    assert client.get("/api/aliases").json()["aliases"] == {"E3Z": "K7XYZ"}


def test_export_writes_files(context, tmp_path) -> None:
    client, *_ = context
    body = client.post("/api/export").json()
    assert body["entries"] == 1
    assert len(body["files"]) == 2
