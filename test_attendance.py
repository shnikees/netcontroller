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

"""Tests for learning who actually turns up.

The guardrail matters more than the scoring: a callsign that appears in the
logs but not on the roster must never be adopted, or a mis-transcription
becomes a station and then biases decoding toward its own mistake.
"""

from __future__ import annotations

import json

from attendance import from_sessions, load
from callsign_match import CallsignMatcher, RosterEntry

ROSTER = {"W6ABC", "K7XYZ", "N5DEF", "KD9MNO"}


def entry(callsign: str | None, source: str = "", timestamp: str = "2026-04-01T19:00:00"):
    return {
        "type": "entry",
        "id": 1,
        "matched": callsign is not None,
        "matched_callsign": callsign,
        "source": source,
        "timestamp": timestamp,
    }


def test_a_regular_outranks_an_occasional_station() -> None:
    sessions = [
        [entry("W6ABC"), entry("K7XYZ")],
        [entry("W6ABC")],
        [entry("W6ABC")],
    ]
    scores = from_sessions(sessions, ROSTER).scores()
    assert scores["W6ABC"] > scores["K7XYZ"]


def test_recent_attendance_outweighs_old_attendance() -> None:
    # Crews change: last month's regular matters more than last season's.
    sessions = [[entry("W6ABC")], [entry("K7XYZ")], [entry("K7XYZ")]]
    scores = from_sessions(sessions, ROSTER).scores()
    assert scores["K7XYZ"] > scores["W6ABC"]


def test_talking_a_lot_in_one_session_is_not_attending_twice() -> None:
    # An event net has people transmitting constantly; that is one attendance.
    chatty = [[entry("W6ABC") for _ in range(20)], [entry("K7XYZ")]]
    result = from_sessions(chatty, ROSTER)
    assert result.records["W6ABC"].sessions == 1
    assert result.records["W6ABC"].transmissions == 20
    assert result.records["K7XYZ"].score > result.records["W6ABC"].score


def test_a_callsign_not_on_the_roster_is_reported_never_adopted() -> None:
    """The guardrail. A mis-transcription that became a station would bias
    decoding toward its own mistake."""
    result = from_sessions([[entry("W6ABC"), entry("VE3ZQR")]], ROSTER)
    assert "VE3ZQR" not in result.records
    assert result.unknown == ["VE3ZQR"]


def test_unmatched_lines_do_not_count() -> None:
    result = from_sessions([[entry(None), entry("W6ABC")]], ROSTER)
    assert list(result.records) == ["W6ABC"]


def test_attendance_is_tracked_per_receiver() -> None:
    sessions = [[entry("W6ABC", source="Repeater"), entry("K7XYZ", source="Simplex")]]
    result = from_sessions(sessions, ROSTER)
    assert "W6ABC" in result.for_source("Repeater")
    assert "W6ABC" not in result.for_source("Simplex")


def test_a_station_never_heard_on_a_source_is_still_scored_globally() -> None:
    result = from_sessions([[entry("W6ABC")]], ROSTER)
    # No source recorded means no opinion, so it stays in the running.
    assert "W6ABC" in result.for_source("Repeater")


def test_no_history_is_not_an_error(tmp_path) -> None:
    result = load(tmp_path / "nope", ROSTER)
    assert result.records == {} and result.sessions == 0


def test_sessions_are_read_from_disk(tmp_path) -> None:
    for index, callsign in enumerate(("W6ABC", "K7XYZ")):
        (tmp_path / f"net-{index}.jsonl").write_text(
            json.dumps({"type": "session"}) + "\n"
            + json.dumps(entry(callsign) | {"id": 1}) + "\n",
            encoding="utf-8",
        )
    result = load(tmp_path, ROSTER)
    assert result.sessions == 2
    assert set(result.records) == {"W6ABC", "K7XYZ"}


def test_a_correction_decides_who_attended(tmp_path) -> None:
    # The machine guessed one station and a human said it was another; the
    # human wins.
    (tmp_path / "net-1.jsonl").write_text(
        json.dumps(entry("W6ABC") | {"id": 7}) + "\n"
        + json.dumps(
            {"type": "correction", "id": 7, "matched": True, "matched_callsign": "K7XYZ"}
        ) + "\n",
        encoding="utf-8",
    )
    result = load(tmp_path, ROSTER)
    assert set(result.records) == {"K7XYZ"}


def test_a_truncated_line_is_skipped(tmp_path) -> None:
    (tmp_path / "net-1.jsonl").write_text(
        json.dumps(entry("W6ABC") | {"id": 1}) + "\n" + '{"type": "entry", "id": 2',
        encoding="utf-8",
    )
    assert set(load(tmp_path, ROSTER).records) == {"W6ABC"}


# --------------------------------------------------------------------------
# What it is for: ordering the prompt
# --------------------------------------------------------------------------


def test_expected_stations_lead_the_prompt() -> None:
    roster = [RosterEntry("W6ABC"), RosterEntry("K7XYZ"), RosterEntry("N5DEF")]
    matcher = CallsignMatcher(roster=roster)
    # N5DEF is the regular; roster order would have put them last.
    scores = {"N5DEF": 3.0, "K7XYZ": 1.0, "W6ABC": 0.2}

    terms = matcher.bias_terms(attendance=scores)
    assert terms.index("N5DEF") < terms.index("K7XYZ") < terms.index("W6ABC")


def test_without_history_roster_order_is_kept() -> None:
    roster = [RosterEntry("W6ABC"), RosterEntry("K7XYZ")]
    terms = CallsignMatcher(roster=roster).bias_terms()
    assert terms.index("W6ABC") < terms.index("K7XYZ")


def test_stations_not_yet_heard_still_come_first() -> None:
    # Attendance orders *within* the groups; who has not spoken tonight is
    # still the stronger signal on a check-in net.
    roster = [RosterEntry("W6ABC"), RosterEntry("K7XYZ")]
    matcher = CallsignMatcher(roster=roster)
    terms = matcher.bias_terms(heard={"W6ABC"}, attendance={"W6ABC": 9.0, "K7XYZ": 0.1})
    assert terms.index("K7XYZ") < terms.index("W6ABC")
