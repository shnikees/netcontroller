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

"""Tests for the correction log and the aliases derived from it."""

from __future__ import annotations

import pytest

from callsign_match import CallsignMatcher, RosterEntry
from feedback import FeedbackLog, record_correction

ROSTER = [
    RosterEntry("W6ABC", "Alice"),
    RosterEntry("K7XYZ", "Bob"),
    RosterEntry("N5DEF", "Carol"),
]


@pytest.fixture
def feedback(tmp_path) -> FeedbackLog:
    return FeedbackLog(tmp_path / "feedback.jsonl")


def correct(feedback: FeedbackLog, candidate, to_callsign, entry_id=1, **kwargs):
    return record_correction(
        feedback,
        entry_id=entry_id,
        candidate=candidate,
        from_callsign=kwargs.pop("from_callsign", None),
        to_callsign=to_callsign,
        raw_text=kwargs.pop("raw_text", "some transcript"),
        **kwargs,
    )


# --------------------------------------------------------------------------
# The log
# --------------------------------------------------------------------------


def test_corrections_round_trip(feedback: FeedbackLog) -> None:
    correct(feedback, "E3Z", "K7XYZ", raw_text="Fictor echo three zulu")
    stored = feedback.all()
    assert len(stored) == 1
    assert stored[0].candidate == "E3Z"
    assert stored[0].to_callsign == "K7XYZ"
    # The transcript is kept because it is the label for future fine-tuning.
    assert stored[0].raw_text == "Fictor echo three zulu"


def test_log_is_append_only(feedback: FeedbackLog) -> None:
    correct(feedback, "E3Z", "K7XYZ", entry_id=1)
    correct(feedback, "W6ABD", "W6ABC", entry_id=2)
    assert [c.entry_id for c in feedback.all()] == [1, 2]


def test_missing_log_is_not_an_error(tmp_path) -> None:
    assert FeedbackLog(tmp_path / "nope.jsonl").all() == []
    assert FeedbackLog(tmp_path / "nope.jsonl").aliases() == {}


def test_truncated_line_costs_only_that_correction(feedback: FeedbackLog) -> None:
    # Simulates a write interrupted by a power cut mid-net.
    correct(feedback, "E3Z", "K7XYZ")
    with open(feedback.path, "a", encoding="utf-8") as fh:
        fh.write('{"timestamp": "2026-04-01T19:0')
    assert len(feedback.all()) == 1
    assert feedback.aliases() == {"E3Z": "K7XYZ"}


# --------------------------------------------------------------------------
# Deriving aliases
# --------------------------------------------------------------------------


def test_aliases_derived_from_log(feedback: FeedbackLog) -> None:
    correct(feedback, "E3Z", "K7XYZ")
    correct(feedback, "W6ABD", "W6ABC")
    assert feedback.aliases() == {"E3Z": "K7XYZ", "W6ABD": "W6ABC"}


def test_later_correction_wins(feedback: FeedbackLog) -> None:
    # The operator corrected the same mis-hearing twice; the second is truth.
    correct(feedback, "E3Z", "K7XYZ")
    correct(feedback, "E3Z", "N5DEF")
    assert feedback.aliases()["E3Z"] == "N5DEF"


def test_correction_without_candidate_yields_no_alias(feedback: FeedbackLog) -> None:
    # Nothing callsign-shaped was heard, so there is no key to learn against --
    # but the correction is still logged as training data.
    correct(feedback, None, "W6ABC")
    assert feedback.aliases() == {}
    assert len(feedback.all()) == 1


# --------------------------------------------------------------------------
# The matcher applying them
# --------------------------------------------------------------------------


def test_learned_alias_matches_next_time() -> None:
    matcher = CallsignMatcher(roster=ROSTER)
    raw = "Fictor echo three zulu quiddac Romeo, visiting station."
    assert not matcher.match(raw).matched

    assert matcher.learn_alias("E3Z", "K7XYZ")

    result = matcher.match(raw)
    assert result.matched
    assert result.callsign == "K7XYZ"
    assert result.via_alias


def test_alias_overrides_an_ambiguous_refusal() -> None:
    matcher = CallsignMatcher(
        roster=[RosterEntry("W6ABC", "Alice"), RosterEntry("W6ABE", "Eve")]
    )
    assert matcher.match("whiskey six alpha bravo delta").reason == "ambiguous"

    matcher.learn_alias("W6ABD", "W6ABE")

    result = matcher.match("whiskey six alpha bravo delta")
    assert result.matched
    assert result.callsign == "W6ABE"


def test_alias_to_a_station_not_on_the_roster_is_refused() -> None:
    matcher = CallsignMatcher(roster=ROSTER)
    assert not matcher.learn_alias("E3Z", "VE3ZQR")
    assert matcher.aliases == {}


def test_very_short_candidates_are_not_learned() -> None:
    # "K7" carries too little signal; keying on it would mis-fire constantly.
    matcher = CallsignMatcher(roster=ROSTER)
    assert not matcher.learn_alias("K7", "K7XYZ")
    assert matcher.aliases == {}


def test_alias_equal_to_the_callsign_is_not_learned() -> None:
    matcher = CallsignMatcher(roster=ROSTER)
    assert not matcher.learn_alias("K7XYZ", "K7XYZ")
    assert matcher.aliases == {}


def test_stale_aliases_are_dropped_when_a_station_leaves_the_roster() -> None:
    # Operator removed K7XYZ from roster.csv; last week's alias must not
    # resurrect it.
    matcher = CallsignMatcher(
        roster=[RosterEntry("W6ABC", "Alice")],
        aliases={"E3Z": "K7XYZ", "W6ABD": "W6ABC"},
    )
    assert matcher.aliases == {"W6ABD": "W6ABC"}


def test_aliases_survive_a_restart(feedback: FeedbackLog) -> None:
    correct(feedback, "E3Z", "K7XYZ")
    # A fresh process replays the log at startup, exactly as app.py does.
    matcher = CallsignMatcher(roster=ROSTER, aliases=feedback.aliases())
    result = matcher.match("Fictor echo three zulu quiddac Romeo, listening")
    assert result.matched
    assert result.callsign == "K7XYZ"
