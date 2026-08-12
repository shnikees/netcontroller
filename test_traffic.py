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

"""Tests for reading a traffic declaration off a transmission.

The negatives matter more than the positives here. Far more stations say "no
traffic" than "with traffic", so a detector that misreads the denials would
flag the whole net and be worse than no detector at all.
"""

from __future__ import annotations

import pytest

from traffic import HAS, NONE, UNKNOWN, detect


@pytest.mark.parametrize(
    "text",
    [
        "kilo seven xray yankee zulu, checking in with traffic",
        "W6ABC, I have traffic for the county EOC",
        "this is N5DEF with one piece of traffic",
        "priority traffic for net control",
        "emergency traffic, break",
        "I have a message for the incident commander",
        "K7XYZ, traffic for Turn 7",
    ],
)
def test_declaring_traffic(text: str) -> None:
    assert detect(text) == HAS


@pytest.mark.parametrize(
    "text",
    [
        "whiskey six alpha bravo charlie, checking in, no traffic",
        "N5DEF, nothing for the net",
        "no traffic tonight, back to you",
        "I have no traffic at this time",
        "negative traffic",
        "nothing to pass",
        "KJ6TUV checking in, nothing further",
    ],
)
def test_denying_traffic(text: str) -> None:
    assert detect(text) == NONE


@pytest.mark.parametrize(
    "text",
    [
        "car off at my corner, no injuries",
        "rider down, need medical at my location",
        "this is W6ABC, good evening",
        "",
        "static, unreadable",
    ],
)
def test_saying_nothing_about_traffic(text: str) -> None:
    # Not mentioning it is a third state, not a denial: the station may well
    # be holding something they have not offered yet.
    assert detect(text) == UNKNOWN


def test_net_control_asking_is_not_net_control_holding() -> None:
    # The question that would otherwise flag the busiest station on the net.
    assert detect("any traffic for the net?") == UNKNOWN
    assert detect("does anyone have traffic?") == UNKNOWN


def test_a_station_answering_that_question_still_counts() -> None:
    assert detect("net control, I have traffic") == HAS


def test_a_denial_and_a_declaration_together_counts_as_holding() -> None:
    # "no traffic for the net, but I have traffic for Turn 7" -- the thing they
    # are holding is what matters.
    assert detect("no traffic for the net, but I have traffic for Turn 7") == HAS


def test_the_negation_window_does_not_reach_across_a_sentence() -> None:
    # "no" belongs to the first clause; the traffic declaration is separate.
    assert detect("no injuries reported at my location. I have traffic") == HAS


def test_traffic_as_a_plain_noun_is_not_a_declaration() -> None:
    # A race net will say this about actual vehicles.
    assert detect("traffic is backing up on the access road") == UNKNOWN


def test_punctuation_and_case_do_not_matter() -> None:
    assert detect("W6ABC -- NO TRAFFIC.") == NONE
    assert detect("With Traffic!") == HAS
