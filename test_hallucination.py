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

"""Telling the prompt coming back from a station checking in.

The strings marked *verbatim* are real output from a live repeater recording,
transcribed with the three known callsigns in both the prompt and hotwords.
"""

from __future__ import annotations

import pytest

from hallucination import filter_matches, looks_hallucinated

BIAS = ["KJ7RAB", "KJ7JXM", "KI7RMU"]


@pytest.mark.parametrize(
    "text,matches,why",
    [
        # verbatim
        ("KJ7RAB, KJ7RAB, KJ7JXM, KI7RMU.", ["KJ7RAB", "KJ7RAB", "KJ7JXM", "KI7RMU"],
         "three stations and a repeat"),
        ("KJ7RAB, KJ7JXM, KI7RMU.", ["KJ7RAB", "KJ7JXM", "KI7RMU"],
         "the whole bias list, nothing else"),
        ("VJ7RAB, KJ7JXM, KI7RMU, alpha bravo charlie delta echo foxtrot golf",
         ["KJ7JXM", "KI7RMU"], "the phonetic alphabet out of the prompt"),
        ("KJ7RAB, KJ7JXM.", ["KJ7RAB", "KJ7JXM"], "two callsigns, no speech"),
    ],
)
def test_the_prompt_coming_back_is_caught(text, matches, why) -> None:
    suspect, reason = looks_hallucinated(text, matches, bias_order=BIAS)
    assert suspect, f"missed: {why}"
    assert reason


@pytest.mark.parametrize(
    "text,matches",
    [
        # A real check-in is *entirely* vocabulary words. That must never be
        # the signal, or every genuine transmission is thrown away.
        ("kilo juliet seven romeo alpha bravo", ["KJ7RAB"]),
        ("Good afternoon, this is KJ7RAB and it is time for the noon net",
         ["KJ7RAB"]),
        # One station naming another is ordinary net traffic.
        ("KJ7RAB here, I have traffic for KJ7JXM when you are ready",
         ["KJ7RAB", "KJ7JXM"]),
        ("KI7RMU, go ahead with your check-in please", ["KI7RMU"]),
    ],
)
def test_real_traffic_survives(text, matches) -> None:
    suspect, reason = looks_hallucinated(text, matches, bias_order=BIAS)
    assert not suspect, f"dropped a real transmission: {reason}"


def test_nothing_matched_is_never_suspect() -> None:
    assert looks_hallucinated("just weather chat here", []) == (False, "")


def test_a_suspect_line_keeps_none_of_its_callsigns() -> None:
    """All or nothing: once a transcript is the prompt coming back, there is no
    principled way to pick which of its callsigns were really said."""
    kept, reason = filter_matches(
        "KJ7RAB, KJ7JXM, KI7RMU.", ["KJ7RAB", "KJ7JXM", "KI7RMU"], bias_order=BIAS
    )
    assert kept == []
    assert "stations" in reason


def test_a_clean_line_keeps_all_of_them() -> None:
    kept, reason = filter_matches(
        "KJ7RAB here with traffic for KJ7JXM, go ahead",
        ["KJ7RAB", "KJ7JXM"], bias_order=BIAS,
    )
    assert kept == ["KJ7RAB", "KJ7JXM"]
    assert reason == ""
