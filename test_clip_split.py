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

"""Tests for splitting a clip that caught two stations.

The interesting half of this is the *refusals*. Splitting a transmission that
merely names another station invents a check-in nobody made, and a phantom
entry in a net log is worse than a missing one because nobody reading it will
know to doubt it.
"""

from __future__ import annotations

from clip_split import split_transmissions
from callsign_match import CallsignMatcher, RosterEntry
from stt_worker import Word

ROSTER = [
    RosterEntry("W6ABC", "Alice"),
    RosterEntry("K7XYZ", "Bob"),
    RosterEntry("N5DEF", "Carol"),
]
MATCHER = CallsignMatcher(roster=ROSTER)


def words_for(text: str, gaps: dict[int, float] | None = None) -> list[Word]:
    """Lay `text` out in time: 0.3 s per word, plus any gaps before word N."""
    gaps = gaps or {}
    words: list[Word] = []
    clock = 0.0
    offset = 0
    for index, token in enumerate(text.split(" ")):
        clock += gaps.get(index, 0.0)
        found = text.index(token, offset)
        words.append(Word(text=token, start=clock, end=clock + 0.3, offset=found))
        offset = found + len(token)
        clock += 0.3
    return words


def split(text: str, gaps=None, duration_ms: int = 12_000, **kwargs):
    return split_transmissions(
        text,
        words_for(text, gaps),
        MATCHER.match_all(text),
        duration_ms,
        **kwargs,
    )


TWO_STATIONS = (
    "whiskey six alpha bravo charlie checking in no traffic "
    "kilo seven xray yankee zulu also checking in"
)


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------


def test_two_stations_with_a_pause_become_two_lines() -> None:
    # A second of dead air where one operator unkeys and another keys up.
    segments = split(TWO_STATIONS, gaps={9: 1.0})
    assert len(segments) == 2
    assert "whiskey six alpha bravo charlie" in segments[0].text
    assert "kilo seven xray yankee zulu" in segments[1].text


def test_the_second_transmission_is_timed_from_its_own_audio() -> None:
    segments = split(TWO_STATIONS, gaps={9: 1.0})
    # First nine words at 0.3 s each, then the gap.
    assert segments[0].start_offset_ms == 0
    assert segments[1].start_offset_ms > 3_000


def test_each_segment_matches_its_own_station() -> None:
    segments = split(TWO_STATIONS, gaps={9: 1.0})
    assert MATCHER.match(segments[0].text).callsign == "W6ABC"
    assert MATCHER.match(segments[1].text).callsign == "K7XYZ"


def test_three_stations_in_one_clip() -> None:
    text = (
        "whiskey six alpha bravo charlie here "
        "kilo seven xray yankee zulu here "
        "november five delta echo foxtrot here"
    )
    segments = split(text, gaps={6: 1.0, 12: 1.0})
    assert len(segments) == 3


# --------------------------------------------------------------------------
# Refusing to split -- the part that keeps phantom check-ins out of the log
# --------------------------------------------------------------------------


def test_one_station_naming_another_is_not_split() -> None:
    """The case that makes text alone useless as evidence.

    "W6ABC here, traffic for K7XYZ" has two roster callsigns and one speaker.
    No pause, so no split.
    """
    text = "whiskey six alpha bravo charlie here i have traffic for kilo seven xray yankee zulu"
    assert len(MATCHER.match_all(text)) == 2  # both callsigns are really there
    assert len(split(text)) == 1


def test_a_short_pause_is_not_a_change_of_transmitter() -> None:
    # 200 ms is somebody drawing breath, not somebody unkeying.
    assert len(split(TWO_STATIONS, gaps={9: 0.2})) == 1


def test_the_gap_threshold_is_configurable() -> None:
    # A net where stations run close together can lower the bar deliberately.
    assert len(split(TWO_STATIONS, gaps={9: 0.35})) == 1
    assert len(split(TWO_STATIONS, gaps={9: 0.35}, min_gap_ms=300)) == 2


def test_no_word_timings_means_no_splitting() -> None:
    # Without timings there is no evidence, so the safe answer is one line.
    segments = split_transmissions(TWO_STATIONS, [], MATCHER.match_all(TWO_STATIONS), 12_000)
    assert len(segments) == 1


def test_a_single_callsign_is_never_split() -> None:
    text = "whiskey six alpha bravo charlie checking in with no traffic tonight"
    assert len(split(text, gaps={4: 2.0})) == 1


def test_no_callsign_at_all_is_never_split() -> None:
    assert len(split("say again you were covered up there", gaps={3: 2.0})) == 1


def test_a_sliver_segment_folds_back_into_one() -> None:
    # A split that would log a fraction of a second as a check-in is more
    # likely a timing artefact than a transmission.
    segments = split(TWO_STATIONS, gaps={9: 1.0}, min_segment_ms=60_000)
    assert len(segments) == 1


def test_the_same_station_twice_is_not_two_stations() -> None:
    text = (
        "whiskey six alpha bravo charlie checking in "
        "whiskey six alpha bravo charlie again"
    )
    assert len(split(text, gaps={6: 1.0})) == 1


# --------------------------------------------------------------------------
# Shape of the result
# --------------------------------------------------------------------------


def test_unsplit_clip_keeps_the_whole_text_and_duration() -> None:
    text = "whiskey six alpha bravo charlie checking in"
    segments = split(text, duration_ms=5_000)
    assert segments[0].text == text
    assert segments[0].duration_ms == 5_000


def test_durations_are_positive_and_roughly_add_up() -> None:
    segments = split(TWO_STATIONS, gaps={9: 1.0}, duration_ms=12_000)
    assert all(s.duration_ms > 0 for s in segments)
    assert sum(s.duration_ms for s in segments) <= 12_000 + 1_000
