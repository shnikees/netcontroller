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

"""Tests for the normalizer and roster matcher.

The strings here are written the way faster-whisper actually emits net traffic:
inconsistent capitalization, phonetics run together, digits sometimes spelled
and sometimes not, filler wrapped around the callsign.
"""

from __future__ import annotations

import pytest

from callsign_match import (
    CallsignMatcher,
    RosterEntry,
    extract_candidates,
    load_roster,
    normalize,
)

ROSTER = [
    RosterEntry("W6ABC", "Alice"),
    RosterEntry("K7XYZ", "Bob"),
    RosterEntry("N5DEF", "Carol"),
    RosterEntry("KD9MNO", "Dave"),
    RosterEntry("AA4PQ", "Erin"),
]


@pytest.fixture
def matcher() -> CallsignMatcher:
    return CallsignMatcher(roster=ROSTER)


# --------------------------------------------------------------------------
# normalize()
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Whiskey Six Alpha Bravo Charlie", "W6ABC"),
        ("whisky six alfa bravo charlie", "W6ABC"),
        ("This is Whiskey 6 Alpha Bravo Charlie", "W6ABC"),
        ("Kilo Seven X-ray Yankee Zulu, over.", "K7XYZ"),
        ("November five delta echo foxtrot for net control", "N5DEF"),
        ("kilo delta niner mike november oscar", "KD9MNO"),
    ],
)
def test_normalize_spoken_callsigns(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


def test_normalize_splits_glued_phonetics() -> None:
    # Whisper regularly runs phonetics together with no space.
    assert normalize("whiskey six alfabravo charlie") == "W6ABC"


def test_normalize_keeps_unknown_words() -> None:
    # Non-vocabulary words survive so they can be shown in the transcript and
    # so a callsign spoken normally ("W6ABC") is still findable.
    assert "WEATHER" in normalize("W6ABC weather is clear")


def test_normalize_drops_filler() -> None:
    normalized = normalize("this is net control, go ahead")
    assert normalized == ""


def test_ambiguous_homophones_only_convert_next_to_spelling() -> None:
    # "for" as a preposition must not become a 4...
    assert "4" not in normalize("standing by for traffic")
    # ...but "alpha for bravo" is someone spelling A-4-B.
    assert normalize("alpha for bravo") == "A4B"


# --------------------------------------------------------------------------
# extract_candidates()
# --------------------------------------------------------------------------


def test_extract_candidates_finds_callsign_shape() -> None:
    assert extract_candidates("W6ABC WEATHER IS CLEAR") == ["W6ABC"]


def test_extract_candidates_empty_on_plain_speech() -> None:
    assert extract_candidates("NOTHING HEARD HERE") == []


def test_extract_candidates_orders_strict_before_loose() -> None:
    candidates = extract_candidates("KD9MNOP W6ABC")
    assert candidates[0] == "W6ABC"
    assert "KD9MNOP" in candidates


# --------------------------------------------------------------------------
# CallsignMatcher.match()
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Whiskey six alpha bravo charlie, checking in", "W6ABC"),
        ("this is kilo seven xray yankee zulu for check in", "K7XYZ"),
        ("November Five Delta Echo Foxtrot, no traffic", "N5DEF"),
        ("Kilo delta niner Mike November Oscar, QNI", "KD9MNO"),
        ("alpha alpha four papa quebec", "AA4PQ"),
    ],
)
def test_match_clean_transmissions(
    matcher: CallsignMatcher, raw: str, expected: str
) -> None:
    result = matcher.match(raw)
    assert result.matched
    assert result.callsign == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        # One mis-heard suffix letter -- still unambiguously one roster entry.
        ("whiskey six alpha bravo charlie", "W6ABC"),
        ("whiskey six alpha bravo delta", "W6ABC"),
        ("kilo seven x-ray yankee zoo loo", "K7XYZ"),
        # Digit heard as its neighbour: 5 <-> 9 is a classic Whisper confusion.
        ("november nine delta echo foxtrot", "N5DEF"),
    ],
)
def test_match_tolerates_single_character_errors(
    matcher: CallsignMatcher, raw: str, expected: str
) -> None:
    result = matcher.match(raw)
    assert result.matched, f"{raw!r} rejected: {result.reason} (score {result.score})"
    assert result.callsign == expected


def test_match_returns_operator_name(matcher: CallsignMatcher) -> None:
    assert matcher.match("whiskey six alpha bravo charlie").name == "Alice"


def test_unmatched_on_garbage(matcher: CallsignMatcher) -> None:
    result = matcher.match("uh, static, unreadable, say again")
    assert not result.matched
    assert result.reason == "no_candidate"


def test_unmatched_on_off_roster_callsign(matcher: CallsignMatcher) -> None:
    result = matcher.match("victor echo three zulu quebec romeo checking in")
    assert not result.matched
    assert result.reason == "below_threshold"
    # The candidate is still surfaced so net control can resolve it by ear.
    assert result.candidate == "VE3ZQR"


def test_unmatched_when_two_roster_entries_are_equally_close() -> None:
    # W6ABD sits one edit from both W6ABC and W6ABE -- refuse to guess.
    matcher = CallsignMatcher(
        roster=[RosterEntry("W6ABC", "Alice"), RosterEntry("W6ABE", "Eve")]
    )
    result = matcher.match("whiskey six alpha bravo delta")
    assert not result.matched
    assert result.reason == "ambiguous"


def test_exact_hit_beats_ambiguity_margin() -> None:
    # An exact match must not be rejected just because a neighbour scores close.
    matcher = CallsignMatcher(
        roster=[RosterEntry("W6ABC", "Alice"), RosterEntry("W6ABD", "Dan")]
    )
    result = matcher.match("whiskey six alpha bravo charlie")
    assert result.matched
    assert result.callsign == "W6ABC"


def test_empty_transmission_is_unmatched(matcher: CallsignMatcher) -> None:
    assert not matcher.match("").matched


def test_match_picks_callsign_from_longer_transmission(
    matcher: CallsignMatcher,
) -> None:
    raw = (
        "Good evening net control, this is whiskey six alpha bravo charlie, "
        "Alice in Sacramento, no traffic tonight, back to you."
    )
    result = matcher.match(raw)
    assert result.matched
    assert result.callsign == "W6ABC"


# --------------------------------------------------------------------------
# Regressions from real transcripts
#
# These strings are verbatim faster-whisper output from replaying recorded
# check-ins through the pipeline. Add to this block whenever a real net turns
# up a new way for the model to mangle a callsign.
# --------------------------------------------------------------------------


def test_whisper_renders_spoken_digit_as_an_ordinal(matcher: CallsignMatcher) -> None:
    raw = "november fifth delta echo foxtrot, good evening, nothing for the net."
    result = matcher.match(raw)
    assert result.matched
    assert result.callsign == "N5DEF"


def test_ordinals_stay_ordinals_outside_a_callsign(matcher: CallsignMatcher) -> None:
    # "net meets the first Tuesday" must not turn into a 1.
    assert "1" not in normalize("the net meets the first tuesday of the month")


def test_whisper_hyphenates_adjacent_spelled_words() -> None:
    # Verbatim from a replay: the hyphen used to swallow both the 6 and the T,
    # leaving no callsign-shaped token at all.
    matcher = CallsignMatcher(roster=ROSTER + [RosterEntry("KJ6TUV", "Frank")])
    result = matcher.match("kilo juliet six-tango uniform victor, frank checking in.")
    assert result.matched
    assert result.callsign == "KJ6TUV"


def test_hyphenated_vocabulary_words_survive() -> None:
    # ...but "x-ray" is itself a vocabulary entry and must not be split.
    assert normalize("kilo seven x-ray yankee zulu") == "K7XYZ"


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Prompting with the phonetic alphabet makes Whisper write callsigns in
        # a clipped style, and these are the forms that came back from it.
        ("Net control, kilo, 7XY, yankee, zulu, checking in", "K7XYZ"),
        ("November 5th, delta, echo, foxtrot, good evening", "N5DEF"),
        ("This is Whiskey 6 alpha, bravo, charlie, checking in", "W6ABC"),
    ],
)
def test_digits_welded_to_letters(raw: str, expected: str) -> None:
    matcher = CallsignMatcher(roster=ROSTER + [RosterEntry("KJ6TUV", "Frank")])
    result = matcher.match(raw)
    assert result.matched, f"{raw!r} -> {normalize(raw)!r} ({result.reason})"
    assert result.callsign == expected


def test_ordinal_written_numerically_is_a_digit() -> None:
    assert normalize("november 5th delta echo foxtrot") == "N5DEF"


def test_a_digit_welded_to_a_phonetic_splits() -> None:
    assert normalize("victor echo 3zulu quebec romeo") == "VE3ZQR"


def test_kilo_heard_as_kelo() -> None:
    # Verbatim from a replay: "KELO 7XY, Yankee, Zulu, checking in with traffic."
    matcher = CallsignMatcher(roster=ROSTER)
    result = matcher.match("KELO 7XY, Yankee, Zulu, checking in with traffic.")
    assert result.matched
    assert result.callsign == "K7XYZ"


def test_mangled_quebec_still_yields_a_full_candidate(
    matcher: CallsignMatcher,
) -> None:
    raw = "Victor echo three zulu quibbic Romeo, visiting station."
    result = matcher.match(raw)
    # Off-roster, so it stays unmatched -- but net control should see the whole
    # callsign it heard, not a truncated one.
    assert not result.matched
    assert result.candidate == "VE3ZQR"


def test_two_stations_in_one_clip_do_not_weld_together(
    matcher: CallsignMatcher,
) -> None:
    """On a fast net, two stations key up inside one VAD gap and land in one
    clip. The dropped filler between them used to leave the callsigns adjacent,
    where they glued into one nonsense token and *both* stations were lost."""
    raw = "whiskey six alpha bravo charlie no traffic kilo seven xray yankee zulu"
    assert normalize(raw) == "W6ABC K7XYZ"
    assert extract_candidates(normalize(raw)) == ["W6ABC", "K7XYZ"]

    result = matcher.match(raw)
    assert result.matched
    assert result.callsign == "W6ABC"


def test_filler_between_spelled_letters_still_separates() -> None:
    # "alpha bravo over charlie delta" is two fragments, not one four-letter run.
    assert normalize("alpha bravo over charlie delta") == "AB CD"


# --------------------------------------------------------------------------
# Roster loading + hotwords
# --------------------------------------------------------------------------


def test_load_roster_with_header(tmp_path) -> None:
    path = tmp_path / "roster.csv"
    path.write_text("callsign,name\nW6ABC,Alice\nk7xyz,Bob\n", encoding="utf-8")
    entries = load_roster(path)
    assert entries == [RosterEntry("W6ABC", "Alice"), RosterEntry("K7XYZ", "Bob")]


def test_roster_comments_are_not_stations(tmp_path) -> None:
    # Operators annotate rosters: a note at the top, a station commented out
    # because they are away. Those lines must not become callsigns.
    path = tmp_path / "roster.csv"
    path.write_text(
        "callsign,name\n# away this month:\n#K7XYZ,Bob\nW6ABC,Alice\n",
        encoding="utf-8",
    )
    assert [e.callsign for e in load_roster(path)] == ["W6ABC"]


def test_the_shipped_example_roster_is_clean() -> None:
    # It is the file everyone copies; it must not teach a broken pattern.
    for entry in load_roster("roster.example.csv"):
        assert not entry.callsign.startswith("#")


def test_load_roster_without_names(tmp_path) -> None:
    path = tmp_path / "roster.csv"
    path.write_text("W6ABC\nN5DEF\n", encoding="utf-8")
    assert [e.callsign for e in load_roster(path)] == ["W6ABC", "N5DEF"]


def test_bias_terms_cover_phonetics_vocabulary_and_the_roster(
    matcher: CallsignMatcher,
) -> None:
    terms = matcher.bias_terms(["QNI"])
    # The alphabet, not one spelling per station: 26 words bias every phonetic
    # callsign on the net, where per-station spellings would eat the budget.
    assert "whiskey" in terms and "zulu" in terms
    assert "QNI" in terms
    assert "W6ABC" in terms


def test_bias_terms_put_the_likeliest_stations_first() -> None:
    roster = [
        RosterEntry("W6ABC", "Alice", ("Repeater",)),
        RosterEntry("K7XYZ", "Bob", ("Simplex",)),
    ]
    matcher = CallsignMatcher(roster=roster)
    terms = matcher.bias_terms(source="Repeater")
    # This receiver's stations outrank the other frequency's, because the
    # prompt will be cut off long before the whole roster fits.
    assert terms.index("W6ABC") < terms.index("K7XYZ")


def test_stations_already_heard_drop_down_the_order() -> None:
    roster = [RosterEntry("W6ABC"), RosterEntry("K7XYZ")]
    matcher = CallsignMatcher(roster=roster)
    terms = matcher.bias_terms(heard={"W6ABC"})
    # On a check-in net the station who has not checked in yet is the one
    # about to speak.
    assert terms.index("K7XYZ") < terms.index("W6ABC")


def test_roster_csv_can_assign_stations_to_frequencies(tmp_path) -> None:
    path = tmp_path / "roster.csv"
    path.write_text(
        "callsign,name,sources\nW6ABC,Alice,Repeater\nK7XYZ,Bob,Repeater;Simplex\nN5DEF,Carol,\n",
        encoding="utf-8",
    )
    entries = {e.callsign: e for e in load_roster(path)}
    assert entries["W6ABC"].sources == ("Repeater",)
    assert entries["K7XYZ"].sources == ("Repeater", "Simplex")
    # No column means the station is expected anywhere.
    assert entries["N5DEF"].sources == ()
    assert entries["N5DEF"].on_source("Repeater")
    assert not entries["W6ABC"].on_source("Simplex")
