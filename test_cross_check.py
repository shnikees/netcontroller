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

"""Tests for judging a biased engine's callsigns against an unbiased one.

The property that matters is the one the first version of this check got wrong:
support has to be judged by the *matcher*, not by a string compare. Engines
write the same callsign differently, and a check that cannot read a phonetic
spelling reports fabrication where there is none -- which would condemn the
better engine.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "tools"))

from cross_check import CALLSIGN_SHAPE, _supports  # noqa: E402


def test_a_phonetic_spelling_counts_as_support() -> None:
    """The regression that started this file.

    Parakeet writes "Kilo Juliet 7, Romeo Alpha Bravo" where Whisper writes
    "KJ7RAB". A regex over collapsed callsigns sees no support and calls a
    correct transcription a fabrication -- which on a real corpus understated
    agreement by a factor of three.
    """
    assert _supports("This is Cammy Kilo Juliet 7, Romeo Alpha Bravo. And it", "KJ7RAB")


def test_a_collapsed_spelling_counts_as_support() -> None:
    assert _supports("Good afternoon, this is KJ7RAB on the net", "KJ7RAB")


def test_unrelated_speech_is_not_support() -> None:
    """The other direction, and the one that makes the measurement mean anything.

    These are the real transcripts of clip c0008, where prompted Whisper
    reported three callsigns. Neither unbiased engine heard anything like them.
    """
    heard = "Whiskey Seven Oscar Hotel.\nwith these seven-hats, go tell."
    assert not _supports(heard, "KJ7RAB")
    assert not _supports(heard, "KJ7JXM")
    assert not _supports(heard, "KI7RAB")


def test_silence_is_not_support() -> None:
    """A missing or empty comparison transcript must count against the claim.

    Parakeet writes no file at all for a clip it judges unintelligible, and that
    silence is evidence, not an absence of evidence -- treating it as "skip"
    would quietly exclude exactly the clips where fabrication happens.
    """
    assert not _supports("", "KJ7RAB")


def test_a_near_miss_is_not_support() -> None:
    """Support must be for *that* callsign, not for something callsign-shaped.

    If any callsign counted, an engine that heard a different station would
    corroborate a fabrication.
    """
    assert not _supports("this is KJ7XYZ signing", "KJ7RAB")


def test_harvesting_finds_callsign_shapes_and_not_words() -> None:
    found = set(CALLSIGN_SHAPE.findall("WELCOME TO KJ7RAB, KJ7JXM, KI7RMU."))
    assert found == {"KJ7RAB", "KJ7JXM", "KI7RMU"}


def test_harvesting_ignores_plain_text() -> None:
    assert not CALLSIGN_SHAPE.findall("AROUND THE CORNER AND BACK AGAIN")


def test_end_to_end_over_two_directories(tmp_path, capsys) -> None:
    """The tool itself, on a corpus small enough to check by hand.

    Two clips: one where the callsign was really said and the unbiased engine
    spells it phonetically, one where it was not said at all. Correct output is
    one supported, one unsupported.
    """
    import cross_check

    suspect = tmp_path / "prompted"
    other = tmp_path / "plain"
    suspect.mkdir()
    other.mkdir()
    (suspect / "c1.txt").write_text("This is Kevin, KJ7RAB, on the net")
    (other / "c1.txt").write_text("This is Kevin Kilo Juliet Seven Romeo Alpha Bravo")
    (suspect / "c2.txt").write_text("Welcome to KJ7RAB.")
    (other / "c2.txt").write_text("around the")

    argv = sys.argv
    sys.argv = ["cross_check", "--suspect", str(suspect), "--against", str(other)]
    try:
        assert cross_check.main() == 0
    finally:
        sys.argv = argv

    output = capsys.readouterr().out
    assert "2 callsign extraction(s)" in output
    assert "1  (50%)" in output
    assert "c2.txt" in output  # the unsupported one is shown for reading
