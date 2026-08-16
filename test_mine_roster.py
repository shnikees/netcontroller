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

"""Tests for mining a draft roster out of recorded nets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "tools"))

from mine_roster import mine, sessions_from  # noqa: E402


def write_session(directory: Path, name: str, texts: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"net-{name}.jsonl").write_text(
        "\n".join(
            json.dumps({"type": "entry", "id": i, "raw_text": t})
            for i, t in enumerate(texts, 1)
        )
        + "\n",
        encoding="utf-8",
    )


def test_a_station_on_several_nets_outranks_a_noisy_one(tmp_path) -> None:
    """The whole idea: recurrence separates stations from transcription noise."""
    write_session(tmp_path, "01", ["whiskey six alpha bravo charlie checking in"])
    write_session(tmp_path, "02", ["whiskey six alpha bravo charlie again"])
    write_session(
        tmp_path,
        "03",
        # One garbled night that mentions the same nonsense repeatedly.
        ["kilo seven x ray 5 " * 1, "kilo seven x ray 5 ", "kilo seven x ray 5 "],
    )
    found = mine(sessions_from(tmp_path))
    real = found["W6ABC"]
    assert len(real["sessions"]) == 2

    # Whatever the garbled night produced, it came from one session only, so it
    # must not outrank the real station however often it was repeated.
    for callsign, record in found.items():
        if callsign != "W6ABC":
            assert len(record["sessions"]) == 1, callsign


def test_mentions_never_beat_distinct_sessions(tmp_path) -> None:
    """One bad transmission repeated is still one piece of evidence."""
    write_session(tmp_path, "01", ["november five delta echo foxtrot"] * 9)
    write_session(tmp_path, "02", ["kilo delta niner mike november oscar"])
    write_session(tmp_path, "03", ["kilo delta niner mike november oscar"])
    found = mine(sessions_from(tmp_path))
    assert found["N5DEF"]["mentions"] == 9
    assert len(found["N5DEF"]["sessions"]) == 1
    assert len(found["KD9MNO"]["sessions"]) == 2

    ranked = sorted(found, key=lambda c: -len(found[c]["sessions"]))
    assert ranked[0] == "KD9MNO", "a repeated one-off outranked a real regular"


def test_only_properly_shaped_callsigns_are_proposed(tmp_path) -> None:
    """Loose shapes exist so an operator can confirm by ear. Nobody is going to
    confirm a roster entry that was never a callsign."""
    write_session(tmp_path, "01", ["this is K7 and also W6ABC here"])
    write_session(tmp_path, "02", ["this is K7 and also W6ABC here"])
    found = mine(sessions_from(tmp_path))
    assert "W6ABC" in found
    assert "K7" not in found


def test_plain_conversation_proposes_nobody(tmp_path) -> None:
    write_session(tmp_path, "01", ["good morning everyone, nice weather today"])
    write_session(tmp_path, "02", ["I see you at the meeting later, thanks"])
    assert mine(sessions_from(tmp_path)) == {}


def test_a_truncated_final_line_does_not_stop_the_mining(tmp_path) -> None:
    """A power cut mid-write leaves half a line; the rest is still evidence."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "net-01.jsonl").write_text(
        json.dumps({"type": "entry", "id": 1, "raw_text": "whiskey six alpha bravo charlie"})
        + '\n{"type": "entry", "id": 2, "raw_te',
        encoding="utf-8",
    )
    assert "W6ABC" in mine(sessions_from(tmp_path))


def test_the_state_is_read_rather_than_sliced_off_the_end() -> None:
    """Slicing the tail of an address looked equivalent and was not -- an
    address ending in a zip+4 or a unit number printed digits where a state
    belonged."""
    from mine_roster import _state_of

    assert _state_of("SEATTLE, WA 98177") == "WA"
    assert _state_of("SOMEWHERE, WA 98101-1725") == "WA"
    assert _state_of("APT 101-1725") == ""
    assert _state_of("") == ""


def test_non_us_prefixes_are_not_called_invalid() -> None:
    """A US licence database cannot answer for a VE7, and the VE7s across the
    border are regulars on a Puget Sound repeater rather than noise."""
    from mine_roster import look_up

    cache: dict = {}
    for callsign in ("VE7NA", "VA7XYZ", "G7ABC"):
        assert look_up(callsign, cache)["status"] == "non-us", callsign
    # ...and nothing was queried over the network to find that out.
    assert set(cache) == {"VE7NA", "VA7XYZ", "G7ABC"}


def test_a_cached_answer_is_not_looked_up_again() -> None:
    from mine_roster import look_up

    cache = {"W6ABC": {"status": "valid", "state": "CA", "grid": "CM87"}}
    assert look_up("W6ABC", cache)["state"] == "CA"


def test_grid_distance_is_roughly_right() -> None:
    from mine_roster import km_apart

    seattle_to_vegas = km_apart("CN87", "DM26")
    assert 1300 < seattle_to_vegas < 1800, seattle_to_vegas
    assert km_apart("CN87", "") is None
