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

"""Properties the matcher must hold for *every* callsign, not just the ones a
net has already mangled.

The regression suite in `test_callsign_match.py` records misses that already
happened. This finds the ones that have not. The four normalizer bugs fixed on
2026-08-12 -- `9er`, `9 or`, `III`, `Alpha4PQ` -- were each a single mechanical
mutation of a correctly-spelled callsign, and every one of them would have been
caught here before a net ever met it.

The trick that makes this possible without labelling anything: `callsign_match`
already knows how to *speak* a callsign (`_spell_phonetically`). So a callsign
can be spelled, mutated the way Whisper is known to render text it heard
correctly, and asserted to survive the round trip. The roster is the ground
truth, as everywhere else in this project.

Two tiers, because they are different claims:

- **Lossless mutations must still match.** Hyphenation, gluing, ordinals, Roman
  numerals, a split "niner" -- none of these destroy information, so failing to
  match one is a bug in the normalizer.
- **No mutation may ever produce a *different* roster callsign.** This is the
  stronger and more important property. A lossy rendering is allowed to come
  back unmatched; it is never allowed to come back as somebody else, because a
  wrong callsign puts a station at the wrong point on the race course.

Generation is seeded rather than random so a failure is reproducible from the
output alone, and needs no `hypothesis` dependency -- this suite has to keep
running in the minimal-dependency CI job.
"""

from __future__ import annotations

import os
import random
import string
from pathlib import Path

import pytest
from rapidfuzz import fuzz

from callsign_match import (
    _DIGIT_TO_SPOKEN,
    _LETTER_TO_PHONETIC,
    _spell_phonetically,
    CallsignMatcher,
    RosterEntry,
    load_roster,
    normalize,
)

SEED = 20260812
ROSTER_SIZE = 40
MAX_ROSTER_SIMILARITY = 70
"""Roster entries are kept mutually distant so the matcher's ambiguity refusal
-- which is correct behaviour -- cannot be mistaken for a failure here."""


# --------------------------------------------------------------------------
# Generating callsigns and a roster that does not fight itself
# --------------------------------------------------------------------------


def a_callsign(rng: random.Random) -> str:
    """A US-shaped callsign: 1-2 letter prefix, a digit, 1-3 letter suffix."""
    letters = string.ascii_uppercase
    prefix = "".join(rng.choice(letters) for _ in range(rng.randint(1, 2)))
    suffix = "".join(rng.choice(letters) for _ in range(rng.randint(1, 3)))
    return f"{prefix}{rng.choice(string.digits)}{suffix}"


def a_roster(size: int = ROSTER_SIZE, seed: int = SEED) -> list[str]:
    rng = random.Random(seed)
    chosen: list[str] = []
    while len(chosen) < size:
        candidate = a_callsign(rng)
        if candidate in chosen:
            continue
        if all(fuzz.ratio(candidate, other) < MAX_ROSTER_SIMILARITY for other in chosen):
            chosen.append(candidate)
    return chosen


ROSTER = a_roster()
MATCHER = CallsignMatcher(roster=[RosterEntry(c, "Op") for c in ROSTER])


# --------------------------------------------------------------------------
# The ways Whisper writes a callsign it heard correctly
#
# Each takes the spoken form and returns how the model might render it. These
# are *renderings*, not mishearings: they are choices about how to write text
# the model got right, so unlike a synthesizer artifact they generalise to real
# audio. See docs/TESTING.md for why that distinction decides what belongs here.
# --------------------------------------------------------------------------

ROMAN = {"2": "II", "3": "III", "4": "IV", "6": "VI", "7": "VII", "8": "VIII", "9": "IX"}
ORDINAL = {
    "1": "first", "2": "second", "3": "third", "4": "fourth", "5": "fifth",
    "6": "sixth", "7": "seventh", "8": "eighth", "9": "ninth",
}


def spoken(callsign: str) -> str:
    return _spell_phonetically(callsign)


def hyphenated(callsign: str) -> str:
    """"kilo delta niner" -> "kilo-delta niner". Whisper hyphenates freely."""
    words = spoken(callsign).split()
    return " ".join(
        f"{a}-{b}" for a, b in zip(words[::2], words[1::2])
    ) + ("" if len(words) % 2 == 0 else " " + words[-1])


def glued(callsign: str) -> str:
    """Adjacent phonetics run together with no space."""
    words = spoken(callsign).split()
    return "".join(words[:2]) + (" " + " ".join(words[2:]) if len(words) > 2 else "")


def numeric_digit(callsign: str) -> str:
    """The digit written as a numeral rather than spelled."""
    digit = next(c for c in callsign if c.isdigit())
    return spoken(callsign).replace(_DIGIT_TO_SPOKEN[digit], digit, 1)


def ordinal_digit(callsign: str) -> str:
    """A spoken digit between phonetics comes back as an ordinal."""
    digit = next(c for c in callsign if c.isdigit())
    if digit not in ORDINAL:
        return spoken(callsign)
    return spoken(callsign).replace(_DIGIT_TO_SPOKEN[digit], ORDINAL[digit], 1)


def numeric_ordinal_digit(callsign: str) -> str:
    """...and sometimes as a numeric ordinal: "5th"."""
    digit = next(c for c in callsign if c.isdigit())
    if digit == "0":
        return spoken(callsign)
    suffix = {"1": "st", "2": "nd", "3": "rd"}.get(digit, "th")
    return spoken(callsign).replace(_DIGIT_TO_SPOKEN[digit], f"{digit}{suffix}", 1)


def roman_digit(callsign: str) -> str:
    digit = next(c for c in callsign if c.isdigit())
    if digit not in ROMAN:
        return spoken(callsign)
    return spoken(callsign).replace(_DIGIT_TO_SPOKEN[digit], ROMAN[digit], 1)


def niner_er(callsign: str) -> str:
    """"niner" written half in digits."""
    return spoken(callsign).replace("niner", "9er")


def niner_split(callsign: str) -> str:
    """"niner" split across two words, orphaning its second syllable."""
    return spoken(callsign).replace("niner", "9 or")


def welded(callsign: str) -> str:
    """A phonetic welded to the digit and the collapsed suffix: "alpha4PQ"."""
    prefix, digit, suffix = _parts(callsign)
    lead = " ".join(_LETTER_TO_PHONETIC[c] for c in prefix[:-1])
    last = _LETTER_TO_PHONETIC[prefix[-1]]
    return f"{lead} {last}{digit}{suffix}".strip()


def collapsed_suffix(callsign: str) -> str:
    """The suffix phonetics collapsed into the letters themselves: "7XY"."""
    prefix, digit, suffix = _parts(callsign)
    return " ".join(_LETTER_TO_PHONETIC[c] for c in prefix) + f" {digit}{suffix}"


def in_net_traffic(callsign: str) -> str:
    """Wrapped in the filler that surrounds a real check-in."""
    return f"Net control, this is {spoken(callsign)}, checking in, no traffic."


def shouted(callsign: str) -> str:
    return spoken(callsign).upper()


def title_cased(callsign: str) -> str:
    return spoken(callsign).title()


def _parts(callsign: str) -> tuple[str, str, str]:
    index = next(i for i, c in enumerate(callsign) if c.isdigit())
    return callsign[:index], callsign[index], callsign[index + 1 :]


LOSSLESS = [
    spoken, hyphenated, glued, numeric_digit, ordinal_digit,
    numeric_ordinal_digit, roman_digit, niner_er, niner_split,
    welded, collapsed_suffix, in_net_traffic, shouted, title_cased,
]
"""Renderings that keep every character of the callsign. Failing to match one
of these is a normalizer bug, not a limit of the audio."""


# --------------------------------------------------------------------------
# The properties
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mutate", LOSSLESS, ids=lambda f: f.__name__)
def test_every_roster_callsign_survives_this_rendering(mutate) -> None:
    """Tier one: a rendering that loses nothing must still match.

    Reported in bulk rather than one at a time, because "three of forty
    callsigns fail this rendering" is a different and more useful signal than
    the first failing example.
    """
    failures = []
    for callsign in ROSTER:
        text = mutate(callsign)
        result = MATCHER.match(text)
        if result.callsign != callsign:
            failures.append(
                f"    {callsign}: {text!r} -> {normalize(text)!r} "
                f"gave {result.callsign or result.reason}"
            )
    assert not failures, (
        f"\n{mutate.__name__} lost {len(failures)}/{len(ROSTER)} callsigns:\n"
        + "\n".join(failures[:8])
    )


@pytest.mark.parametrize("mutate", LOSSLESS, ids=lambda f: f.__name__)
def test_no_rendering_ever_produces_a_different_station(mutate) -> None:
    """Tier two, and the one that matters most.

    Coming back unmatched is a disappointment; coming back as *somebody else*
    sends help to the wrong part of the course. This must hold even for
    renderings tier one is allowed to fail.
    """
    wrong = []
    for callsign in ROSTER:
        result = MATCHER.match(mutate(callsign))
        if result.matched and result.callsign != callsign:
            wrong.append(f"    {callsign} came back as {result.callsign}")
    assert not wrong, f"\n{mutate.__name__} misattributed:\n" + "\n".join(wrong)


def test_ordinary_net_speech_is_never_a_callsign() -> None:
    """The other direction: over-firing is how a normalizer fix goes wrong.

    Every rule that recovers a callsign is a rule that could invent one, so the
    phrases a net says constantly must stay unmatched.
    """
    said_on_every_net = [
        "net control this is a check in with no traffic",
        "we have nine or ten people at the aid station",
        "runner down at mile 12 requesting medical",
        "road is clear at the crossing sweep vehicle is through",
        "IV started on the patient at mile 8",
        "go ahead with your traffic",
        "standing by for the next station",
        "copy that, thanks, out",
        "the last runner has passed my position",
        "medical is on scene situation is under control",
    ]
    matched = [
        f"    {phrase!r} -> {MATCHER.match(phrase).callsign}"
        for phrase in said_on_every_net
        if MATCHER.match(phrase).matched
    ]
    assert not matched, "\ninvented a callsign from net speech:\n" + "\n".join(matched)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_the_property_holds_on_other_rosters(seed) -> None:
    """The fixed roster could be lucky. These are not the same forty callsigns."""
    roster = a_roster(size=25, seed=seed)
    matcher = CallsignMatcher(roster=[RosterEntry(c, "Op") for c in roster])
    failures = []
    for callsign in roster:
        for mutate in (spoken, hyphenated, numeric_digit, niner_split, welded):
            result = matcher.match(mutate(callsign))
            if result.callsign != callsign:
                failures.append(f"    seed {seed} {mutate.__name__} {callsign}")
    assert not failures, "\n" + "\n".join(failures[:8])


def test_the_generated_roster_is_actually_hard() -> None:
    """A guard on the test itself.

    If generation drifted toward callsigns that are trivially far apart, the
    suite above would keep passing while testing nothing. Every shape a US
    callsign can take should appear.
    """
    shapes = {
        (len(p), len(s))
        for p, _, s in (_parts(c) for c in a_roster(size=120, seed=SEED))
    }
    assert shapes == {(a, b) for a in (1, 2) for b in (1, 2, 3)}


# --------------------------------------------------------------------------
# The same properties, against the roster you actually run
#
# Generated callsigns are deliberately spread out. A real roster is not: event
# nets collect callsigns that genuinely resemble each other, and that is worth
# knowing about *before* the net rather than during it. Skipped when there is
# no roster to read, so CI is unaffected.
# --------------------------------------------------------------------------

REAL_ROSTER = Path(os.environ.get("NETCONTROLLER_ROSTER", "roster.csv"))

real_roster_only = pytest.mark.skipif(
    not REAL_ROSTER.exists(), reason=f"no roster at {REAL_ROSTER}"
)


def _real() -> tuple[CallsignMatcher, list[str]]:
    entries = load_roster(REAL_ROSTER)
    return CallsignMatcher(roster=entries), [e.callsign for e in entries]


@real_roster_only
def test_your_roster_is_never_misattributed() -> None:
    """The property that must hold whatever your roster looks like.

    Two similar callsigns may be impossible to tell apart -- that is a fact
    about the roster, and the matcher is meant to refuse. What it must never do
    is pick the wrong one.
    """
    matcher, roster = _real()
    wrong = [
        f"    {callsign} came back as {result.callsign} from {mutate.__name__}"
        for callsign in roster
        for mutate in LOSSLESS
        for result in [matcher.match(mutate(callsign))]
        if result.matched and result.callsign != callsign
    ]
    assert not wrong, "\n" + "\n".join(wrong[:12])


@real_roster_only
def test_your_roster_survives_every_rendering() -> None:
    """Which of your stations the matcher can recover from any rendering.

    Callsigns with a close neighbour on the roster are reported rather than
    failed: refusing to guess between two similar stations is correct, and the
    useful output is *which* stations those are, so net control knows which
    ones will need confirming by ear.
    """
    matcher, roster = _real()
    ambiguous = {
        c for c in roster
        if any(fuzz.ratio(c, o) >= MAX_ROSTER_SIMILARITY for o in roster if o != c)
    }
    failures = [
        f"    {callsign}: {mutate.__name__} -> {result.callsign or result.reason}"
        for callsign in roster
        if callsign not in ambiguous
        for mutate in LOSSLESS
        for result in [matcher.match(mutate(callsign))]
        if result.callsign != callsign
    ]
    if ambiguous:
        print(
            f"\n{len(ambiguous)} callsign(s) on this roster are close enough to "
            f"another that the matcher will refuse rather than guess, and net "
            f"control should expect to confirm them by ear:\n    "
            + ", ".join(sorted(ambiguous))
        )
    assert not failures, "\n" + "\n".join(failures[:12])
