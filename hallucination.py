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

"""Catch the model quoting its own prompt back at us.

Whisper is told which callsigns to expect, because that is far and away the
biggest lever on getting them right. The cost is that when a clip carries
nothing intelligible -- a squelch tail, a burst of noise, someone keying up and
thinking better of it -- the model will sometimes emit the bias terms verbatim
instead of admitting it heard nothing. Measured on a real net, prompting *and*
hotwords together produced 505 apparent callsign recoveries across 690
transmissions, against 14 with no biasing at all. Most of the difference was
output like this:

    "KJ7RAB, KJ7RAB, KJ7JXM, KI7RMU."
    "VJ7RAB, KJ7JXM, KI7RMU, alpha bravo charlie delta echo foxtrot golf..."

which is the prompt read back, not a transmission.

This matters more here than a missed callsign ever could. An unmatched line
sends somebody to ask again; a manufactured one puts a station at a point on
the race course they were never at. The whole matcher is built to refuse rather
than guess, and biasing quietly reintroduces guessing one layer upstream.

The detection deliberately does *not* look at how many phonetic words a
transcript contains. A real check-in is nothing but phonetic words -- "kilo
juliet seven romeo alpha bravo" is 100% vocabulary and completely genuine. What
separates an echo from a check-in is not the words, it is that an echo names
*several different stations* with no speech around them, or names one station
several times over.
"""

from __future__ import annotations

import re
from collections import Counter

MAX_STATIONS_PER_CLIP = 2
"""One transmission naming another station is normal -- "W6ABC, traffic for
K7XYZ". Three separate stations in one clip is not something a half-duplex net
produces."""

MAX_REPEATS = 3
"""How often one callsign may appear before it reads as a stuck loop."""

MIN_WORDS_AROUND = 3
"""Two callsigns with almost no other words is a list, not a transmission."""

NATO = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey",
    "xray", "yankee", "zulu",
]
ALPHABET_RUN = 4
"""Consecutive letters of the phonetic alphabet, in canonical order, before it
reads as the model reciting the prompt. Nobody spells a callsign "alpha bravo
charlie delta" -- a callsign is at most six characters and its letters are not
in alphabetical order."""


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def looks_hallucinated(
    text: str,
    matches: list[str],
    *,
    bias_order: list[str] | None = None,  # accepted and ignored; see below
) -> tuple[bool, str]:
    """Whether this transcript looks like the prompt rather than a transmission.

    `matches` is every roster callsign the matcher found, in the order found.
    `bias_order` is accepted for callers that have it, but deliberately unused:
    flagging callsigns that appear in the order they were prompted sounds
    clever and is not, because with two callsigns the order matches half the
    time by chance. It threw out "KJ7RAB here, I have traffic for KJ7JXM",
    which is exactly the traffic this app exists to capture.

    Returns (suspect, reason). The reason is for the log and the dashboard --
    a line dropped without explanation is worse than one kept.
    """
    if not matches:
        return False, ""

    counts = Counter(matches)
    distinct = list(counts)

    if len(distinct) > MAX_STATIONS_PER_CLIP:
        return True, f"{len(distinct)} stations named in one transmission"

    worst = counts.most_common(1)[0]
    if worst[1] >= MAX_REPEATS:
        return True, f"{worst[0]} repeated {worst[1]} times"

    # Strip the callsigns out and see whether anything was actually said.
    residue = _words(text)
    for callsign in distinct:
        residue = [w for w in residue if w != callsign.lower()]
    if len(distinct) >= 2 and len(residue) < MIN_WORDS_AROUND:
        return True, "callsigns with no speech around them"

    if _recites_alphabet(text):
        return True, "the phonetic alphabet recited out of the prompt"

    return False, ""


def _recites_alphabet(text: str) -> bool:
    """Whether the text walks the phonetic alphabet in canonical order.

    This is the other half of the prompt coming back. The bias string carries
    the alphabet so that *any* spelled callsign decodes better, and a model
    with nothing to transcribe will happily read it out. It is unmistakable,
    because a real callsign is short and its letters are in no particular
    order.
    """
    position = {word: index for index, word in enumerate(NATO)}
    run = best = 0
    previous = None
    for word in _words(text):
        index = position.get(word)
        if index is not None and previous is not None and index == previous + 1:
            run += 1
        elif index is not None:
            run = 1
        else:
            run = 0
        previous = index
        best = max(best, run)
    return best >= ALPHABET_RUN


def filter_matches(
    text: str, matches: list[str], *, bias_order: list[str] | None = None
) -> tuple[list[str], str]:
    """The matches worth keeping, and why any were dropped.

    All or nothing on purpose. Once a transcript is judged to be the prompt
    coming back, there is no principled way to decide which of the callsigns in
    it were really said -- so none of them are trusted.
    """
    suspect, reason = looks_hallucinated(text, matches, bias_order=bias_order)
    return ([], reason) if suspect else (matches, "")
