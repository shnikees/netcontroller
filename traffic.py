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

"""Work out whether a transmission declared traffic.

A net exists partly to move traffic, and "who still has something to pass" is a
list net control keeps in their head otherwise. The phrasing is stereotyped
enough to read off the transcript.

**The trap is that the negative outnumbers the positive.** Far more stations
say "no traffic" than "with traffic", so a detector that just looks for the
word would flag the entire net and be worse than nothing -- net control would
learn to ignore the column inside one session. Negation is therefore the first
thing checked, not an afterthought.

Three outcomes, because "said they have nothing" and "did not say" are
different facts:

    HAS      "with traffic", "I have traffic for the county EOC"
    NONE     "no traffic", "nothing for the net", "nothing to pass"
    UNKNOWN  never mentioned it -- or asked about it, which is net control
             soliciting rather than a station declaring

Nothing here is a safety-critical decision: it colours a badge and fills a
column. Where the phrasing is genuinely ambiguous the answer is UNKNOWN, which
costs a badge rather than misleading somebody.
"""

from __future__ import annotations

import re

HAS = "yes"
NONE = "no"
UNKNOWN = ""

# Words that flip a mention of traffic into a denial of it. Checked in a short
# window before the word, which covers "no traffic", "I have no traffic" and
# "negative on traffic" without reaching into a neighbouring sentence.
NEGATORS = frozenset(
    {"no", "not", "negative", "without", "nothing", "none", "zero", "clear"}
)
NEGATION_WINDOW = 3

# Asking is not declaring. Net control saying "any traffic for the net?" must
# not mark net control as holding traffic.
SOLICITORS = frozenset({"any", "anyone", "anybody", "who", "whom"})
SOLICIT_WINDOW = 2

# Words that make a bare mention into a declaration: "I have traffic",
# "checking in with traffic", "got traffic for you".
DECLARERS = frozenset({"have", "got", "holding", "hold", "with", "carrying"})

# Phrases that settle it without the word "traffic" appearing at all.
NONE_PHRASES = (
    "nothing for the net",
    "nothing to pass",
    "nothing further",
    "nothing at this time",
    "nothing tonight",
)
HAS_PHRASES = (
    "traffic for",
    "with traffic",
    "priority traffic",
    "emergency traffic",
    "piece of traffic",
    "pieces of traffic",
)

# Traffic declared without ever using the word. Checked after the denials, so
# "nothing to pass" is already settled by the time these are looked at.
HAS_PHRASES_WITHOUT_THE_WORD = (
    "message for",
    "health and welfare",
    "something to pass",
    "one to pass",
)


def detect(text: str) -> str:
    """Return HAS, NONE or UNKNOWN for one transmission."""
    if not text:
        return UNKNOWN
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return UNKNOWN
    joined = " ".join(words)

    # A denial phrase is decisive: "nothing to pass" leaves nothing to weigh.
    if any(phrase in joined for phrase in NONE_PHRASES):
        return NONE

    mentions = [i for i, word in enumerate(words) if word.startswith("traffic")]
    if not mentions:
        # Plenty of traffic is offered without the word: "I have a message for
        # the incident commander" is a station holding something.
        if any(phrase in joined for phrase in HAS_PHRASES_WITHOUT_THE_WORD):
            return HAS
        return UNKNOWN

    verdicts = [_classify(words, index, joined) for index in mentions]
    # One genuine declaration outweighs other mentions: a station who says
    # "no traffic, but I have traffic for Turn 7" is holding traffic.
    if HAS in verdicts:
        return HAS
    if NONE in verdicts:
        return NONE
    return UNKNOWN


def _classify(words: list[str], index: int, joined: str) -> str:
    before = words[max(0, index - NEGATION_WINDOW) : index]
    if any(word in NEGATORS for word in before):
        return NONE

    # "any traffic", "does anyone have traffic" -- a question, not a report.
    asking = words[max(0, index - SOLICIT_WINDOW) : index]
    if any(word in SOLICITORS for word in asking):
        return UNKNOWN

    # "I have traffic", "checking in with traffic": the verb settles it, and
    # this has to come before the phrase table because the commonest
    # declaration of all is just those two words.
    if any(word in DECLARERS for word in before):
        return HAS

    following = " ".join(words[index : index + 4])
    if any(phrase in following or phrase in joined for phrase in HAS_PHRASES):
        return HAS

    # "traffic" alone, with nothing pointing either way -- a race net saying
    # "traffic is backing up on the access road", for instance.
    return UNKNOWN


def summarise(entries) -> dict:
    """Counts for the dashboard: who is holding traffic, who has cleared."""
    holding = {
        e.matched_callsign
        for e in entries
        if e.traffic == HAS and e.matched and e.matched_callsign
    }
    return {
        "holding": sorted(holding),
        "lines_with_traffic": sum(1 for e in entries if e.traffic == HAS),
        "lines_without": sum(1 for e in entries if e.traffic == NONE),
    }
