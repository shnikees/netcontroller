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

"""Normalize Whisper output into callsign candidates and match them to a roster.

This is the highest-risk part of the pipeline: Whisper hears "whiskey six alpha
bravo charlie" as anything from "Whiskey 6 Alpha Bravo Charlie" to "whisky sex
alfabravo charlie", and net control needs the right roster entry out the far end.

The flow is:

    raw text -> normalize() -> extract_candidates() -> match() -> MatchResult

Each stage is separately testable; see test_callsign_match.py.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from rapidfuzz import fuzz, process

# --------------------------------------------------------------------------
# Phonetic + digit vocabulary
# --------------------------------------------------------------------------

# Standard NATO/ITU phonetics plus the spellings Whisper actually emits. Keys
# are lowercase, whitespace-free. Anything added here should be something a
# transcript has plausibly contained -- this table is the main lever for
# improving matching against real net audio.
PHONETIC_MAP: dict[str, str] = {
    # A
    "alpha": "A", "alfa": "A", "alph": "A",
    # B
    "bravo": "B", "brava": "B", "bravos": "B",
    # C
    "charlie": "C", "charley": "C", "charly": "C",
    # D
    "delta": "D", "adelta": "D",
    # E
    "echo": "E", "eco": "E", "ekko": "E",
    # F
    "foxtrot": "F", "fox": "F", "foxtrott": "F",
    # G
    "golf": "G", "gulf": "G",
    # H
    "hotel": "H", "hotell": "H",
    # I
    "india": "I", "indias": "I",
    # J
    "juliet": "J", "juliett": "J", "julliet": "J", "juliette": "J",
    # K
    "kilo": "K", "keelo": "K", "kilos": "K",
    # L
    "lima": "L", "leema": "L", "limo": "L",
    # M
    "mike": "M", "mic": "M", "mikes": "M",
    # N
    "november": "N", "novembre": "N",
    # O
    "oscar": "O", "oskar": "O",
    # P
    "papa": "P", "poppa": "P", "papas": "P",
    # Q
    "quebec": "Q", "quebeck": "Q", "kebec": "Q", "quibbic": "Q", "quibec": "Q",
    # R
    "romeo": "R", "romio": "R",
    # S
    "sierra": "S", "siera": "S", "sera": "S",
    # T
    "tango": "T", "tanga": "T",
    # U
    "uniform": "U", "unicorn": "U",
    # V
    "victor": "V", "viktor": "V",
    # W
    "whiskey": "W", "whisky": "W", "wiskey": "W",
    # X
    "xray": "X", "x-ray": "X", "exray": "X",
    # Y
    "yankee": "Y", "yanky": "Y", "yankie": "Y",
    # Z
    "zulu": "Z", "zuluu": "Z",
}

DIGIT_MAP: dict[str, str] = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3", "tree": "3",
    "four": "4",
    "five": "5", "fife": "5",
    "six": "6", "sex": "6", "sixe": "6",
    "seven": "7", "sevin": "7",
    "eight": "8",
    "nine": "9", "niner": "9", "niner's": "9",
}

# Homophones that are digits inside a spelled callsign but ordinary English
# everywhere else ("checking in to the net" must not yield a 2). These convert
# only when flanked by spelling tokens -- see _is_digit_position.
#
# The ordinals are here because Whisper reliably renders a spoken digit between
# two phonetics as an ordinal: "november five delta" comes back as "november
# fifth delta".
AMBIGUOUS_DIGIT_MAP: dict[str, str] = {
    "oh": "0", "owe": "0",
    "won": "1", "wun": "1", "first": "1",
    "to": "2", "too": "2", "second": "2",
    "third": "3",
    "for": "4", "fore": "4", "fourth": "4",
    "fifth": "5",
    "sixth": "6",
    "seventh": "7",
    "ate": "8", "eighth": "8",
    "ninth": "9",
}

# Words that show up glued around callsigns in net traffic and should never be
# treated as part of one. Digit homophones like "for" are handled by
# AMBIGUOUS_DIGIT_MAP instead, and are dropped as filler when no callsign
# context surrounds them.
FILLER_WORDS: frozenset[str] = frozenset(
    {
        "this", "is", "here", "and", "the", "a", "de", "from",
        "net", "control", "station", "check", "checking", "in", "into",
        "over", "out", "back", "clear", "roger", "copy", "thanks", "thank",
        "you", "please", "go", "ahead", "good", "morning", "evening",
        "afternoon", "hello", "hi", "my", "callsign", "call", "sign",
        "sign's", "name", "with", "traffic", "no", "nothing", "qni", "qrz",
        "qsl", "qth", "monitoring", "listening", "standing", "by",
    }
)

# US amateur callsign structure: 1-2 letter prefix, one digit, 1-3 letter suffix.
CALLSIGN_RE = re.compile(r"\b([A-Z]{1,2}[0-9][A-Z]{1,3})\b")

# Digit-bearing token that is *nearly* callsign shaped -- used to salvage
# candidates that a mis-heard letter pushed outside the strict pattern.
LOOSE_CALLSIGN_RE = re.compile(r"\b([A-Z]{1,3}[0-9][A-Z]{0,4})\b")


@dataclass(frozen=True)
class RosterEntry:
    callsign: str
    name: str = ""


@dataclass
class MatchResult:
    """Outcome of matching one transmission against the roster."""

    matched: bool
    callsign: str | None = None
    name: str = ""
    score: float = 0.0
    candidate: str | None = None
    """The callsign-shaped token we matched from, before roster correction."""
    runner_up: str | None = None
    runner_up_score: float = 0.0
    reason: str = ""
    """Why an unmatched result was rejected: no_candidate/below_threshold/ambiguous."""


# --------------------------------------------------------------------------
# Roster loading
# --------------------------------------------------------------------------


def load_roster(path: str | Path) -> list[RosterEntry]:
    """Load a `callsign,name` CSV. A header row is optional; name is optional."""
    entries: list[RosterEntry] = []
    seen: set[str] = set()
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row or not row[0].strip():
                continue
            callsign = row[0].strip().upper()
            if callsign in {"CALLSIGN", "CALL", "CALL_SIGN"}:
                continue  # header
            if callsign in seen:
                continue
            seen.add(callsign)
            name = row[1].strip() if len(row) > 1 else ""
            entries.append(RosterEntry(callsign=callsign, name=name))
    return entries


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------


def _split_glued_phonetics(token: str) -> list[str]:
    """Split tokens Whisper ran together, e.g. "alfabravo" -> ["alfa", "bravo"].

    Only splits when the *entire* token decomposes into known phonetic/digit
    words, so ordinary English is left alone.
    """
    vocab = {**PHONETIC_MAP, **DIGIT_MAP}
    max_word = max(len(w) for w in vocab)


    n = len(token)
    # best[i] = word list spelling token[:i], or None if unreachable.
    best: list[list[str] | None] = [None] * (n + 1)
    best[0] = []
    for i in range(1, n + 1):
        for length in range(1, min(max_word, i) + 1):
            start = i - length
            if best[start] is None:
                continue
            piece = token[start:i]
            if piece in vocab:
                best[i] = best[start] + [piece]
                break
    result = best[n]
    # A single-word "split" is not a split; require at least two pieces so we
    # don't churn on tokens that were already fine.
    return result if result and len(result) > 1 else [token]


def normalize(text: str) -> str:
    """Turn spoken-form transcript text into a compact letters/digits string.

    Phonetics become letters, spoken digits become numerals, filler is dropped,
    and runs of adjacent single characters are glued into callsign-shaped tokens.
    """
    lowered = text.lower()
    # Keep letters, digits, and the hyphen/apostrophe that appear inside
    # vocabulary entries ("x-ray", "niner's"); everything else is a separator.
    raw_tokens = re.findall(r"[a-z0-9'\-]+", lowered)

    expanded: list[str] = []
    for token in raw_tokens:
        for piece in _split_glued_phonetics(token):
            expanded.append(piece)

    out: list[str] = []
    for i, token in enumerate(expanded):
        if token in PHONETIC_MAP:
            out.append(PHONETIC_MAP[token])
        elif token in DIGIT_MAP:
            out.append(DIGIT_MAP[token])
        elif token in AMBIGUOUS_DIGIT_MAP:
            if _is_digit_position(expanded, i):
                out.append(AMBIGUOUS_DIGIT_MAP[token])
            # Otherwise it is ordinary English; drop it like other filler.
        elif token in FILLER_WORDS:
            continue
        elif re.fullmatch(r"[0-9]+", token):
            out.append(token)
        else:
            out.append(token.replace("-", "").replace("'", "").upper())

    return _glue_singles(out)


def _is_digit_position(tokens: list[str], index: int) -> bool:
    """True if the token at `index` sits where a callsign's digit would sit.

    US callsigns put the digit between prefix and suffix letters, so an
    ambiguous homophone only counts as a digit when it is flanked on *both*
    sides by spelling tokens. "foxtrot for net control" keeps "for" as filler;
    "alpha for bravo" reads it as A-4-B.
    """
    if index == 0 or index + 1 >= len(tokens):
        return False
    return all(_is_spelling_token(tokens[i]) for i in (index - 1, index + 1))


def _is_spelling_token(token: str) -> bool:
    return (
        token in PHONETIC_MAP
        or token in DIGIT_MAP
        or (len(token) == 1 and token.isalnum())
    )


def _glue_singles(tokens: list[str]) -> str:
    """Join runs of 1-char tokens into words: [W,6,A,B,C] -> "W6ABC"."""
    result: list[str] = []
    run: list[str] = []
    for token in tokens:
        if len(token) == 1:
            run.append(token)
        else:
            if run:
                result.append("".join(run))
                run = []
            result.append(token)
    if run:
        result.append("".join(run))
    return " ".join(result)


# --------------------------------------------------------------------------
# Candidate extraction
# --------------------------------------------------------------------------


def extract_candidates(normalized: str) -> list[str]:
    """Pull callsign-shaped tokens out of a normalized string, best first."""
    text = normalized.upper()
    strict = CALLSIGN_RE.findall(text)
    loose = [c for c in LOOSE_CALLSIGN_RE.findall(text) if c not in strict]
    # Strict hits first; among loose hits, longer tokens carry more signal.
    return strict + sorted(loose, key=len, reverse=True)


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------


@dataclass
class CallsignMatcher:
    """Fuzzy-matches transcript text against a fixed roster.

    threshold: minimum rapidfuzz score (0-100) to accept a match. The default
        of 78 is just under the 80 that one wrong character in a five-character
        callsign scores, so a single Whisper slip still matches; two do not.
    ambiguity_margin: if the runner-up is within this many points of the best
        match, the result is rejected as ambiguous rather than guessed at --
        net control would rather see "unmatched" than a confident wrong call.
    """

    roster: list[RosterEntry]
    threshold: float = 78.0
    ambiguity_margin: float = 5.0
    _by_callsign: dict[str, RosterEntry] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._by_callsign = {e.callsign: e for e in self.roster}

    @property
    def callsigns(self) -> list[str]:
        return list(self._by_callsign)

    def hotwords(self, extra_vocabulary: list[str] | None = None) -> str:
        """Build a Whisper `initial_prompt` biasing decoding toward the roster."""
        spoken = [_spell_phonetically(c) for c in self._by_callsign]
        parts = list(self._by_callsign) + spoken + list(extra_vocabulary or [])
        return "Amateur radio net check-ins. " + ", ".join(parts) + "."

    def match(self, text: str) -> MatchResult:
        normalized = normalize(text)
        candidates = extract_candidates(normalized)
        if not candidates:
            return MatchResult(matched=False, reason="no_candidate")

        best: MatchResult | None = None
        for candidate in candidates:
            result = self._match_candidate(candidate)
            if result.matched:
                return result
            if best is None or result.score > best.score:
                best = result
        assert best is not None
        return best

    def _match_candidate(self, candidate: str) -> MatchResult:
        scored = process.extract(
            candidate,
            self._by_callsign.keys(),
            scorer=fuzz.ratio,
            limit=2,
        )
        if not scored:
            return MatchResult(matched=False, candidate=candidate, reason="no_candidate")

        top_call, top_score, _ = scored[0]
        runner_up, runner_up_score = (None, 0.0)
        if len(scored) > 1:
            runner_up, runner_up_score = scored[1][0], scored[1][1]

        base = MatchResult(
            matched=False,
            candidate=candidate,
            score=top_score,
            runner_up=runner_up,
            runner_up_score=runner_up_score,
        )

        if top_score < self.threshold:
            base.reason = "below_threshold"
            return base
        # An exact hit is never ambiguous, even if a roster neighbor scores close.
        if (
            candidate != top_call
            and runner_up is not None
            and top_score - runner_up_score < self.ambiguity_margin
        ):
            base.reason = "ambiguous"
            return base

        entry = self._by_callsign[top_call]
        base.matched = True
        base.callsign = entry.callsign
        base.name = entry.name
        return base


_LETTER_TO_PHONETIC = {
    "A": "alpha", "B": "bravo", "C": "charlie", "D": "delta", "E": "echo",
    "F": "foxtrot", "G": "golf", "H": "hotel", "I": "india", "J": "juliet",
    "K": "kilo", "L": "lima", "M": "mike", "N": "november", "O": "oscar",
    "P": "papa", "Q": "quebec", "R": "romeo", "S": "sierra", "T": "tango",
    "U": "uniform", "V": "victor", "W": "whiskey", "X": "xray", "Y": "yankee",
    "Z": "zulu",
}
_DIGIT_TO_SPOKEN = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "niner",
}


def _spell_phonetically(callsign: str) -> str:
    return " ".join(
        _LETTER_TO_PHONETIC.get(ch, _DIGIT_TO_SPOKEN.get(ch, ch)) for ch in callsign
    )
