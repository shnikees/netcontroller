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
    "kilo": "K", "keelo": "K", "kilos": "K", "kelo": "K", "killo": "K",
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

BREAK = "\x00"
"""Marks where a word was dropped, so gluing does not run across the gap."""

# US amateur callsign structure: 1-2 letter prefix, one digit, 1-3 letter suffix.
CALLSIGN_RE = re.compile(r"\b([A-Z]{1,2}[0-9][A-Z]{1,3})\b")

# Digit-bearing token that is *nearly* callsign shaped -- used to salvage
# candidates that a mis-heard letter pushed outside the strict pattern.
LOOSE_CALLSIGN_RE = re.compile(r"\b([A-Z]{1,3}[0-9][A-Z]{0,4})\b")


@dataclass(frozen=True)
class RosterEntry:
    callsign: str
    name: str = ""
    sources: tuple[str, ...] = ()
    """Which receivers this station is expected on; empty means all of them.

    With 40-100 stations across two frequencies, no single prompt can bias
    Whisper toward all of them -- see CallsignMatcher.bias_terms.
    """

    def on_source(self, source: str) -> bool:
        return not self.sources or not source or source in self.sources


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
    via_alias: bool = False
    """True when an operator correction, not fuzzy matching, produced this."""
    start: int = -1
    end: int = -1
    """Where in the raw transcript this callsign was heard; -1 when unknown."""


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
            # Optional third column: the frequencies this station is expected
            # on, separated by ";" or "|" (a comma would break the CSV).
            raw_sources = row[2].strip() if len(row) > 2 else ""
            sources = tuple(
                part.strip()
                for part in re.split(r"[;|]", raw_sources)
                if part.strip()
            )
            entries.append(
                RosterEntry(callsign=callsign, name=name, sources=sources)
            )
    return entries


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------


ORDINAL_SUFFIX_RE = re.compile(r"^([0-9]+)(st|nd|rd|th)$")


def _split_alphanumeric(token: str) -> list[str]:
    """Pull apart tokens where Whisper mixed digits and letters in one word.

    Prompting the model with the phonetic alphabet makes it write callsigns in
    a more clipped style, and these are the forms that come back:

        "5th"        an ordinal written numerically  -> 5
        "3zulu"      a digit welded to a phonetic    -> 3, zulu
        "7xy"        a run of spelled characters     -> 7, x, y

    Left alone each of these blocks a match that would otherwise have worked.
    """
    ordinal = ORDINAL_SUFFIX_RE.match(token)
    if ordinal:
        return [ordinal.group(1)]

    leading = re.fullmatch(r"([0-9]+)([a-z]+)", token)
    if leading and leading.group(2) in PHONETIC_MAP:
        return [leading.group(1), leading.group(2)]
    trailing = re.fullmatch(r"([a-z]+)([0-9]+)", token)
    if trailing and trailing.group(1) in PHONETIC_MAP:
        return [trailing.group(1), trailing.group(2)]

    # A short mixed run is the model collapsing spelled-out characters
    # ("x-ray yankee" -> "XY"); splitting it back recovers them.
    if (leading or trailing) and len(token) <= 5:
        return list(token)
    return [token]


def _split_hyphens(token: str) -> list[str]:
    """Split hyphen-joined words, unless the hyphen belongs to the word itself.

    Whisper hyphenates adjacent spelled-out words freely -- "kilo juliet
    six-tango uniform victor" -- and treating that as one token loses both the
    digit and the letter. But "x-ray" is a vocabulary entry in its own right, so
    anything already in the tables is left alone.
    """
    if "-" not in token or token in PHONETIC_MAP or token in DIGIT_MAP:
        return [token]
    return [part for part in token.split("-") if part]


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


@dataclass(frozen=True)
class Token:
    """A normalized token, and the span of raw text it came from."""

    text: str
    start: int
    end: int


def normalize(text: str) -> str:
    """Turn spoken-form transcript text into a compact letters/digits string.

    Phonetics become letters, spoken digits become numerals, filler is dropped,
    and runs of adjacent single characters are glued into callsign-shaped tokens.
    """
    return " ".join(token.text for token in tokenize(text))


def tokenize(text: str) -> list[Token]:
    """normalize(), but keeping each token's position in the original text.

    The positions are what let a clip containing two stations be split at the
    right place: find the callsigns in the text, then find the pause between
    them in the word timings.
    """
    lowered = text.lower()
    # Keep letters, digits, and the hyphen/apostrophe that appear inside
    # vocabulary entries ("x-ray", "niner's"); everything else is a separator.
    matches = list(re.finditer(r"[a-z0-9'\-]+", lowered))

    expanded: list[str] = []
    spans: list[tuple[int, int]] = []
    for match in matches:
        for hyphenated in _split_hyphens(match.group()):
            for part in _split_alphanumeric(hyphenated):
                for piece in _split_glued_phonetics(part):
                    expanded.append(piece)
                    # Every piece of a split token points at the whole token;
                    # good enough to locate a callsign, simpler than sub-spans.
                    spans.append((match.start(), match.end()))

    out: list[Token] = []
    for i, token in enumerate(expanded):
        start, end = spans[i]

        def emit(value: str) -> None:
            out.append(Token(value, start, end))

        if token in PHONETIC_MAP:
            emit(PHONETIC_MAP[token])
        elif token in DIGIT_MAP:
            emit(DIGIT_MAP[token])
        elif token in AMBIGUOUS_DIGIT_MAP:
            if _is_digit_position(expanded, i):
                emit(AMBIGUOUS_DIGIT_MAP[token])
            else:
                emit(BREAK)
        elif token in FILLER_WORDS:
            # A dropped word still separates what was on either side of it.
            # Without this, "W6ABC no traffic K7XYZ" leaves the two callsigns
            # adjacent and they weld into one nonsense token -- losing both
            # stations, which is exactly what happens on a fast net where two
            # people key up inside one VAD gap.
            emit(BREAK)
        elif re.fullmatch(r"[0-9]+", token):
            emit(token)
        else:
            emit(token.replace("-", "").replace("'", "").upper())

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


def _glue_singles(tokens: list[Token]) -> list[Token]:
    """Join runs of 1-char tokens into words: [W,6,A,B,C] -> "W6ABC".

    BREAK markers end a run without contributing anything themselves, so words
    that were dropped still keep their neighbours apart. A glued token spans
    from the first contributing raw word to the last.
    """
    result: list[Token] = []
    run: list[Token] = []

    def flush() -> None:
        if run:
            result.append(
                Token("".join(t.text for t in run), run[0].start, run[-1].end)
            )
            run.clear()

    for token in tokens:
        if token.text == BREAK:
            flush()
        elif len(token.text) == 1:
            run.append(token)
        else:
            flush()
            result.append(token)
    flush()
    return result


# --------------------------------------------------------------------------
# Candidate extraction
# --------------------------------------------------------------------------


def extract_candidate_tokens(tokens: list[Token]) -> list[Token]:
    """Callsign-shaped tokens, in the order they were spoken.

    Unlike extract_candidates() this keeps positions and does not reorder, so
    the caller can line the callsigns up against the word timings.
    """
    found: list[Token] = []
    for token in tokens:
        text = token.text.upper()
        if CALLSIGN_RE.fullmatch(text) or LOOSE_CALLSIGN_RE.fullmatch(text):
            found.append(Token(text, token.start, token.end))
    return found


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
    aliases: dict[str, str] = field(default_factory=dict)
    """Learned candidate -> callsign corrections, e.g. {"E3Z": "K7XYZ"}."""
    _by_callsign: dict[str, RosterEntry] = field(init=False, repr=False)

    MIN_ALIAS_LENGTH = 3
    """Shorter candidates carry too little signal to key a correction on."""

    def __post_init__(self) -> None:
        self._by_callsign = {e.callsign: e for e in self.roster}
        # Drop aliases pointing at stations no longer on the roster; a stale
        # alias would silently resurrect a callsign the operator removed.
        self.aliases = {
            candidate: callsign
            for candidate, callsign in self.aliases.items()
            if callsign in self._by_callsign
        }

    def learn_alias(self, candidate: str | None, callsign: str) -> bool:
        """Record that `candidate` really means `callsign`. Returns whether it took.

        Called from the web thread while the capture thread reads `aliases`.
        Individual dict get/set are atomic under the GIL, and a correction that
        lands mid-transmission simply applies to the next one, so no lock is
        needed here.
        """
        if not candidate or len(candidate) < self.MIN_ALIAS_LENGTH:
            return False
        if callsign not in self._by_callsign:
            return False
        if candidate == callsign:
            return False  # nothing to learn; fuzzy matching already gets this
        self.aliases[candidate.upper()] = callsign
        return True

    @property
    def callsigns(self) -> list[str]:
        return list(self._by_callsign)

    def hotwords(self, extra_vocabulary: list[str] | None = None) -> str:
        """Whisper prompt for the whole roster. Small nets only -- see below."""
        return " ".join(self.bias_terms(extra_vocabulary))

    def bias_terms(
        self,
        extra_vocabulary: list[str] | None = None,
        *,
        source: str = "",
        heard: set[str] | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """Terms to bias decoding, most valuable first.

        Whisper's prompt window is 224 tokens, and a written callsign costs
        about four of them -- so roughly 48 fit, total. A net with 40-100
        stations across two frequencies cannot be covered by one prompt, and a
        prompt that overflows is silently truncated at an arbitrary point,
        which is worse than a short one chosen on purpose.

        So the list is ordered by how likely each station is to be the next
        voice on *this* receiver:

        1. The phonetic alphabet itself, then net vocabulary. Spelling out one
           callsign per station would cost seven tokens each; the alphabet they
           are all spelled from costs about thirty once, and biases toward
           every phonetic spelling on the net rather than a chosen few.
        2. Stations assigned to this source who have not checked in yet. On a
           check-in net those are precisely the ones about to speak.
        3. Stations assigned to this source who already have.
        4. Everyone else, in case somebody turns up on the wrong frequency.

        The caller trims to whatever the token budget allows; this only decides
        the order. Matching still runs against the entire roster, so a station
        that never makes the prompt is still matched correctly -- they just do
        not get the decoding hint.
        """
        heard = heard or set()
        mine = [e for e in self.roster if e.on_source(source)]
        others = [e for e in self.roster if not e.on_source(source)]

        pending = [e.callsign for e in mine if e.callsign not in heard]
        already = [e.callsign for e in mine if e.callsign in heard]
        elsewhere = [e.callsign for e in others]

        terms = (
            PHONETIC_ALPHABET
            + list(extra_vocabulary or [])
            + pending
            + already
            + elsewhere
        )
        return terms[:limit] if limit else terms

    def nearest(self, candidate: str, limit: int = 8) -> list[str]:
        """Roster callsigns closest to a heard token.

        This is what makes a second pass affordable on a big roster: instead of
        biasing toward 100 stations -- impossible inside the prompt window --
        bias toward the handful the first pass was already close to. A short,
        targeted list fits easily and pushes hard in the right direction.
        """
        if not candidate:
            return []
        scored = process.extract(
            candidate, self._by_callsign.keys(), scorer=fuzz.ratio, limit=limit
        )
        return [name for name, _score, _index in scored]

    def match_all(self, text: str) -> list[MatchResult]:
        """Every distinct roster station heard in this text, in order.

        A clip can catch two transmissions when stations key up inside one VAD
        gap. This finds each of them; deciding whether they really were two
        transmissions is the caller's job, and needs the word timings -- a
        transcript naming somebody else's callsign ("traffic for K7XYZ") looks
        identical here.
        """
        results: list[MatchResult] = []
        seen: set[str] = set()
        for token in extract_candidate_tokens(tokenize(text)):
            result = self._match_candidate(token.text)
            if not result.matched or result.callsign in seen:
                continue
            seen.add(result.callsign)
            result.start = token.start
            result.end = token.end
            results.append(result)
        return results

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
        # A learned correction is operator ground truth: it beats anything the
        # fuzzy matcher would have concluded, including an "ambiguous" refusal.
        learned = self.aliases.get(candidate)
        if learned is not None:
            entry = self._by_callsign[learned]
            return MatchResult(
                matched=True,
                callsign=entry.callsign,
                name=entry.name,
                score=100.0,
                candidate=candidate,
                via_alias=True,
            )

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


PHONETIC_ALPHABET: list[str] = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey",
    "xray", "yankee", "zulu", "niner",
]
"""Spoken once, biases every callsign spelled from it -- see bias_terms."""


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
