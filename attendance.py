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

"""Who actually turns up, learned from the nets already run.

The roster is a list of who *might* be on. The transcripts record who was, and
that is a better prior for biasing decoding -- especially on an event net,
where everyone speaks repeatedly and "who has not checked in yet" says almost
nothing about who is about to talk.

Two things are weighed:

- **Frequency.** A station on every net is likelier than one who appears
  twice a year.
- **Recency.** Crews change. Somebody who worked the last three events matters
  more than somebody who worked six events two seasons ago, so older sessions
  count for less on a fixed decay.

Only stations already on the roster are scored. A callsign that appears in the
transcripts but not the roster is *reported* rather than adopted: promoting a
mis-transcription to a station would bias decoding toward its own mistake, and
the whole design refuses to let the machine's errors become facts on their own.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

DECAY = 0.85
"""How much each session further back counts for. At 0.85 a station from ten
nets ago carries about a fifth the weight of one from last night."""


@dataclass
class Record:
    """What the transcripts say about one station."""

    callsign: str
    sessions: int = 0
    transmissions: int = 0
    score: float = 0.0
    last_seen: str = ""
    sources: set = field(default_factory=set)

    @property
    def known(self) -> bool:
        """Whether this callsign is on the roster. Set by `from_sessions`."""
        return self._known

    _known: bool = True


@dataclass
class Attendance:
    """Attendance across every session found, most expected station first."""

    records: dict[str, Record] = field(default_factory=dict)
    sessions: int = 0
    unknown: list[str] = field(default_factory=list)
    """Callsigns seen in the logs that are not on the roster -- for a human to
    look at, never adopted automatically."""

    def scores(self) -> dict[str, float]:
        return {c: r.score for c, r in self.records.items()}

    def expected(self, limit: int | None = None) -> list[str]:
        ordered = sorted(
            self.records.values(), key=lambda r: (-r.score, r.callsign)
        )
        names = [r.callsign for r in ordered]
        return names[:limit] if limit else names

    def for_source(self, source: str) -> dict[str, float]:
        """Scores restricted to stations heard on this receiver before."""
        if not source:
            return self.scores()
        return {
            c: r.score
            for c, r in self.records.items()
            if not r.sources or source in r.sources
        }


def from_sessions(sessions: list[list[dict]], roster: set[str]) -> Attendance:
    """Score stations from a list of sessions, oldest first."""
    attendance = Attendance(sessions=len(sessions))
    unknown: set[str] = set()

    for index, entries in enumerate(sessions):
        # Newest session has age 0 and full weight.
        weight = DECAY ** (len(sessions) - 1 - index)
        seen_this_session: set[str] = set()

        for entry in entries:
            callsign = entry.get("matched_callsign")
            if not entry.get("matched") or not callsign:
                continue
            if callsign not in roster:
                unknown.add(callsign)
                continue

            record = attendance.records.get(callsign)
            if record is None:
                record = Record(callsign=callsign)
                attendance.records[callsign] = record
            record.transmissions += 1
            if entry.get("source"):
                record.sources.add(entry["source"])
            if entry.get("timestamp"):
                record.last_seen = max(record.last_seen, entry["timestamp"])
            if callsign not in seen_this_session:
                seen_this_session.add(callsign)
                record.sessions += 1
                record.score += weight

    attendance.unknown = sorted(unknown)
    return attendance


def load(directory: str | Path, roster: set[str]) -> Attendance:
    """Read every session file in a directory and score attendance.

    Corrections replace the line they correct, so a station counts as attending
    when the operator says they did -- not when the machine first guessed.
    """
    directory = Path(directory)
    if not directory.exists():
        return Attendance()

    sessions: list[list[dict]] = []
    for path in sorted(directory.glob("net-*.jsonl"), key=lambda p: p.name):
        by_id: dict[int, dict] = {}
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            log.warning("Could not read %s: %s", path, exc)
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a truncated final line from a power cut
            if record.get("type") in ("entry", "correction", "traffic"):
                by_id[int(record.get("id", 0))] = record
        if by_id:
            sessions.append(list(by_id.values()))

    return from_sessions(sessions, roster)


def summary(attendance: Attendance, limit: int = 5) -> str:
    """One line for the startup log."""
    if not attendance.records:
        return "no attendance history yet"
    top = attendance.expected(limit)
    return (
        f"{len(attendance.records)} station(s) over {attendance.sessions} session(s); "
        f"most expected: {', '.join(top)}"
    )
