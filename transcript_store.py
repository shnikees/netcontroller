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

"""In-memory session log, with CSV/text export at the end of the net."""

from __future__ import annotations

import csv
import json
from bisect import bisect_right
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class TranscriptEntry:
    id: int
    timestamp: str
    """ISO 8601 local time the transmission started."""
    matched: bool
    matched_callsign: str | None
    operator_name: str
    raw_text: str
    confidence: float
    """STT confidence, 0-1."""
    match_score: float
    """Roster match score, 0-100. Zero when nothing callsign-shaped was heard."""
    clip_duration: float
    """Seconds."""
    candidate: str | None = None
    """The callsign-shaped token heard, even when it matched nothing."""
    unmatched_reason: str = ""
    corrected: bool = False
    """True once an operator has confirmed or fixed the callsign by hand."""
    original_callsign: str | None = None
    """What the matcher concluded before the operator corrected it."""
    via_alias: bool = False
    """True when a previously learned correction produced this match."""
    late: bool = False
    """Transcribed from the disk backlog, after the transmission had passed."""
    source: str = ""
    """Which receiver heard it, when more than one is configured."""
    escalated: bool = False
    """Re-transcribed by a larger model after the first pass was unsure."""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TranscriptStore:
    entries: list[TranscriptEntry] = field(default_factory=list)
    _next_id: int = 1

    def add(
        self,
        *,
        started_at: datetime,
        matched: bool,
        matched_callsign: str | None,
        operator_name: str,
        raw_text: str,
        confidence: float,
        match_score: float,
        clip_duration: float,
        candidate: str | None = None,
        unmatched_reason: str = "",
        via_alias: bool = False,
        late: bool = False,
        source: str = "",
    ) -> TranscriptEntry:
        entry = TranscriptEntry(
            id=self._next_id,
            timestamp=started_at.isoformat(timespec="seconds"),
            matched=matched,
            matched_callsign=matched_callsign,
            operator_name=operator_name,
            raw_text=raw_text,
            confidence=round(confidence, 3),
            match_score=round(match_score, 1),
            clip_duration=round(clip_duration, 2),
            candidate=candidate,
            unmatched_reason=unmatched_reason,
            via_alias=via_alias,
            late=late,
            source=source,
        )
        self._next_id += 1
        # Keep the log in transmission order. A clip recovered from the disk
        # backlog arrives after later ones, but belongs where it was spoken --
        # otherwise the exported net log reads out of sequence.
        position = bisect_right([e.timestamp for e in self.entries], entry.timestamp)
        self.entries.insert(position, entry)
        return entry

    def get(self, entry_id: int) -> TranscriptEntry | None:
        return next((e for e in self.entries if e.id == entry_id), None)

    def correct(
        self, entry_id: int, callsign: str, operator_name: str = ""
    ) -> TranscriptEntry | None:
        """Apply an operator correction to one entry.

        `original_callsign` keeps whatever the matcher concluded, so the log
        still shows where the machine was wrong -- that record is the point.
        """
        entry = self.get(entry_id)
        if entry is None:
            return None
        if not entry.corrected:
            entry.original_callsign = entry.matched_callsign
        entry.matched = True
        entry.matched_callsign = callsign
        entry.operator_name = operator_name
        entry.corrected = True
        entry.unmatched_reason = ""
        return entry

    def improve(
        self,
        entry_id: int,
        *,
        raw_text: str,
        matched: bool,
        matched_callsign: str | None,
        operator_name: str,
        confidence: float,
        match_score: float,
        candidate: str | None,
        unmatched_reason: str,
    ) -> TranscriptEntry | None:
        """Replace a line with the result of a second, better transcription.

        An operator correction always wins: if a human has already fixed this
        line, a machine re-run must not undo their work.
        """
        entry = self.get(entry_id)
        if entry is None or entry.corrected:
            return None
        entry.raw_text = raw_text
        entry.matched = matched
        entry.matched_callsign = matched_callsign
        entry.operator_name = operator_name
        entry.confidence = round(confidence, 3)
        entry.match_score = round(match_score, 1)
        entry.candidate = candidate
        entry.unmatched_reason = unmatched_reason
        entry.escalated = True
        return entry

    def all(self) -> list[dict]:
        return [e.to_dict() for e in self.entries]

    def check_ins(self) -> list[str]:
        """Distinct matched callsigns, in the order they first checked in."""
        seen: list[str] = []
        for entry in self.entries:
            if entry.matched and entry.matched_callsign not in seen:
                seen.append(entry.matched_callsign)  # type: ignore[arg-type]
        return seen

    # -- export ------------------------------------------------------------

    def export_csv(self, path: str | Path) -> Path:
        path = Path(path)
        fields = list(TranscriptEntry.__dataclass_fields__)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for entry in self.entries:
                writer.writerow(entry.to_dict())
        return path

    def export_text(self, path: str | Path) -> Path:
        path = Path(path)
        lines = [
            f"Net session log -- {len(self.entries)} transmissions, "
            f"{len(self.check_ins())} stations",
            "",
        ]
        multi = len({e.source for e in self.entries if e.source}) > 1
        for entry in self.entries:
            who = entry.matched_callsign if entry.matched else "UNMATCHED"
            if entry.matched and entry.operator_name:
                who = f"{who} ({entry.operator_name})"
            if entry.corrected:
                was = entry.original_callsign or "unmatched"
                who = f"{who} [corrected from {was}]"
            where = f" ({entry.source})" if multi and entry.source else ""
            lines.append(f"[{entry.timestamp}]{where} {who}: {entry.raw_text}")
        lines += ["", "Check-ins: " + ", ".join(self.check_ins())]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def export_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.all(), indent=2), encoding="utf-8")
        return path
