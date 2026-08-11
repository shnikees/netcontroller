"""In-memory session log, with CSV/text export at the end of the net."""

from __future__ import annotations

import csv
import json
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
        )
        self._next_id += 1
        self.entries.append(entry)
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
        for entry in self.entries:
            who = entry.matched_callsign if entry.matched else "UNMATCHED"
            if entry.matched and entry.operator_name:
                who = f"{who} ({entry.operator_name})"
            lines.append(f"[{entry.timestamp}] {who}: {entry.raw_text}")
        lines += ["", "Check-ins: " + ", ".join(self.check_ins())]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def export_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.all(), indent=2), encoding="utf-8")
        return path
