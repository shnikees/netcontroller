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

"""Append-only log of operator corrections, and the aliases learned from it.

One file does both jobs. The log is the source of truth; aliases are derived by
replaying it at startup, so there is no separate state file that can drift out
of sync with the record of what the operator actually said.

It is also the labelled dataset. Every line pairs what Whisper produced with the
callsign a human confirmed, which is exactly the input a future fine-tuning run
would want -- see docs/ARCHITECTURE.md.

Format is JSON Lines: one self-describing record per line, appended, never
rewritten. Safe to `tail -f` during a net, and a truncated final line (power cut
mid-write) costs one correction rather than the whole file.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Correction:
    """One operator correction of one transcript line."""

    timestamp: str
    entry_id: int
    candidate: str | None
    """The callsign-shaped token heard, which becomes the alias key."""
    from_callsign: str | None
    """What the matcher concluded: a callsign, or null when unmatched."""
    to_callsign: str
    """What the operator says it actually was."""
    raw_text: str
    """Whisper's transcript, kept for future fine-tuning."""
    confidence: float = 0.0
    clip_duration: float = 0.0


class FeedbackLog:
    """Append-only JSONL log of corrections."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, correction: Correction) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(correction)) + "\n")

    def all(self) -> list[Correction]:
        if not self.path.exists():
            return []
        corrections: list[Correction] = []
        for number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                corrections.append(
                    Correction(
                        **{
                            k: v
                            for k, v in data.items()
                            if k in Correction.__dataclass_fields__
                        }
                    )
                )
            except (json.JSONDecodeError, TypeError) as exc:
                # One malformed line -- most likely a write interrupted by a
                # power cut -- must not cost the operator every other alias.
                log.warning("Skipping bad line %d in %s: %s", number, self.path, exc)
        return corrections

    def aliases(self) -> dict[str, str]:
        """Replay the log into candidate -> callsign, last correction winning."""
        aliases: dict[str, str] = {}
        for correction in self.all():
            if correction.candidate:
                aliases[correction.candidate.upper()] = correction.to_callsign
        return aliases


def record_correction(
    feedback: FeedbackLog,
    *,
    entry_id: int,
    candidate: str | None,
    from_callsign: str | None,
    to_callsign: str,
    raw_text: str,
    confidence: float = 0.0,
    clip_duration: float = 0.0,
    now: datetime | None = None,
) -> Correction:
    correction = Correction(
        timestamp=(now or datetime.now()).isoformat(timespec="seconds"),
        entry_id=entry_id,
        candidate=candidate,
        from_callsign=from_callsign,
        to_callsign=to_callsign,
        raw_text=raw_text,
        confidence=confidence,
        clip_duration=clip_duration,
    )
    feedback.append(correction)
    return correction
