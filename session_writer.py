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

"""Write the transcript to disk as the net runs, not just at the end.

Until now the session lived in memory and reached disk on a clean exit or when
somebody pressed Export. That is fine right up until the Pi loses power or
somebody closes the laptop, and then two hours of net vanish with nothing to
show for it -- the one outcome this whole app exists to prevent.

So every line is written as it is produced. Two files per session, because they
answer different questions:

    net-<stamp>.jsonl   append-only, one JSON object per line, fsynced.
                        The durable record. Survives a power cut mid-write with
                        the loss of at most the last line, and keeps corrections
                        as their own entries so the history is auditable.

    net-<stamp>.txt     the human-readable log, rewritten in transmission order
                        after each change. This is the file that gets pasted
                        into a net report, and it is always current -- no
                        "remember to press Export" step.

Rewriting the text file per entry is cheap at net scale (a busy net is a few
hundred lines) and means the readable copy is never behind the durable one.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


class SessionWriter:
    """Streams a session to disk. Failures degrade, they do not stop the net.

    fsync: force each line to the platter before returning. The point of this
        module is surviving power loss, and a buffered write that never reached
        disk would defeat it. Costs a few ms per transmission, which is nothing
        against a Whisper inference.
    """

    def __init__(
        self,
        directory: str | Path,
        store,
        *,
        fsync: bool = True,
        started_at: datetime | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.store = store
        self.fsync = fsync
        self.started_at = started_at or datetime.now()
        self.jsonl_path: Path | None = None
        self.text_path: Path | None = None
        self.entries_written = 0
        self._failed = False

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> Path | None:
        """Create the session files. Returns the JSONL path, or None if disabled
        by a filesystem that will not take it."""
        stamp = self.started_at.strftime("%Y%m%d-%H%M%S")
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.jsonl_path = self.directory / f"net-{stamp}.jsonl"
            self.text_path = self.directory / f"net-{stamp}.txt"
            self._write_line({"type": "session", "started_at": self.started_at.isoformat()})
            return self.jsonl_path
        except OSError as exc:
            self._disable("could not create session files", exc)
            return None

    def close(self) -> None:
        """Final flush of the readable copy, so it matches the durable one."""
        if self._failed:
            return
        self._rewrite_text()

    # -- writing -----------------------------------------------------------

    def append(self, entry) -> None:
        """Record a new transcript line."""
        self._write_line({"type": "entry", **entry.to_dict()})
        self.entries_written += 1
        self._rewrite_text()

    def record_correction(self, entry) -> None:
        """Record an operator correction as its own line.

        Appended rather than rewritten: the JSONL is a history, and the fact
        that the machine got it wrong before a human fixed it is part of the
        record worth keeping.
        """
        self._write_line({"type": "correction", **entry.to_dict()})
        self._rewrite_text()

    # -- internals ---------------------------------------------------------

    def _write_line(self, payload: dict) -> None:
        if self._failed or self.jsonl_path is None:
            return
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
                handle.flush()
                if self.fsync:
                    os.fsync(handle.fileno())
        except OSError as exc:
            self._disable("could not write session line", exc)

    def _rewrite_text(self) -> None:
        if self._failed or self.text_path is None:
            return
        try:
            self.store.export_text(self.text_path)
        except OSError as exc:
            self._disable("could not write readable session log", exc)

    def _disable(self, what: str, exc: Exception) -> None:
        """Stop trying, once and loudly.

        A full disk should produce one error in the log, not one per
        transmission for the rest of the net.
        """
        if not self._failed:
            log.error("Session writing disabled -- %s: %s", what, exc)
        self._failed = True


def read_session(path: str | Path) -> list[dict]:
    """Read a session JSONL back, skipping anything malformed.

    For recovering a net after a crash: the entries are all there, including
    the ones written after the last readable-log rewrite.
    """
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            # A truncated final line is the expected shape of a power cut.
            log.warning("Skipping bad line %d in %s: %s", number, path, exc)
    return records
