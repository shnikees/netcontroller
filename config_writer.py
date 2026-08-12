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

"""Write settings back into config.yaml without destroying it.

A YAML round-trip would be simpler and would throw away every comment in the
file. Those comments are most of what makes the config readable -- they explain
what each threshold is guessing about and what to do when it is wrong -- so
values are patched in place instead, leaving the surrounding text alone.

The trade is that this only edits keys that already exist. A key that has been
deleted from the file is reported back rather than silently appended, because
adding it would put it somewhere arbitrary with no explanation next to it.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path


def to_yaml(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        # Quote anything that could be read back as another type.
        if value == "" or re.search(r"[:#\s]", value) or value.lower() in {
            "true", "false", "null", "yes", "no", "on", "off",
        }:
            return f'"{value}"'
        return value
    return str(value)


def patch(path: str | Path, values: dict[str, object], *, backup: bool = True) -> dict:
    """Set dotted keys in a config file. Returns what happened.

    `values` is {"vad.silence_ms": 900, "traffic.detect": False}. Nested source
    entries ("sources.Repeater.gain") are matched inside the named list item.
    """
    path = Path(path)
    if not path.exists():
        return {"written": [], "missing": list(values), "error": f"{path} not found"}

    text = path.read_text(encoding="utf-8")
    original = text
    written: list[str] = []
    missing: list[str] = []

    for dotted, value in values.items():
        parts = dotted.split(".")
        if parts[0] == "sources" and len(parts) == 3:
            text, changed = _patch_source(text, parts[1], parts[2], value)
        else:
            text, changed = _patch_section(text, parts[0], ".".join(parts[1:]), value)
        (written if changed else missing).append(dotted)

    if not written:
        return {"written": [], "missing": missing}
    if backup and text != original:
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    path.write_text(text, encoding="utf-8")
    return {"written": written, "missing": missing}


def _patch_section(text: str, section: str, key: str, value) -> tuple[str, bool]:
    # Anchored to the section so `enabled:` under one block cannot be mistaken
    # for `enabled:` under another.
    pattern = re.compile(
        rf"(^{re.escape(section)}:.*?^\s+{re.escape(key)}:[ \t]*)([^\n#]*)",
        re.MULTILINE | re.DOTALL,
    )
    updated, count = pattern.subn(
        lambda m: f"{m.group(1)}{to_yaml(value)}"
        + (" " if m.group(2).rstrip() != m.group(2) else ""),
        text,
        count=1,
    )
    return updated, bool(count)


def _patch_source(text: str, name: str, key: str, value) -> tuple[str, bool]:
    """Set a key on the `sources:` entry with this name."""
    # Find the list item whose name matches, then the key within it -- stopping
    # at the next item so a value is never written into the wrong receiver.
    item = re.compile(
        rf"(^\s*-\s+name:[ \t]*[\"']?{re.escape(name)}[\"']?[ \t]*$)(.*?)(?=^\s*-\s+name:|^\S|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = item.search(text)
    if not match:
        return text, False

    body = match.group(2)
    key_pattern = re.compile(rf"(^\s+{re.escape(key)}:[ \t]*)([^\n#]*)", re.MULTILINE)
    new_body, count = key_pattern.subn(
        lambda m: f"{m.group(1)}{to_yaml(value)}", body, count=1
    )
    if not count:
        return text, False
    return text[: match.start(2)] + new_body + text[match.end(2) :], True
