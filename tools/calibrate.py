#!/usr/bin/env python3

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

"""Set thresholds from the nets you have already run.

    python tools/calibrate.py            # look at what is there, print advice
    python tools/calibrate.py --apply    # and write it into config.yaml

No arguments needed and nothing to record: this reads the transcripts and voice
profiles the app writes anyway. Run it after each net for the first few weeks —
it gets more confident every time, and it says so when it does not have enough
to go on yet.

It never edits config.yaml without `--apply`, and `--apply` keeps a backup.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibrate import (  # noqa: E402
    calibrate_escalation,
    calibrate_voice,
    load_entries,
)
from config import load_config  # noqa: E402
from voice_id import VoiceProfiles  # noqa: E402


def bar(values: list[float], low: float = 0.0, high: float = 1.0, width: int = 34) -> str:
    """A one-line histogram, so the numbers come with their evidence."""
    if not values:
        return " " * width
    counts, _ = np.histogram(values, bins=width, range=(low, high))
    blocks = " ▁▂▃▄▅▆▇█"
    peak = counts.max() or 1
    return "".join(blocks[min(8, int(round(c / peak * 8)))] for c in counts)


def report_escalation(entries: list[dict]) -> dict:
    print("\n" + "=" * 74)
    print("escalation.min_confidence -- when a line is worth a second pass")
    print("=" * 74)

    result = calibrate_escalation(entries)
    print(f"  matched lines:   {len(result.matched):>4}  {bar(result.matched)}")
    print(f"  unmatched lines: {len(result.unmatched):>4}  {bar(result.unmatched)}")
    print("                         0.0" + " " * 28 + "1.0")

    if not result.usable:
        print(f"\n  {result.note}")
        return {}

    print(
        f"\n  matched lines sit {result.separation:.2f} higher in confidence "
        "than unmatched ones,"
    )
    print("  so confidence is worth thresholding on.")
    print(f"\n  suggested: min_confidence: {result.threshold}")
    print(
        f"  at that setting {result.escalate_fraction:.0%} of lines get a "
        "second pass"
    )
    if result.escalate_fraction > 0.4:
        print(
            "  note: that is a lot of re-transcription. Lower the threshold if "
            "the\n        machine cannot keep up -- late lines are the cost."
        )
    return {"min_confidence": result.threshold}


def report_voice(profiles: VoiceProfiles) -> dict:
    print("\n" + "=" * 74)
    print("voice.min_similarity -- how alike two clips of one operator are")
    print("=" * 74)

    result = calibrate_voice(profiles)
    stations = [c for c, p in profiles.profiles.items() if len(p.samples) >= 2]
    print(f"  stations with 2+ clips: {len(stations)}")
    print(f"  same station:      {len(result.same):>5} pairs  {bar(result.same, 0.5, 1.0)}")
    print(f"  different station: {len(result.different):>5} pairs  {bar(result.different, 0.5, 1.0)}")
    print("                                     0.5" + " " * 26 + "1.0")

    if not result.usable and not result.same:
        print(f"\n  {result.note}")
        return {}
    if not result.usable and len(result.same) < 5:
        # Never suggest a number the calibration just said it cannot support:
        # printing the default here would look like a measurement.
        print(f"\n  {result.note}")
        return {}

    print(f"\n  suggested: min_similarity: {result.threshold}")
    print(f"             margin: {result.margin}")
    print(
        f"  at that setting {result.recall:.0%} of a station's own clips would "
        f"be recognised,\n  with {result.false_accepts:.1%} of other stations "
        "wrongly accepted"
    )
    return {"min_similarity": result.threshold, "margin": result.margin}


def apply_to_config(path: Path, escalation: dict, voice: dict) -> None:
    """Patch the two sections in place, keeping comments and layout intact.

    A regex rather than a YAML round-trip: PyYAML would rewrite the file and
    throw away every comment in it, and those comments are most of what makes
    config.yaml.example worth reading.
    """
    import re

    if not path.exists():
        print(f"\n  {path} does not exist -- copy config.yaml.example first")
        return

    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    text = path.read_text(encoding="utf-8")

    changes = [
        (section, key, value)
        for section, values in (("escalation", escalation), ("voice", voice))
        for key, value in values.items()
    ]
    applied = []
    for section, key, value in changes:
        # Only inside the right section: min_similarity appears once, but
        # something like `enabled` would not.
        pattern = re.compile(
            rf"(^{section}:.*?^\s+{key}:\s*)([^\s#]+)", re.MULTILINE | re.DOTALL
        )
        text, count = pattern.subn(rf"\g<1>{value}", text, count=1)
        if count:
            applied.append(f"{section}.{key} = {value}")

    if not applied:
        print(f"\n  nothing to apply (backup left at {backup.name})")
        return
    path.write_text(text, encoding="utf-8")
    print(f"\n  wrote {path} (backup: {backup.name})")
    for line in applied:
        print(f"    {line}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--transcripts", help="default: from config")
    parser.add_argument("--voices", help="default: from config")
    parser.add_argument(
        "--apply", action="store_true", help="write the suggestions into config.yaml"
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = load_config(config_path if config_path.exists() else None)
    transcripts = Path(args.transcripts or config.transcripts.dir)
    voices_path = Path(args.voices or config.voice.path)

    entries = load_entries(transcripts)
    profiles = VoiceProfiles(path=voices_path)
    profiles.load()

    sessions = len(list(transcripts.glob("*.jsonl"))) if transcripts.exists() else 0
    print(f"Reading {transcripts}/ -- {sessions} session(s), {len(entries)} lines")
    print(f"Reading {voices_path} -- {len(profiles.profiles)} voice profile(s)")

    if not entries and not profiles.profiles:
        print(
            "\nNothing to calibrate from yet. Run a net with `transcripts.live: true`"
            "\n(the default), and turn on `voice.enabled` if you want voice"
            "\nsuggestions calibrated too."
        )
        return 0

    escalation = report_escalation(entries)
    voice = report_voice(profiles)

    print("\n" + "=" * 74)
    if not escalation and not voice:
        print("No settings changed -- run another net and try again.")
        return 0

    print("Suggested config:\n")
    if escalation:
        print("escalation:")
        for key, value in escalation.items():
            print(f"  {key}: {value}")
    if voice:
        print("voice:")
        for key, value in voice.items():
            print(f"  {key}: {value}")

    if args.apply:
        apply_to_config(config_path, escalation, voice)
    else:
        print("\nRe-run with --apply to write these into config.yaml.")
    print("\nRun this again after the next net; it gets more confident each time.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
