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

"""Rebuild voice profiles from the kept enrolment audio.

    python tools/rebuild_voices.py

Run this after changing the embedder. Vectors from two different models mean
nothing to each other, so the stored profiles are void the moment the model
changes — this re-embeds the clips that produced them and writes fresh ones,
which is the whole reason the audio is kept.

Also worth running after correcting a station's profile by hand: delete the bad
clips from `voice_audio/<CALLSIGN>/` and rebuild, rather than living with a
centroid that a wrong match pulled off-centre.

With `--compare`, it reports how well the current embedder separates your
operators, using the same measurement `calibrate.py` uses — so two embedders
can be scored on identical audio instead of on two different months of traffic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibrate import calibrate_voice  # noqa: E402
from config import load_config  # noqa: E402
from voice_id import EnrolmentAudio, VoiceProfiles  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--audio-dir", help="default: from config")
    parser.add_argument("--voices", help="default: from config")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="report how well this embedder separates your operators",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report, but do not write profiles"
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = load_config(config_path if config_path.exists() else None)
    store = EnrolmentAudio(
        args.audio_dir or config.voice.audio_dir,
        per_station=config.voice.audio_per_station,
        max_seconds=config.voice.audio_max_seconds,
    )
    voices_path = Path(args.voices or config.voice.path)

    stations = store.stations()
    if not stations:
        print(
            f"No enrolment audio in {store.directory}/.\n"
            "Nothing to rebuild from -- run a net with voice.enabled and\n"
            "voice.keep_audio turned on."
        )
        return 0

    print(f"Rebuilding from {store.directory}/ -- {len(stations)} station(s)")
    profiles = VoiceProfiles(
        path=voices_path,
        min_similarity=config.voice.min_similarity,
        margin=config.voice.margin,
        min_enrolments=config.voice.min_enrolments,
        audio=store,
    )
    rebuilt, clips = profiles.rebuild()
    print(f"  re-embedded {clips} clip(s) into {rebuilt} profile(s)")
    for callsign in sorted(profiles.profiles):
        profile = profiles.profiles[callsign]
        print(f"    {callsign:<10} {profile.count} clip(s)")

    if args.compare:
        result = calibrate_voice(profiles)
        print("\nSeparation on this audio, with the current embedder:")
        if result.same and result.different:
            print(f"  same station:      mean {sum(result.same)/len(result.same):.3f}")
            print(
                f"  different station: mean "
                f"{sum(result.different)/len(result.different):.3f}"
            )
        if result.usable:
            print(f"  suggested min_similarity: {result.threshold}")
            print(
                f"  {result.recall:.0%} of a station's own clips recognised, "
                f"{result.false_accepts:.1%} wrongly accepted"
            )
        else:
            print(f"  {result.note}")
        print(
            "\nRun this again after changing the embedder: the same clips "
            "scored\ntwo ways is the only fair comparison."
        )

    if args.dry_run:
        print(f"\nDry run -- {voices_path} not written.")
        return 0
    if profiles.save():
        print(f"\nWrote {voices_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
