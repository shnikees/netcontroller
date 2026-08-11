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

"""Generate a synthetic net recording for testing without an SDR.

Speaks a list of check-ins with the system TTS, splices them together with dead
air between transmissions, and adds hiss so the VAD sees something closer to a
receiver feed than a studio mic. The result feeds straight into:

    python app.py --file test-net.wav --model tiny

Requires a system TTS: `say` on macOS, `espeak-ng` on Linux
(`apt install espeak-ng`). Neither is a dependency of the app itself.

    python tools/make_test_audio.py                       # default script
    python tools/make_test_audio.py --script my-net.txt   # one line per transmission
    python tools/make_test_audio.py --gap 3 --noise 200   # longer pauses, more hiss

This is a stand-in for real net audio, not a replacement. TTS enunciates far
better than a handheld into a repeater, so it will make the pipeline look more
accurate than it will be on the air. Use it to exercise the plumbing and to
build up matcher regression cases -- then retune against a real recording.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

RATE = 16_000

# Deliberately a mixed bag: clean check-ins, one station that gives its callsign
# only at the end, and one off-roster visitor that should come out unmatched.
DEFAULT_SCRIPT = [
    "This is whiskey six alpha bravo charlie, checking in, no traffic.",
    "Net control, kilo seven x-ray yankee zulu, checking in with traffic.",
    "November five delta echo foxtrot, good evening, nothing for the net.",
    "Kilo delta niner mike november oscar, Dave here, QNI please.",
    "Good evening everyone, nothing to pass tonight, alpha alpha four papa quebec.",
    "Victor echo three zulu quebec romeo, visiting station, listening.",
]


def synthesize(text: str, out: Path) -> None:
    """Speak `text` into a 16 kHz mono 16-bit WAV at `out`."""
    if shutil.which("say"):
        aiff = out.with_suffix(".aiff")
        subprocess.run(["say", "-o", str(aiff), text], check=True)
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", f"LEI16@{RATE}", "-c", "1",
             str(aiff), str(out)],
            check=True,
        )
        aiff.unlink()
    elif shutil.which("espeak-ng"):
        subprocess.run(
            ["espeak-ng", "-w", str(out), "-s", "150", text], check=True
        )
        _resample_in_place(out)
    else:
        raise SystemExit(
            "No system TTS found. Install espeak-ng (Linux) or run this on "
            "macOS, which has `say` built in."
        )


def _resample_in_place(path: Path) -> None:
    """espeak-ng writes 22.05 kHz; drop it to 16 kHz by linear interpolation."""
    with wave.open(str(path)) as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels)[:, 0]
    if rate != RATE:
        target_len = int(len(samples) * RATE / rate)
        samples = np.interp(
            np.linspace(0, len(samples) - 1, target_len),
            np.arange(len(samples)),
            samples,
        ).astype(np.int16)
    _write_wav(path, samples)


def _write_wav(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(samples.tobytes())


def build(script: list[str], gap_s: float, noise: float, seed: int) -> np.ndarray:
    gap = np.zeros(int(RATE * gap_s), dtype=np.int16)
    parts: list[np.ndarray] = [gap]
    with tempfile.TemporaryDirectory() as tmp:
        for index, line in enumerate(script):
            path = Path(tmp) / f"{index}.wav"
            synthesize(line, path)
            with wave.open(str(path)) as wav:
                parts.append(
                    np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
                )
            parts.append(gap)
            print(f"  {index + 1}. {line}")
    audio = np.concatenate(parts).astype(np.int32)
    if noise > 0:
        rng = np.random.default_rng(seed)
        audio += rng.normal(0, noise, len(audio)).astype(np.int32)
    return np.clip(audio, -32768, 32767).astype(np.int16)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-o", "--output", default="test-net.wav")
    parser.add_argument(
        "--script", help="text file with one transmission per line"
    )
    parser.add_argument(
        "--gap", type=float, default=1.5,
        help="seconds of dead air between transmissions (default 1.5)",
    )
    parser.add_argument(
        "--noise", type=float, default=60,
        help="hiss amplitude in int16 units; 0 for clean audio (default 60)",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    script = (
        [
            line.strip()
            for line in Path(args.script).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        if args.script
        else DEFAULT_SCRIPT
    )

    print(f"Synthesizing {len(script)} transmissions:")
    audio = build(script, args.gap, args.noise, args.seed)
    _write_wav(Path(args.output), audio)
    print(
        f"\nWrote {args.output} -- {len(audio) / RATE:.1f}s, "
        f"{len(script)} transmissions\n\n"
        f"    python app.py --file {args.output} --model tiny\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
