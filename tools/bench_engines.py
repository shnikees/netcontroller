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

"""Time one STT engine against another on real net audio.

Buying decisions here should come from a measurement rather than a spec sheet,
and the measurement has to be of *this* workload: short VAD clips with a roster
prompt, not one long file. Feeding a whole recording to an engine flatters it,
because a 30-second window amortises everything a 4-second transmission cannot.

    python tools/bench_engines.py --audio net.wav
    python tools/bench_engines.py --audio net.wav --repeat 3 \\
        --whisper-cpp ~/whisper.cpp/build/bin/whisper-cli \\
        --ggml ~/whisper.cpp/models/ggml-base.bin

With `--expected` it also scores *callsign recovery* -- the number that matters
here, since a transcript that loses the callsign has lost where on the course
the transmission came from. Give it one callsign per line, in transmission
order, and the roster it should be matched against.

Two traps this avoids, both of which quietly produce a wrong answer:

- whisper.cpp's `-mc 0` looks like faster-whisper's
  `condition_on_previous_text=False` and is not: it discards the initial prompt
  as well, which costs several callsigns a net and makes the engine look worse
  than it is.
- Timing a single run. Run-to-run spread on a laptop is around ten percent, so
  `--repeat 3` and a median is the difference between a measurement and a
  coincidence.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from callsign_match import CallsignMatcher, RosterEntry  # noqa: E402
from stt_worker import LEAD_IN  # noqa: E402
from vad_segmenter import VadSegmenter  # noqa: E402

RATE = 16_000


# --------------------------------------------------------------------------
# Preparing the workload
# --------------------------------------------------------------------------


def segment(audio_path: Path, out_dir: Path) -> list[Path]:
    """Cut a recording into clips exactly as the live pipeline would."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("clip-*.wav"):
        stale.unlink()

    with wave.open(str(audio_path)) as source:
        if source.getframerate() != RATE or source.getnchannels() != 1:
            raise SystemExit(f"{audio_path} must be 16 kHz mono")
        pcm = source.readframes(source.getnframes())

    segmenter = VadSegmenter()
    frame_bytes = segmenter.frame_ms * 32  # 16 kHz, 16-bit
    frames = [
        pcm[i : i + frame_bytes]
        for i in range(0, len(pcm) - frame_bytes + 1, frame_bytes)
    ]

    paths = []
    for index, clip in enumerate(segmenter.segment(frames), 1):
        samples = np.clip(clip.audio * 32767.0, -32768, 32767).astype("<i2")
        path = out_dir / f"clip-{index:03d}.wav"
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(RATE)
            out.writeframes(samples.tobytes())
        paths.append(path)
    return paths


def speech_seconds(clips: list[Path]) -> float:
    total = 0.0
    for clip in clips:
        with wave.open(str(clip)) as handle:
            total += handle.getnframes() / RATE
    return total


# --------------------------------------------------------------------------
# The engines
# --------------------------------------------------------------------------


def run_faster_whisper(clips, prompt, model_size, compute_type, threads):
    from faster_whisper import WhisperModel

    started = time.perf_counter()
    model = WhisperModel(
        model_size, device="cpu", compute_type=compute_type, cpu_threads=threads
    )
    load = time.perf_counter() - started

    texts, started = [], time.perf_counter()
    for clip in clips:
        segments, _ = model.transcribe(
            str(clip),
            language="en",
            beam_size=5,
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt=prompt,
        )
        texts.append(" ".join(segment.text for segment in segments))
    return {"load": load, "compute": time.perf_counter() - started, "texts": texts}


def run_whisper_cpp(clips, prompt, binary, ggml, gpu, threads):
    """One invocation for every clip, so the model is loaded once.

    Deliberately *not* passing `-mc 0`: it reads like the analogue of
    `condition_on_previous_text=False` but also throws the prompt away.
    """
    command = [
        str(binary), "-m", str(ggml), "-l", "en",
        "-bs", "5", "-nt", "-t", str(threads), "-oj",
    ]
    if not gpu:
        command.append("-ng")
    if prompt:
        command += ["--prompt", prompt]
    for clip in clips:
        command += ["-f", str(clip)]
        Path(str(clip) + ".json").unlink(missing_ok=True)

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"whisper-cli failed:\n{result.stderr[-2000:]}")

    def timing(label: str) -> float:
        found = re.search(rf"{label} time =\s+([\d.]+) ms", result.stderr)
        return float(found.group(1)) / 1000 if found else 0.0

    load = timing("load")
    texts = []
    for clip in clips:
        data = json.loads(Path(str(clip) + ".json").read_text())
        texts.append(" ".join(part["text"] for part in data["transcription"]))
    return {"load": load, "compute": timing("total") - load, "texts": texts}


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def score(texts, expected, matcher):
    """Right, wrong and unmatched -- kept apart on purpose.

    A wrong callsign is far worse than an unmatched line: it puts a station in
    the wrong place on the course. An engine that trades unmatched lines for
    wrong ones has made this worse, however good its word error rate looks.
    """
    right = wrong = unmatched = 0
    misses = []
    for text, want in zip(texts, expected):
        result = matcher.match(text)
        if not result.matched:
            unmatched += 1
            misses.append(f"    unmatched  {text.strip()[:64]}")
        elif result.callsign == want:
            right += 1
        else:
            wrong += 1
            misses.append(f"    WRONG {result.callsign} for {want}: {text.strip()[:52]}")
    return right, wrong, unmatched, misses


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--audio", required=True, help="16 kHz mono WAV of a net")
    parser.add_argument("--model", default="base")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=1, help="median of N runs")
    parser.add_argument("--roster", help="CSV or one callsign per line, for the prompt")
    parser.add_argument("--expected", help="one callsign per line, transmission order")
    parser.add_argument("--no-prompt", action="store_true")
    parser.add_argument("--whisper-cpp", help="path to whisper-cli")
    parser.add_argument("--ggml", help="path to a ggml model for whisper.cpp")
    parser.add_argument("--work-dir", default=".bench-clips")
    args = parser.parse_args()

    clips = segment(Path(args.audio), Path(args.work_dir))
    if not clips:
        raise SystemExit("The VAD found no transmissions in that recording.")
    audio_seconds = speech_seconds(clips)

    callsigns = []
    if args.roster:
        for line in Path(args.roster).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                callsigns.append(line.split(",")[0].strip().upper())
    prompt = None
    if callsigns and not args.no_prompt:
        prompt = f"{LEAD_IN} " + ", ".join(callsigns) + "."

    expected = []
    if args.expected:
        expected = [
            line.strip().upper()
            for line in Path(args.expected).read_text().splitlines()
            if line.strip()
        ]
    matcher = CallsignMatcher(roster=[RosterEntry(c) for c in callsigns])

    engines = [
        (
            f"faster-whisper {args.model} {args.compute_type}  CPU",
            lambda: run_faster_whisper(
                clips, prompt, args.model, args.compute_type, args.threads
            ),
        )
    ]
    if args.whisper_cpp and args.ggml:
        for gpu in (False, True):
            label = "GPU" if gpu else "CPU"
            engines.append(
                (
                    f"whisper.cpp {Path(args.ggml).stem:<18} {label}",
                    lambda gpu=gpu: run_whisper_cpp(
                        clips, prompt, args.whisper_cpp, args.ggml, gpu, args.threads
                    ),
                )
            )

    print(
        f"\n{len(clips)} transmissions, {audio_seconds:.1f}s of speech"
        f"{', roster prompt of ' + str(len(callsigns)) if prompt else ', no prompt'}"
        f"{' stations' if prompt else ''}\n"
    )
    header = f"{'engine':38} {'load':>6} {'compute':>8} {'realtime':>9}"
    if expected:
        header += f" {'ok':>6} {'wrong':>6}"
    print(header)
    print("-" * len(header))

    all_misses = {}
    for name, run in engines:
        times, loads, texts = [], [], None
        for _ in range(args.repeat):
            result = run()
            times.append(result["compute"])
            loads.append(result["load"])
            texts = result["texts"]
        compute = statistics.median(times)
        line = (
            f"{name:38} {statistics.median(loads):5.2f}s {compute:7.2f}s "
            f"{compute / audio_seconds:8.3f}x"
        )
        if expected:
            right, wrong, _, misses = score(texts, expected, matcher)
            line += f" {right:3d}/{len(expected):<2d} {wrong:6d}"
            all_misses[name] = misses
        print(line)

    for name, misses in all_misses.items():
        if misses:
            print(f"\n  {name}")
            print("\n".join(misses))

    print(
        "\nRealtime under ~0.5x leaves room for a bigger live model; near 1.00x "
        "there is none.\nA wrong callsign costs more than an unmatched line -- "
        "weigh that column first.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
