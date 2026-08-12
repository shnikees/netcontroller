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

"""Tune the thresholds against a recording of your own net.

Every timing constant in this app ships as a reasoned guess: how long your
operators pause while spelling a callsign, how much dead air sits between two
stations, where "unsure" begins for your audio. None of that can be known in
advance, and all of it can be measured from ten minutes of recorded traffic.

    python tools/tune.py --audio net-recording.wav --roster roster.csv

**No hand-labelling.** The roster is the supervision: a good setting is one
where each clip comes out as exactly one confident roster match. A setting that
merges two stations produces clips with two callsigns in them; one that cuts
too early produces fragments with none. Both are visible without anyone
transcribing anything by hand.

The tool prints the evidence, not just an answer. A sweep that picks a winner
by a hair is telling you the setting does not matter much on your net, and that
is worth seeing rather than having decided for you.

Cost: one full transcription pass per VAD setting, so a ten-minute recording
with `base` is a minute or two per candidate. Threshold sweeps that do not
change segmentation are free -- they reuse the transcripts already computed.
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from callsign_match import CallsignMatcher, load_roster  # noqa: E402
from clip_split import split_transmissions  # noqa: E402
from resample import Resampler  # noqa: E402
from stt_worker import SttWorker  # noqa: E402
from vad_segmenter import VadSegmenter  # noqa: E402

TARGET_RATE = 16_000


# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------


def read_wav(path: str) -> np.ndarray:
    """Load any 16-bit PCM WAV as float32 mono at 16 kHz."""
    with wave.open(path, "rb") as handle:
        if handle.getsampwidth() != 2:
            raise SystemExit("Recording must be 16-bit PCM")
        rate = handle.getframerate()
        channels = handle.getnchannels()
        raw = handle.readframes(handle.getnframes())

    samples = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels)[:, 0]
    if rate != TARGET_RATE:
        resampler = Resampler(rate, TARGET_RATE)
        samples = np.concatenate([resampler.process(samples), resampler.flush()])
    return samples


def frames_of(samples: np.ndarray, frame_ms: int = 30):
    size = int(TARGET_RATE * frame_ms / 1000)
    for start in range(0, len(samples) - size + 1, size):
        yield samples[start : start + size].tobytes()


def recommend_gain(samples: np.ndarray) -> tuple[float, str]:
    """Level is arithmetic, not a search: measure it and solve for the target."""
    peak = float(np.max(np.abs(samples))) / 32768.0
    if peak < 1e-4:
        return 1.0, "no signal in this recording at all -- check the cable"
    gain = 0.6 / peak
    if 0.8 <= gain <= 1.25:
        return 1.0, f"peak {peak:.2f} of full scale; leave gain alone"
    if peak > 0.98:
        return round(gain, 2), f"peak {peak:.2f} -- clipping, bring it down"
    return round(gain, 2), f"peak {peak:.2f} of full scale"


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


@dataclass
class Score:
    """What one setting did to the recording, in terms the operator can judge."""

    clips: int = 0
    clean: int = 0
    """Clips yielding exactly one roster station -- the outcome we want."""
    merged: int = 0
    """Clips containing two or more stations: the VAD ran them together."""
    empty: int = 0
    """Clips with no callsign at all: a fragment, a tail, or just chatter."""
    stations: set = field(default_factory=set)
    seconds: float = 0.0

    @property
    def value(self) -> float:
        """One number to rank by, with the reasoning visible in the parts.

        Clean clips are the point. Merged clips cost more than empty ones: an
        empty clip is usually somebody saying "roger", while a merged one has
        lost a station from the log.
        """
        return self.clean + 0.5 * len(self.stations) - 1.5 * self.merged - 0.2 * self.empty

    def row(self, label: str) -> str:
        return (
            f"{label:<12}{self.clips:>7}{self.clean:>8}{self.merged:>8}"
            f"{self.empty:>8}{len(self.stations):>10}{self.value:>9.1f}{self.seconds:>9.1f}s"
        )


HEADER = (
    f"{'setting':<12}{'clips':>7}{'clean':>8}{'merged':>8}"
    f"{'empty':>8}{'stations':>10}{'score':>9}{'time':>10}"
)


def score_clips(clips, matcher: CallsignMatcher, stt: SttWorker, gap_ms: int) -> Score:
    """Transcribe each clip and judge the segmentation by what came out."""
    score = Score(clips=len(clips))
    for clip in clips:
        transcription = stt.transcribe(clip.audio)
        if not transcription.text.strip():
            score.empty += 1
            continue

        found = matcher.match_all(transcription.text)
        segments = split_transmissions(
            transcription.text,
            transcription.words,
            found,
            clip.duration_ms,
            min_gap_ms=gap_ms,
        )
        # After splitting, judge each transmission separately: a clip holding
        # two stations that split cleanly is a success, not a failure.
        for segment in segments:
            result = matcher.match(segment.text)
            if result.matched:
                score.clean += 1
                score.stations.add(result.callsign)
            else:
                score.empty += 1
        if len(segments) == 1 and len({f.callsign for f in found}) > 1:
            score.merged += 1
            score.clean -= 1
    return score


# --------------------------------------------------------------------------
# Sweeps
# --------------------------------------------------------------------------


def sweep_vad(samples, matcher, stt, silences, aggressiveness, gap_ms) -> dict:
    results: dict[int, Score] = {}
    print("\nVAD silence threshold -- how long a pause ends a transmission")
    print(HEADER)
    for silence_ms in silences:
        started = time.monotonic()
        segmenter = VadSegmenter(
            silence_ms=silence_ms, aggressiveness=aggressiveness, min_clip_ms=400
        )
        clips = list(segmenter.segment(frames_of(samples)))
        score = score_clips(clips, matcher, stt, gap_ms)
        score.seconds = time.monotonic() - started
        results[silence_ms] = score
        print(score.row(f"{silence_ms} ms"))
    return results


def sweep_split(samples, matcher, stt, silence_ms, aggressiveness, gaps) -> dict:
    """Sweep the split threshold. Segmentation is fixed, so this is cheap."""
    segmenter = VadSegmenter(
        silence_ms=silence_ms, aggressiveness=aggressiveness, min_clip_ms=400
    )
    clips = list(segmenter.segment(frames_of(samples)))
    # Transcribe once; every gap threshold reuses the same transcripts.
    cached = [(clip, stt.transcribe(clip.audio)) for clip in clips]

    print("\nSplit gap -- dead air before two callsigns count as two stations")
    print(HEADER)
    results: dict[int, Score] = {}
    for gap_ms in gaps:
        score = Score(clips=len(cached))
        for clip, transcription in cached:
            if not transcription.text.strip():
                score.empty += 1
                continue
            found = matcher.match_all(transcription.text)
            segments = split_transmissions(
                transcription.text,
                transcription.words,
                found,
                clip.duration_ms,
                min_gap_ms=gap_ms,
            )
            for segment in segments:
                result = matcher.match(segment.text)
                if result.matched:
                    score.clean += 1
                    score.stations.add(result.callsign)
                else:
                    score.empty += 1
            if len(segments) == 1 and len({f.callsign for f in found}) > 1:
                score.merged += 1
                score.clean -= 1
        results[gap_ms] = score
        print(score.row(f"{gap_ms} ms"))
    return results


def best(results: dict) -> int:
    """Highest score, breaking ties toward the existing default."""
    return max(results, key=lambda key: (results[key].value, -key))


def indistinguishable(results: dict, winner: int, tolerance: float = 0.5) -> list[int]:
    """Candidates that scored within `tolerance` of the winner.

    Reported by name rather than as "it was close", because "500 and 650 are
    the same on this recording" is useful and "the winner won by 0.1" is not.
    """
    top = results[winner].value
    return sorted(k for k, s in results.items() if k != winner and top - s.value < tolerance)


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--audio", required=True, help="a recording of your net")
    parser.add_argument("--roster", default="roster.csv")
    parser.add_argument("--model", default="base", help="tiny/base/small/...")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--silence", default="500,650,800,1000,1200",
        help="vad.silence_ms candidates",
    )
    parser.add_argument(
        "--gap", default="300,400,500,700,900", help="split.min_gap_ms candidates"
    )
    parser.add_argument("--aggressiveness", type=int, default=3)
    args = parser.parse_args(argv)

    roster = load_roster(args.roster)
    matcher = CallsignMatcher(roster=roster)
    samples = read_wav(args.audio)
    duration = len(samples) / TARGET_RATE
    print(f"Recording: {duration:.0f}s, roster: {len(roster)} stations, model: {args.model}")

    gain, note = recommend_gain(samples)
    print(f"\nLevel: {note}")
    if gain != 1.0:
        print(f"  suggested audio.gain: {gain}")

    stt = SttWorker(model_size=args.model, device=args.device)
    stt.load()
    stt.initial_prompt = stt.build_prompt(matcher.bias_terms(["net control", "QNI"]))

    silences = [int(v) for v in args.silence.split(",")]
    gaps = [int(v) for v in args.gap.split(",")]

    vad_results = sweep_vad(samples, matcher, stt, silences, args.aggressiveness, gaps[len(gaps)//2])
    best_silence = best(vad_results)
    split_results = sweep_split(samples, matcher, stt, best_silence, args.aggressiveness, gaps)
    best_gap = best(split_results)

    print("\n" + "=" * 78)
    print("Suggested config, from this recording:\n")
    print("vad:")
    print(f"  silence_ms: {best_silence}")
    print(f"  aggressiveness: {args.aggressiveness}")
    print("split:")
    print(f"  min_gap_ms: {best_gap}")
    if gain != 1.0:
        print("audio:")
        print(f"  gain: {gain}")

    # A win by a hair means the setting does not matter much here, and saying
    # so is more useful than presenting a coin flip as a result.
    for name, results, winner in (
        ("vad.silence_ms", vad_results, best_silence),
        ("split.min_gap_ms", split_results, best_gap),
    ):
        if len(results) < 2:
            continue
        tied = indistinguishable(results, winner)
        if tied:
            names = ", ".join(str(v) for v in tied)
            print(
                f"\nNote: {name}={winner} scored no better than {names} on this "
                "recording -- the choice between them is not settled by this "
                "audio. Anything not listed there did do worse."
            )

    print(
        "\nThese come from one recording. Re-run after a net that sounded "
        "different -- a contest weekend, a mobile check-in, bad conditions."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
