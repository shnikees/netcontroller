# Testing

```bash
.venv/bin/python -m pytest
```

CI runs this on every push, on Python 3.11, 3.12 and 3.13 — the versions
Raspberry Pi OS and Debian stable ship, plus one canary. A second job installs
**without `soxr` or `scipy`**, because both have fallbacks that only run when
the library is missing, and a fallback nobody exercises is a fallback that is
broken. That job earned its keep immediately: the scipy-free high-pass turned
out not to remove rumble at all, and is now a boxcar filter that does.

417 tests, all offline, no audio hardware. About 20 seconds, most of it
the pipeline tests deliberately running a slow transcriber in real time.

## What each suite covers

| Suite | Covers | Approach |
| --- | --- | --- |
| `test_callsign_match.py` | Normalizer, candidate extraction, roster matching, threshold and ambiguity behaviour | Realistic messy strings, including verbatim Whisper output |
| `test_vad_segmenter.py` | Clip boundaries: splitting, merging, min length, pre-roll, hangover trim, max length, end-of-stream flush | Scripted speech patterns through a stubbed VAD |
| `test_transcript_store.py` | Session log, check-in list, CSV/text export | Direct |
| `test_feedback.py` | Correction log, alias derivation, matcher applying aliases | Direct, plus a simulated truncated write |
| `test_server.py` | HTTP API, mainly `/api/correct` | FastAPI `TestClient` |
| `test_session_writer.py` | Writing the session to disk mid-net, corrections as history, truncated writes, unwritable disks | Reads the file back *during* the session, never only after a clean close |
| `test_clip_split.py` | Splitting a clip with two stations, and — more importantly — every case where it must refuse | Synthetic word timings, so the pause is exact |
| `test_audio_prep.py` | Conditioning: quiet input lifted, hiss not, rumble removed, waveform undistorted | Tones and noise, checked with an FFT |
| `test_voice_id.py` | Voice embedding, enrolment, persistence, and every case where a suggestion must be withheld | Synthetic speakers: a glottal buzz shaped by fixed formants |
| `test_calibrate.py` | Threshold calibration from collected data, and every case where it must refuse to answer | Synthetic distributions, including ones with no signal in them |
| `test_traffic.py` | Reading traffic declarations, weighted toward the denials and the question forms that must not be flagged | Verbatim net phrasing |
| `test_settings.py` | Which settings exist, their bounds, and patching config.yaml without destroying it | Direct, including a config file with comments to preserve |
| `test_attendance.py` | Scoring who turns up, and refusing to adopt callsigns that are not on the roster | Synthetic session histories |
| `test_voice_onnx.py` | The ONNX adapter against every input convention a downloaded model might use | Stand-in models built on the spot with `onnx`, since a real one is tens of megabytes |
| `test_health.py` | Watchdog state machine: stalls, silence, restarts, backlog | Injected clock, so a 5-minute silence tests in microseconds |
| `test_resample.py` | Resampling to 16 kHz | Tones in, FFT out — run against *both* engines, so the fallback is not dead code |
| `test_buffering.py` | Ring buffer and disk spill: wraparound, overrun policy, concurrency, corrupt files | Direct, with real threads |
| `test_pipeline.py` | The whole chain with a deliberately slow transcriber, plus multi-source capture | End-to-end over synthetic recordings; asserts nothing is lost and a dead receiver does not stop the others |

`test_vad_segmenter.py` deliberately stubs out webrtcvad. Whether webrtcvad
correctly identifies speech is webrtcvad's problem; where the segmenter draws
clip boundaries given a speech pattern is ours, and that is what is pinned.

## Generating test audio without an SDR

```bash
python tools/make_test_audio.py
```

```bash
python app.py --file test-net.wav --model tiny
```

This speaks a set of check-ins with the system TTS (`say` on macOS,
`espeak-ng` on Linux), splices in dead air and hiss, and writes a WAV the app
can replay. It exercises the entire pipeline except the audio device itself.

Use your own script to test against your actual roster:

```bash
python tools/make_test_audio.py --script my-net.txt --gap 2
```

One transmission per line; `#` comments are ignored.

**Read the results with appropriate suspicion.** TTS enunciates far better than
a handheld into a repeater. It is a plumbing test and a source of regression
cases — not evidence the matcher will hold up on the air. Real audio is the
only thing that proves that, which is what
[FIELD-BRINGUP.md](FIELD-BRINGUP.md) is for.

## Tuning against real audio

`tools/tune.py` measures the thresholds that `tools/make_test_audio.py` cannot
tell you anything about:

```bash
python tools/tune.py --audio net-recording.wav --roster roster.csv
```

Synthetic speech is uniform, so it cannot say how long *your* operators pause
or how much dead air sits between two stations on *your* repeater. A recording
can, and the roster does the labelling.

## Comparing engines and model sizes

`tools/bench_engines.py` runs a recording through the VAD and then through one
or more engines, reporting realtime factor and *callsign recovery* rather than
word error rate — a transcript that loses the callsign has lost where on the
course the transmission came from.

```bash
python tools/bench_engines.py --audio net.wav --roster roster.csv \
    --expected callsigns.txt --repeat 5
```

`--expected` is one callsign per line in transmission order, which is the only
labelling it needs. Use `--repeat 5` and read the median: run-to-run spread on
a laptop is around ten percent, enough to reverse a ranking by itself. Results
so far are in [HARDWARE.md](HARDWARE.md).

## Properties, for the misses that have not happened yet

`test_matcher_properties.py` is the other half of the regression block below.
Regressions record ways a callsign has *already* been mangled; the property
suite generates callsigns, spells them with the same phonetic table the app
uses, mutates them the way Whisper renders text, and asserts they survive.

Two tiers, and the second is the point:

- **Lossless renderings must still match.** Hyphenation, gluing, ordinals,
  Roman numerals, a split "niner" -- these lose nothing, so a failure is a
  normalizer bug.
- **No rendering may ever produce a *different* roster callsign.** Coming back
  unmatched is a disappointment; coming back as somebody else sends help to the
  wrong part of the course.

It needs no audio, no labelling and no `hypothesis` dependency -- generation is
seeded, so a failure reproduces from its own output. It earned its place on the
first run by finding a live bug: the letter **A**, arriving as a bare character
in a collapsed suffix ("whiskey 6ABC"), was being read as the English article
in `FILLER_WORDS`, dropped, and -- because a dropped word emits a `BREAK` --
taking the rest of the callsign with it. Every callsign with an A in a
collapsed suffix was affected, and no net had reported it.

It also runs against a real roster when there is one, which generated
callsigns cannot substitute for — event nets collect callsigns that genuinely
resemble each other:

```bash
python -m pytest test_matcher_properties.py -q -s   # uses roster.csv if present
```

Those checks are skipped when no roster is present, so CI is unaffected.

**When you add a normalizer rule, add its mutation here too.** That is what
turns one fix into a property that holds for every callsign rather than the one
in the bug report.

## Adding a regression

This is the main maintenance loop, and it is deliberately cheap. When a real
net produces a miss:

**1. Copy the exact transcript** from the log line or the exported CSV's
`raw_text` — verbatim, including the model's odd capitalization and spelling.

**2. Add it to the regression block** at the bottom of
`test_callsign_match.py`, under "Regressions from real transcripts":

```python
def test_whisper_hears_lima_as_lisa() -> None:
    result = matcher.match("lisa seven x-ray yankee zulu checking in")
    assert result.matched
    assert result.callsign == "K7XYZ"
```

**3. Watch it fail**, so you know the test is real.

**4. Add the missing spelling** to the tables in `callsign_match.py`:

- `PHONETIC_MAP` — a new way the model spells a phonetic (`"lisa": "L"`).
- `DIGIT_MAP` — a new way it spells a digit.
- `AMBIGUOUS_DIGIT_MAP` — for words that are digits *only* inside a callsign
  ("for", "fifth"). These convert only when flanked by spelling tokens.
- `FILLER_WORDS` — net phrases that should never be part of a callsign.

If the miss is not a *mishearing* but a *rendering* — the model heard correctly
and wrote it oddly — the fix is a splitter rather than a table entry:
`_split_alphanumeric` (ordinals, `9er`, phonetics welded to digits),
`_split_hyphens`, `_split_glued_phonetics`, or `_is_niner_tail`. Keep these
narrow: each one must be unable to fire on ordinary English, and each needs a
guard test proving it does not.

**5. Run the whole suite.** The existing tests are there to catch a new entry
that breaks an old case — particularly in `AMBIGUOUS_DIGIT_MAP`, where a
too-eager entry turns ordinary English into callsign digits.

### What not to add

Resist adding *mishearings* from synthesized audio. The TTS test set has
produced artifacts like `Fictor` for "victor" and `quiddac` for "quebec" that
no human voice would generate. Padding the tables with those makes matching
looser for no real-world gain, and looser matching means wrong callsigns.

The bar for a table entry: **have I heard a real transmission produce this?**
If not, leave it.

**Renderings are the exception, and the distinction matters.** `9er` for
"niner", `III` for "three", `Alpha4PQ`, a hyphen welding two spelled words —
these are choices Whisper makes about *how to write text it heard correctly*,
and it makes them the same way whatever the audio source was. They generalise
off synthetic audio in a way an acoustic artifact never does, and every one of
them is a callsign silently discarded rather than a match made looser. The
regressions added from the `bench_engines.py` run are all of this kind.

The test: would this artifact survive if the same words came from a human on a
handheld? A mishearing would not. A rendering would.

## Tuning thresholds

If real traffic shows the matcher too strict or too loose, the knobs are
`roster.threshold` and `roster.ambiguity_margin`.

Some arithmetic to reason with: `rapidfuzz.fuzz.ratio` scores one wrong
character in a five-character callsign at 80, and two at 60. The default
threshold of 78 therefore forgives exactly one slip. Raising it to 85 demands a
clean read; dropping it to 70 starts accepting two-character errors, which on a
roster of similar callsigns will produce confident wrong matches.

Change the threshold, then run the suite — several tests assert behaviour right
at that boundary and will tell you what you traded away.
