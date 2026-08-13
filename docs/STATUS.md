# Status: what is done, and what is left

Written honestly, because the interesting question about this app is not what
it does — it is what has actually been *proven* to work.

## What this is actually for

A **high-traffic event or race net**, run from a trailer with several people
talking at once, not an orderly weekly check-in. The job is *"oh shit, what was
that last transmission"* — making sure nothing gets lost when three things
happen simultaneously and the person who needed to hear one of them was
talking to somebody else.

That shifts what matters, and parts of this app were built assuming the other
kind of net:

- **The callsign is a location.** This is the thing to understand about an
  event net: operators are posted around the course, so the callsign is how
  net control knows *where* a transmission came from. The roster carries each
  operator's post and every line shows it. "Car off, need a tow" is
  only actionable once you know it came from Turn 7. Callsign attribution is
  not secondary to the transcript here — the two together are the product, and
  a wrong callsign is a wrong location.

  That makes the matcher's existing bias — refuse rather than guess — more
  right, not less. An unmatched line sends somebody to ask; a confidently
  wrong one sends help to the wrong part of the course.

- **Voice identification matters more, not less.** Event traffic often carries
  no callsign at all, and a trailer full of people is exactly where "who said
  that" is hardest — and, because callsign means position, most consequential.
- **Everyone speaks repeatedly**, so ordering the prompt by "who has not
  checked in yet" is the wrong prior for this use — attendance frequency and
  recency are better ones.
- **Overlapping and back-to-back speech is the normal case**, not an edge case,
  which puts the splitter and the VAD thresholds on the critical path rather
  than at the margins.

FCC callsign lookup is deliberately *not* on the list below: it is being
handled in the rally deployment app instead.

## Proven

Everything downstream of the audio device, against recorded and synthesised
audio, with 378 offline tests run on every push across Python 3.11–3.13 plus a
job with the optional libraries removed.

| Area | What works |
| --- | --- |
| Capture | Any input — SDR loopback, line in, microphone — at any sample rate, with per-source channel, gain and VAD settings |
| Multiple receivers | Independent capture, health and priority per source; a tab each on the dashboard |
| Segmentation | One clip per transmission, and splitting a clip that caught two stations on the pause between them |
| Matching | Phonetic normalisation, fuzzy roster match, learned aliases, refusal when ambiguous |
| Positions | The roster carries each operator's post; every line and the export show it |
| Traffic | Declarations read off the transcript, badged, counted, filterable, and cleared with a click |
| Voice | Profiles learned from clean matches and corrections; suggestions on unmatched lines only |
| Attendance | Who turns up, learned from past sessions, used to order the prompt |
| Durability | Crash-safe transcripts, disk spill, `--resume`, watchdog with auto-restart |
| Operation | Settings panel, health strip, corrections, export, container image |
| Escalation | Queued lines badged *waiting* and counted on the strip; re-transcribed ones badged *2nd pass*, and never left stranded when a pass fails or is dropped |

Some of it was verified in ways worth trusting:

- **Crash safety** — `kill -9` mid-net, no cleanup: the complete log was on
  disk. Restarting with `--resume` continued the same log with the five
  check-ins intact.
- **A slow transcriber** — six transmissions through a queue of depth one with
  a deliberately slow model: all six logged, in order, nothing lost.
- **Two stations in one clip** — 0.6 s apart, inside the VAD window: split into
  two correctly attributed lines.
- **Voice** — a station checked in, spoke again later with no callsign at all,
  and was suggested correctly.
- **A model swapped mid-net** — `tiny` to `base` partway through a recording:
  all six lines still landed, the pause costing latency rather than audio.
- **Traffic** — declarations detected including one that never says the word,
  denials not flagged, and the filter narrowing four lines to the two holding.

## Not proven: the live audio path

**No part of this has run against a real radio.** Everything above went through
`--file` or a synthetic device. Specifically unverified:

- Opening a real capture device by name (`find_device`) against real
  PulseAudio names.
- Two receivers on two real sound cards. Tested with two recorded files.
- A device that *vanishes mid-net* — a USB SDR unplugged. The restart-with-
  backoff path has only been exercised against a device that never existed.
- The container's PulseAudio socket mount. The image builds and serves; it has
  never carried audio.
- The ring buffer's drop path under a machine genuinely falling behind.
- The settings panel changing something that matters *while it matters*. Each
  control is tested; none has been used under pressure.
- Attendance against real history. It has only ever scored sessions produced by
  replaying the same handful of recordings.
- Any of it on a GPU. CUDA is auto-detected, the status strip reports what it
  resolved to, and none of it has been run against an actual NVIDIA device.
- The ONNX speaker backend against a *real* model. The adapter is tested
  against stand-ins for each input convention, but no downloaded ECAPA or
  TitaNet export has been run through it.

[FIELD-BRINGUP.md](FIELD-BRINGUP.md) is the checklist for closing this gap, in
an order where each step fails in a way you can diagnose.

## Not proven: every tuning constant

These are guesses. They are *reasoned* guesses, and several have arithmetic
behind them, but not one has met a real net. **One recorded net replaces most
of this table with measurements**, and `tools/tune.py` does the measuring:

```bash
python tools/tune.py --audio net-recording.wav --roster roster.csv
```

No hand-labelling — the roster is the supervision. A good setting is one where
each clip comes out as exactly one confident roster match; a setting that runs
two stations together leaves two callsigns in one clip, and one that cuts too
early leaves fragments with none. Both are countable without anyone
transcribing anything by hand.

It reports the evidence rather than just a verdict, and says so plainly when
two candidates are indistinguishable on the audio you gave it.

| Setting | Default | What it is guessing about |
| --- | --- | --- |
| `vad.silence_ms` | 800 | How long your operators pause while spelling a callsign |
| `vad.aggressiveness` | 3 | How much hiss your receiver passes between transmissions |
| `split.min_gap_ms` | 500 | The gap between two stations keying up, versus a pause mid-sentence |
| `voice.min_similarity` | 0.82 | How alike two recordings of one operator look over FM |
| `escalation.min_confidence` | 0.55 | Where "unsure" begins for your audio |
| `audio.gain` | 1.0 | Entirely dependent on what you plugged into what — measured, not searched: `tune.py` reads the level and solves for it |
| `traffic` phrasing | — | The phrase tables assume how *your* net announces traffic. Wrong wording shows up as missing badges, not wrong ones |
| `attendance.DECAY` | 0.85 | How fast a crew turns over between events |

`roster.threshold` (78) is the exception: it sits just under the 80 that one
wrong character in a five-character callsign scores, which is arithmetic rather
than a guess.

## The tools that settle the guesses

| Tool | What it needs | What it gives |
| --- | --- | --- |
| `tools/tune.py` | A recording and the roster | VAD and split thresholds, and the input level, measured rather than guessed |
| `tools/calibrate.py` | Nets already run | Escalation and voice thresholds, from the transcripts and profiles the app writes anyway. `--apply` patches the config |
| `tools/rebuild_voices.py` | Kept enrolment audio | Profiles rebuilt after an embedder change; `--compare` scores two embedders on identical clips |
| `tools/make_test_audio.py` | Nothing | Synthetic net audio, for exercising the pipeline with no radio |
| `tools/bench_engines.py` | A recording, and optionally a second engine | Realtime factor and callsign recovery per engine, on real clip-sized workloads — the measurement that should precede buying anything |

None of them needs hand-labelling: the roster is the supervision throughout.

## Off by default, waiting on a decision

- **`escalation.enabled`** — loads a second, larger model. Real memory, so it
  is opt-in. Turn it on where you have the RAM or a GPU.
- **`voice.enabled`** — needs a net or two of enrolment before it suggests
  anything, and its threshold needs tuning against real voices.

Both can now be switched on from the dashboard without restarting.

## On hardware

Moved to [HARDWARE.md](HARDWARE.md), which now carries the engine and
model-size benchmarks, the accelerator options, and the buying argument.

The conclusion in one line: **nothing needs buying yet.** `base` scores the
same as `medium` on the test net once the normalizer stops discarding
callsigns, and runs at a fraction of realtime on an ordinary CPU. The number
that would change that is `speed` on the status strip, measured against a real
net.

## Where the CPU goes

Decided deliberately: **during an event, spare cycles go to transcription
quality, not to features.** The transcript is what stops something being lost,
and everything else is downstream of it being right.

Two consequences worth knowing:

- Escalation is the intended use of idle time — a second pass with a bigger
  model on the lines the first was unsure about. On a genuinely busy net there
  may be *no* idle time, so those lines will land after the event rather than
  during it. That is the right trade for a race net: late and correct beats
  fast and wrong, and nothing is lost either way.
- Anything that competes for cycles mid-event needs to justify itself against
  a larger model. Audio conditioning (0.8 ms) and voice embedding (0.6 ms) are
  noise against a transcription measured in seconds; audio playback,
  re-rendering, and anything speculative are not automatically so.

## Worth doing next

Roughly in order of value per effort, with transcription quality ranked above
features:

1. **Run a net.** Everything above is downstream of this.

2. **Benchmark alternatives to Whisper.** The highest-value item under "cycles
   go to transcription": Parakeet is faster *and* scores better on English, so
   it buys accuracy and headroom at the same time rather than trading one for
   the other. Riva adds real *word boosting*, which is a proper answer to the
   224-token prompt ceiling instead of the packing heuristic in use now — and
   with 40-100 stations across two frequencies, that ceiling is a live
   constraint rather than a theoretical one.

   `stt_worker.py` is the only module that knows which engine is in use, and
   the escalation design means a second engine can be tried on the hard lines
   before committing to it for the live ones.

   **Worth doing before buying hardware, not after**, since the engine decides
   which accelerators are candidates at all — CTranslate2 being CUDA-only is
   the whole reason the buying list is NVIDIA. Measured comparison, the
   accelerators a Vulkan-capable engine would unlock, and the buying argument
   are in [HARDWARE.md](HARDWARE.md).

   Two findings from that benchmark belong here rather than there. `small`
   scored *worse* than `base` at callsign recovery in every engine
   configuration, which makes the current `escalation.model_size` default of
   `small` the worst available choice if a real net reproduces it. And a build
   of whisper.cpp ships a `parakeet-cli`, so the Parakeet half of this item can
   be tested through the same binary rather than a second stack.

3. **Make matching source-aware.** Per-frequency rosters currently bias
   decoding but do not influence matching. Preferring same-frequency stations
   would cut wrong matches on a 100-station roster — carefully, since people do
   turn up on the other frequency.
4. **Replay the audio behind a line.** Deferred deliberately, not forgotten.
   The appeal is settling "what did they actually say" when the transcript
   itself is in doubt — but a race trailer already has several people talking,
   and playing a clip back into that room adds to the problem it is meant to
   solve. Reading the line is quieter and faster.

   Revisit if transcripts turn out to be wrong often enough that the text
   cannot be trusted; the answer then is probably a better model rather than
   playback, but the stored clips make either possible. The clips kept for
   voice enrolment already prove the storage side works.

5. **Property-test the matcher instead of only regressing it.**
   `_spell_phonetically()` already turns a callsign into its spoken form, so
   every roster entry can be spelled, mutated with the *rendering* artifacts
   Whisper is known to produce -- hyphenation, glued phonetics, ordinals, Roman
   numerals, a split "niner" -- and asserted to still match. No audio and no
   hand-labelling: the roster is the ground truth, as everywhere else here.
   All four normalizer bugs fixed on 2026-08-12 were single mutations of that
   kind, which is the argument for it -- regressions only record misses that
   already happened.

   The second half is mining `feedback.jsonl`: every operator correction is a
   labelled pair of raw text and the right answer, so a tool could emit
   ready-made regression cases and close the copy-paste loop TESTING.md
   currently describes by hand.

6. **Improve the logging logic.** Raised as a future item, not yet scoped.
   What exists today: `logging_setup.py` writes rotating application logs,
   `session_writer.py` writes the fsynced JSONL transcript that `--resume`
   reads back, and the export writes CSV and text at the end. The pieces work,
   but they were each added for their own reason and have never been looked at
   together — so the open questions are which of the three a reader actually
   goes to after a net, what is missing from each, and whether the split
   between them is the right one. Worth settling before adding anything to
   them.

7. **Review after the net, not during it.** Everything self-supervised is in
   place, but the operator-supplied labels — corrections — currently have to
   be made live. A post-net review mode, working from the session file and the
   clip audio, would let those be batched into a few minutes afterwards
   instead of requiring somebody at the keyboard while the net runs.

   This matters more than its position suggests for a net nobody can attend:
   it converts "I was there" into "I had ten minutes that week".

## Known limitations, not bugs

- One Whisper model is shared across receivers, so two busy frequencies
  serialise. This is deliberate: two models on a Pi would thrash.
- A clip is matched to one callsign per transmission; the splitter handles two
  stations, but a single transmission naming several stations logs once.
- Traffic detection reads declarations off the transcript; whether it was
  *passed* is the operator's click, not something the app infers from later
  transmissions.
- Voice suggestions need enrolment, so the first net of a new roster offers
  none. Attendance is the same: the first event has no history to order the
  prompt by, and both improve every time the app is run.
- Settings changed from the dashboard apply to the running process. Saving to
  `config.yaml` is a separate click, so a change made mid-net and never saved
  is gone at the next restart — deliberate, but worth knowing.
- Transcription is voice only — no CW, no digital modes.
