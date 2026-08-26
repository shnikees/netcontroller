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

Everything downstream of the audio device, with 442 offline tests run on every
push across Python 3.11–3.13 plus a job with the optional libraries removed.

**As of 2026-08-16 that includes real human audio**: about two and a half hours
of a live repeater, pulled off a club's public stream. Not a radio plugged into
this machine, but real operators on a real net rather than a synthesizer, which
is the thing every earlier number was caveated on.

| Area | What works |
| --- | --- |
| Capture | Any input — SDR loopback, line in, microphone — at any sample rate, with per-source channel, gain and VAD settings |
| Multiple receivers | Independent capture, health and priority per source; a tab each on the dashboard |
| Segmentation, noisy feeds | A noise gate relative to the tracked floor, so an open squelch or streamed repeater segments at all -- 39 capped blocks became 223 transmissions on a real recording |
| Segmentation | One clip per transmission, and splitting a clip that caught two stations on the pause between them |
| Matching | Phonetic normalisation, fuzzy roster match, learned aliases, refusal when ambiguous |
| Positions | The roster carries each operator's post; every line and the export show it |
| Traffic | Declarations read off the transcript, badged, counted, filterable, and cleared with a click |
| Voice | Profiles learned from clean matches and corrections; suggestions on unmatched lines only |
| Attendance | Who turns up, learned from past sessions, used to order the prompt |
| Durability | Crash-safe transcripts, disk spill, `--resume`, watchdog with auto-restart |
| Operation | Settings panel, health strip, corrections, export, container image |
| Matching, generatively | Every roster callsign spelled, mutated the way Whisper renders text, and asserted to survive -- and never to come back as a different station |
| Spelling without phonetics | Letter names ("kay jay seven jay ex em") and hyphenated callsigns, both behind guards that keep ordinary English out |
| Roster from nothing | `mine_roster.py` proposes a roster from recordings alone, ranked by how many separate nets a callsign appears on and checked against a licence database |
| Batch replay | A folder of recordings processed in one pass, skipping what is already done and what is still being written |
| Prompt echo | A transcript that is the bias list read back names nobody -- the text is still logged, the callsigns are not trusted, and the reason is on the line |
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

## What real audio has now shown

About five hours of a live repeater went through the pipeline on 2026-08-16,
and a member of that net confirmed three callsigns on it -- `KJ7RAB`,
`KJ7JXM`, `KI7RMU` -- which is the first ground truth this project has had.
Six things came out of it that no amount of synthetic audio would have:

- **An open-squelch feed did not segment at all.** webrtcvad called 74% of the
  recording speech at *every* aggressiveness, because the gaps between overs
  carry AGC'd hiss rather than silence. 35 of 39 clips ran to the two-minute
  cap. The noise gate fixed it -- 223 transmissions at a 9.5 s median -- and
  nothing in the synthetic test set could have surfaced it.
- **Batch mode was silently discarding most of every recording.** A 75-minute
  net came back as 34 lines out of 223 clips and exited successfully. Found
  only because a real recording was long enough for the queue to spill.
- **The prefer-unmatched bias holds on real audio.** Of 50 callsign-shaped
  candidates mined, 22 are issued to nobody -- fragments of mangled
  conversation like `I7R` and `B0I`. Every one would be rejected by a roster.
- **People do not use phonetics on a conversational net.** They say "kay jay
  seven jay ex em", which produced no candidate at all, and mixed renderings
  silently dropped the prefix while matching on the remainder.
- **A bigger model recovered *fewer* callsigns.** Over a 30-minute slice with
  the three known stations in the prompt, `base` recovered 16, `small` 11 and
  `medium` 7 -- monotonically worse as the model grows, and the exact opposite
  of the synthetic benchmark where `medium` scored full marks. A larger model
  is a more confident language model, and a callsign is not language. See
  [HARDWARE.md](HARDWARE.md), which now leads with this.
- **The roster prompt outweighs everything else measured.** The same recordings
  processed *without* the known callsigns in the prompt yielded 8 recoveries
  across four hours; with them, 16 in twenty-four minutes. The token budgeting
  in `stt_worker.py` is doing more work than any model upgrade on offer.

What it still has *not* shown is recall. Three confirmed callsigns is enough to
compare models against each other, but nobody has counted how many times those
stations actually identified, so "16 recovered" has no denominator. The cheapest
way to get one is somebody listening to twenty clips and writing down what they
hear.

## Not proven: the live audio path

**No part of this has run against a real radio.** The audio above arrived over
the internet; nothing has been plugged into this machine. Specifically
unverified:

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
- Attendance against real history. It has now scored two sessions of real net
  audio, but with no roster to score against, so the ranking is untested.
- Any of it on a GPU. CUDA is auto-detected, the status strip reports what it
  resolved to, and none of it has been run against an actual NVIDIA device.
- The ONNX speaker backend against a *real* model. The adapter is tested
  against stand-ins for each input convention, but no downloaded ECAPA or
  TitaNet export has been run through it.

[FIELD-BRINGUP.md](FIELD-BRINGUP.md) is the checklist for closing this gap, in
an order where each step fails in a way you can diagnose.

## Not proven: every tuning constant

These are guesses. They are *reasoned* guesses, and several have arithmetic
behind them, but not one has met a real net *on the air*. Note that a streamed
feed cannot settle them either -- its gating, levels and inter-station delays
belong to a signal path you will not be using, which is why
[AUDIO-INPUT.md](AUDIO-INPUT.md) says not to run `tune.py` against one. **One recorded net replaces most
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
| `escalation.min_confidence` | 0.55 | Where "unsure" begins for your audio. Note that escalation itself is now in doubt: on real audio no larger model beat `base` at callsign recovery |
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
| `tools/mine_roster.py` | Nets already recorded | A *draft* roster, ranked by how many separate sessions each callsign appears on, optionally checked against a licence database. Proposes; never adopts |
| `tools/bench_engines.py` | A recording, and optionally a second engine | Realtime factor and callsign recovery per engine, on real clip-sized workloads — the measurement that should precede buying anything |

None of them needs hand-labelling: the roster is the supervision throughout.

## Off by default, waiting on a decision

- **`escalation.enabled`** — loads a second, larger model. Real memory, so it
  is opt-in, and it should now stay off until somebody shows it helps. The
  premise was that a bigger model rescues the lines a small one fumbled; on the
  only real audio measured, every bigger model recovered *fewer* callsigns than
  `base`. The mechanism is intact and worth keeping -- re-transcribing the
  unsure lines in the gaps is still the right shape -- but the model it
  escalates *to* is now an open question rather than an obvious one. A
  different engine may be the answer where a bigger Whisper is not.
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

1. **Run a net.** Everything above is downstream of this, and it is the only
   item that needs hardware rather than time. Recording a repeater over the
   internet has taken this a useful distance -- the pipeline has now met real
   voices, and two significant bugs fell out of that -- but a streamed feed
   cannot exercise the capture path, the device-restart path, or the
   thresholds, and a conversational net is the wrong *shape* for a race net
   regardless.

   The cheapest next increment, still without a radio: a **formal net**. A
   check-in net uses phonetics and reads a roll call, which is both the case
   this matcher was designed for and, effectively, a roster read aloud. PSRG
   links its repeaters for one on Monday evenings and it is on the same
   stream.

2. **Benchmark alternatives to Whisper.** Promoted in importance by the real-
   audio result: if a bigger Whisper makes callsign recovery *worse*, then more
   compute spent on Whisper is not the lever, and the interesting question is
   whether a different architecture behaves differently on strings that are not
   language. Parakeet is faster *and* scores better on English, so it buys
   accuracy and headroom at the same time rather than trading one for the
   other. Riva adds real *word boosting*, which is a proper answer to the
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

3. **Improve callsign recovery on real audio.** The whole point, and now the
   best-evidenced gap. Ranked by what the real-audio result predicts, not by
   effort. The first two need no roster and no hardware.

   **Engines to try, and why each is a real test rather than a shopping list:**

   | Candidate | The prediction it tests |
   | --- | --- |
   | **Parakeet TDT 0.6b** | A transducer with a far weaker internal language model. If "more language model, worse callsigns" is the mechanism, this should *invert* the trend rather than continue it. `whisper.cpp` already builds `parakeet-cli`, so it is one model download away |
   | **`large-v3-turbo`** | Full large encoder, decoder distilled from 32 layers to 4 — strong acoustics, weak language model. The hypothesis predicts it *beats* `medium` despite being nominally bigger, which is a falsifiable claim worth having |
   | **A pure CTC model** (wav2vec2) | No language model at all. The extreme end, which bounds the size of the effect. A one-off experiment rather than a candidate engine, since it drags in torch |
   | `large-v3` | Only to confirm the decline continues. Expensive, and the trend already predicts the answer |

   **Levers that are not the model, and are probably worth more:**

   - **`hotwords`.** faster-whisper supports it, and it is a *different*
     mechanism from `initial_prompt` -- it biases the decoder directly rather
     than seeding it with text. The prompt alone roughly doubled recovery, so
     this is the cheapest untested lever there is.
   - **Confusion-aware matching.** The near-misses are not random: `KG7RIB` for
     `KJ7RAB` is G↔J and I↔A, and `KG7JX` for `KJ7JXM` is G↔J again. Whisper's
     errors are *acoustically structured*, and `fuzz.ratio` treats every
     substitution as equally bad, so a two-character acoustic confusion scores
     the same as two arbitrary wrong letters. A distance weighted by how
     confusable letters are over FM would score the real confusions far higher
     **without** loosening the threshold for genuinely different callsigns.
     This attacks the measured failure mode directly, helps whatever engine
     wins, and is a matcher change rather than a compute change.
   - **Multi-pass agreement.** Accept a callsign only where two passes agree --
     the same model at two temperatures, or two different engines. Buys
     precision, and precision is what would let the threshold drop far enough
     to catch the near-misses safely.
   - **A real roster.** Still the largest single lever measured, and still the
     one thing here that cannot be done without a human.

   **Measuring any of it needs a denominator.** Twenty clips listened to and
   written down by hand turns every number above from relative into absolute.

4. **Make matching source-aware.** Per-frequency rosters currently bias
   decoding but do not influence matching. Preferring same-frequency stations
   would cut wrong matches on a 100-station roster — carefully, since people do
   turn up on the other frequency.
5. **Replay the audio behind a line.** Deferred deliberately, not forgotten.
   The appeal is settling "what did they actually say" when the transcript
   itself is in doubt — but a race trailer already has several people talking,
   and playing a clip back into that room adds to the problem it is meant to
   solve. Reading the line is quieter and faster.

   Revisit if transcripts turn out to be wrong often enough that the text
   cannot be trusted; the answer then is probably a better model rather than
   playback, but the stored clips make either possible. The clips kept for
   voice enrolment already prove the storage side works.

6. **Improve the logging logic.** Raised as a future item, not yet scoped.
   What exists today: `logging_setup.py` writes rotating application logs,
   `session_writer.py` writes the fsynced JSONL transcript that `--resume`
   reads back, and the export writes CSV and text at the end. The pieces work,
   but they were each added for their own reason and have never been looked at
   together. Four questions to settle before adding anything to them:

   **a. Who reads which file, and when?** Four readers, and it is not obvious
   any of them is well served. *During* a net nobody reads a file — the
   dashboard is the interface. *Straight after*, somebody writes up the event
   and wants the net record. *Months later*, somebody asks "was Turn 7 covered
   at last year's race" or "what was said at 10:42". And *when something
   breaks*, somebody wants to know why audio stopped at 09:15. The first three
   want the transcript; the fourth wants the application log; and nothing today
   was designed against any of them specifically.

   **b. What is missing from the net record?** Three concrete gaps, each
   verifiable in the code today. The session file opens with
   `{"type": "session", "started_at": ...}` and carries **no record of the
   roster or thresholds in force** — so a transcript cannot be reinterpreted
   later, because the roster it was matched against may have changed. There is
   **no session-end marker**, so a file that stops cannot be told apart from a
   net that crashed. And timestamps are written **without a timezone**
   (`isoformat()` on a naive local datetime), which is fine on one machine and
   wrong the moment a record crosses a timezone or a DST boundary.

   **c. Should the two records converge?** A device dying, a model being
   swapped mid-net, an operator changing a threshold — all go to the
   application log only, and share no key with the session file. So "why did
   the transcripts get worse after 10:00" requires reading two files side by
   side and matching them up by eye. Putting operational events into the
   session JSONL would make one file the whole account of a net; the cost is a
   transcript format that is no longer only transcripts.

   **d. What is the retention policy, and who is silently depending on it?**
   The application log rotates by size (`maxBytes`/`backupCount`), so a busy
   install can lose the log covering an earlier net while that net's transcript
   survives — the two records age out on unrelated schedules. Session files are
   never pruned at all. And **attendance is derived from them at every
   startup**, so deleting old transcripts quietly changes the decoding priors
   for the next net. Any retention rule has to be decided with that coupling in
   view, not after.

   **First concrete requirement: attendance per day.** Today `attendance.py`
   answers "who is likely to be on next time" — a decayed aggregate, held in
   memory, recomputed from the transcripts at every startup and never written
   anywhere. What is missing is the plain historical record: *who was on, on
   which date, on which frequency, and how much did they say*. That is a
   different artifact with a different reader — the aggregate is for the
   decoder, the per-day record is for a human writing up an event afterwards
   or answering "did we have Turn 7 covered at last year's race".

   Worth noting it needs no new capture: every session file already holds it,
   so this is a report over data on disk rather than a change to what gets
   recorded. It also answers half the scoping question above — this is the
   transcript side, not the application logs.

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
