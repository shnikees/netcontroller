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
audio, with 329 offline tests. Segmentation, transcription, callsign matching,
splitting a clip that caught two stations, corrections and alias learning,
voice suggestions, multi-source capture, buffering and disk spill, live
transcript writing, the watchdog, export, and the container image.

Some of it was verified in ways worth trusting:

- **Crash safety** — `kill -9` mid-net, no cleanup: the complete log was on
  disk.
- **A slow transcriber** — six transmissions through a queue of depth one with
  a deliberately slow model: all six logged, in order, nothing lost.
- **Two stations in one clip** — 0.6 s apart, inside the VAD window: split into
  two correctly attributed lines.
- **Voice** — a station checked in, spoke again later with no callsign at all,
  and was suggested correctly.

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

`roster.threshold` (78) is the exception: it sits just under the 80 that one
wrong character in a five-character callsign scores, which is arithmetic rather
than a guess.

## Off by default, waiting on a decision

- **`escalation.enabled`** — loads a second, larger model. Real memory, so it
  is opt-in. Turn it on where you have the RAM or a GPU.
- **`voice.enabled`** — needs a net or two of enrolment before it suggests
  anything, and its threshold needs tuning against real voices.

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

2. **Expected stations from history.** The roster is hand-maintained; the
   transcripts already record who actually turns up. Deriving attendance from
   past sessions gives a better prior for prompt biasing than roster order —
   on an event net, frequency and recency beat "not yet heard from" — and it
   keeps itself current as the crew changes.

   `calibrate.load_entries()` already reads every past session, so the data is
   in hand. The trap is auto-adding unknown callsigns: a mis-transcription that
   became a "station" would then bias decoding toward its own mistake, so
   anything not on the roster stays a suggestion for a human.

3. **Benchmark alternatives to Whisper.** The highest-value item under "cycles
   go to transcription": Parakeet is faster *and* scores better on English, so
   it buys accuracy and headroom at the same time rather than trading one for
   the other. Riva adds real *word boosting*, which is a proper answer to the
   224-token prompt ceiling instead of the packing heuristic in use now — and
   with 40-100 stations across two frequencies, that ceiling is a live
   constraint rather than a theoretical one.

   `stt_worker.py` is the only module that knows which engine is in use, and
   the escalation design means a second engine can be tried on the hard lines
   before committing to it for the live ones.

4. **A better voice embedder.** The current one is log-mel cepstral statistics
   in numpy — 24 numbers describing average timbre, never trained to tell one
   speaker from another. It also conflates the voice with the *channel*, so a
   profile is really "Frank on his HT" and breaks when he checks in mobile.

   A trained speaker network (ECAPA-TDNN, TitaNet) is discriminatively trained
   on thousands of speakers and augmented specifically for channel robustness.
   The route that preserves "installs on a Pi with no deep-learning stack" is
   an **ONNX export run under `onnxruntime`** — an 18 MB wheel plus a ~25 MB
   model, against hundreds of megabytes for PyTorch. Best added as
   `voice.backend: mfcc | ecapa` so both can be compared.

   `voice_id.embed()` is the only function that changes, and the evaluation
   already exists: `tools/calibrate.py` prints same-station against
   different-station similarity distributions, so the separation between those
   two histograms *is* the measurement of whether the upgrade helped.

   The clips are already being kept (`voice.keep_audio`), so this is now a
   swap-and-rebuild rather than weeks of re-enrolment:
   `tools/rebuild_voices.py --compare` scores both embedders on identical
   audio.

5. **Make matching source-aware.** Per-frequency rosters currently bias
   decoding but do not influence matching. Preferring same-frequency stations
   would cut wrong matches on a 100-station roster — carefully, since people do
   turn up on the other frequency.
6. **Replay the audio behind a line.** Deferred deliberately, not forgotten.
   The appeal is settling "what did they actually say" when the transcript
   itself is in doubt — but a race trailer already has several people talking,
   and playing a clip back into that room adds to the problem it is meant to
   solve. Reading the line is quieter and faster.

   Revisit if transcripts turn out to be wrong often enough that the text
   cannot be trusted; the answer then is probably a better model rather than
   playback, but the stored clips make either possible. The clips kept for
   voice enrolment already prove the storage side works.

7. **Review after the net, not during it.** Everything self-supervised is in
   place, but the operator-supplied labels — corrections — currently have to
   be made live. A post-net review mode, working from the session file and the
   clip audio, would let those be batched into a few minutes afterwards
   instead of requiring somebody at the keyboard while the net runs.

## Known limitations, not bugs

- One Whisper model is shared across receivers, so two busy frequencies
  serialise. This is deliberate: two models on a Pi would thrash.
- A clip is matched to one callsign per transmission; the splitter handles two
  stations, but a single transmission naming several stations logs once.
- Traffic detection reads declarations off the transcript; whether it was
  *passed* is the operator's click, not something the app infers from later
  transmissions.
- Voice suggestions need enrolment, so the first net of a new roster offers
  none.
- Transcription is voice only — no CW, no digital modes.
