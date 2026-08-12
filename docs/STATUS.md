# Status: what is done, and what is left

Written honestly, because the interesting question about this app is not what
it does — it is what has actually been *proven* to work.

## Proven

Everything downstream of the audio device, against recorded and synthesised
audio, with 231 offline tests. Segmentation, transcription, callsign matching,
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
behind them, but not one has met a real net. **One recorded net through
`--file` replaces the entire table with measurements** — which is why the
bring-up doc puts recording ten minutes of traffic before anything else.

| Setting | Default | What it is guessing about |
| --- | --- | --- |
| `vad.silence_ms` | 800 | How long your operators pause while spelling a callsign |
| `vad.aggressiveness` | 3 | How much hiss your receiver passes between transmissions |
| `split.min_gap_ms` | 500 | The gap between two stations keying up, versus a pause mid-sentence |
| `voice.min_similarity` | 0.82 | How alike two recordings of one operator look over FM |
| `escalation.min_confidence` | 0.55 | Where "unsure" begins for your audio |
| `audio.gain` | 1.0 | Entirely dependent on what you plugged into what |

`roster.threshold` (78) is the exception: it sits just under the 80 that one
wrong character in a five-character callsign scores, which is arithmetic rather
than a guess.

## Off by default, waiting on a decision

- **`escalation.enabled`** — loads a second, larger model. Real memory, so it
  is opt-in. Turn it on where you have the RAM or a GPU.
- **`voice.enabled`** — needs a net or two of enrolment before it suggests
  anything, and its threshold needs tuning against real voices.

## Worth doing next

Roughly in order of value per effort:

1. **Run a net.** Everything above is downstream of this.
2. **A better voice embedder.** The current one is log-mel cepstral statistics
   in numpy, chosen so the app installs on a Pi without a deep-learning
   runtime. A trained speaker network (ECAPA-TDNN, TitaNet) would be
   substantially better and is nearly free on a GPU. `voice_id.embed()` is the
   only function that changes.
3. **Benchmark alternatives to Whisper.** NVIDIA Parakeet is faster and scores
   better on English; Riva offers real *word boosting*, which is a proper
   answer to the 224-token prompt ceiling rather than a workaround. `stt_worker.py`
   is the only module that knows which engine is in use.
4. **Make matching source-aware.** Per-frequency rosters currently bias
   decoding but do not influence matching. Preferring same-frequency stations
   would cut wrong matches on a 100-station roster — carefully, since people do
   turn up on the other frequency.
5. **Session recovery.** A crashed net leaves a complete `transcripts/*.jsonl`,
   but nothing reads it back in. A `--resume` flag would let a restarted app
   continue the same log rather than starting a second one.
6. **CI.** The tests only run when somebody remembers to run them.

## Known limitations, not bugs

- One Whisper model is shared across receivers, so two busy frequencies
  serialise. This is deliberate: two models on a Pi would thrash.
- A clip is matched to one callsign per transmission; the splitter handles two
  stations, but a single transmission naming several stations logs once.
- Voice suggestions need enrolment, so the first net of a new roster offers
  none.
- Transcription is voice only — no CW, no digital modes.
