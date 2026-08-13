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
audio, with 364 offline tests run on every push across Python 3.11–3.13 plus a
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

Worth recording because it changes what to buy. **Escalation lowers the bar
substantially**: the big model only has to handle the lines the fast one was
unsure about, in the gaps, so the machine does not need to run `large-v3` on
every transmission in real time.

That makes `base` or `small` live on CPU with `large-v3` on escalation a
comfortable target for a 6 GB laptop GPU, rather than something needing a
desktop card. A laptop also brings a built-in UPS, which in a trailer running
off a generator is worth more than the extra speed of a desktop.

The number that settles it is on the status strip: **speed** (the realtime
factor). Under about 0.5× there is headroom for a bigger model; near 1.00×
there is not. The strip also shows which device inference actually resolved to,
because `device: auto` quietly falling back to the CPU on a machine with a GPU
is a thing to notice beforehand rather than during.

### Cost against performance

There is no deadline here, which makes the sensible order **measure, then
buy**. The app already reports the number that decides it: the realtime factor
on the status strip, transcription time over audio length.

1. **Start with the machine you have.** Run a recording through `--file` with
   `base`, and read `speed` off the strip. Under about 0.5× there is headroom
   for a bigger model; near 1.00× there is not. That measurement costs nothing
   and answers the question better than any table.
2. **Remember what escalation changes.** The big model only handles the lines
   the fast one was unsure about, in the gaps — so "can it run `large-v3` on
   every transmission" is the wrong question. `base` live with `small` or
   `medium` on escalation is a much cheaper target, and on an event net with
   any gaps at all it catches up.
3. **Only then buy**, and buy for the measurement rather than the spec sheet.

Rough tiers, cheapest first. Prices move and the used market moves faster, so
treat the ordering as the useful part:

| Spend | What it buys |
| --- | --- |
| Nothing | Existing machine, `base` on CPU. Genuinely worth testing before assuming it is not enough |
| Mini PC | `base`/`small` on CPU, reliable and quiet, no CUDA |
| Jetson Orin Nano | Everything up to `large-v3`, at 25 W, in exchange for an evening of build friction |
| Used RTX laptop | The most compute per pound, plus a screen and a UPS |
| RTX A2000 12 GB | The same capability in a box that is already there, at 70 W |

The honest summary: **a GPU is a convenience here, not a requirement**, and
which one matters less than whether the live model is small enough to keep up
while escalation quietly does the hard lines behind it.

### Measured: whisper.cpp against faster-whisper

Run rather than assumed, since the engine choice gates the hardware choice.
18 synthetic event-net transmissions (86 s of speech, `tools/make_test_audio.py`),
cut into clips by the project's own VAD so this measures the real workload — short
clips, `base`, beam 5, the roster prompt — rather than one long file. Apple M1
Pro, 10 core. Median of five runs; `ok` counts transmissions where the project's
own matcher recovered the right roster callsign. Reproduce with
`tools/bench_engines.py`.

| Engine | Device | Compute | Realtime | Callsigns |
| --- | --- | --- | --- | --- |
| faster-whisper, int8 | CPU | 11.2 s | 0.130× | 15/18 |
| whisper.cpp, fp16 | CPU | 8.2 s | 0.095× | 13/18 |
| whisper.cpp, fp16 | GPU (Metal) | **3.3 s** | **0.039×** | 14/18 |

Three things fall out of this, and only the third is much of an argument for
switching.

**On the CPU whisper.cpp is faster, but not decisively.** Around 1.4× here,
and that figure is the least trustworthy one in the table: faster-whisper was
steady near 11.2 s across samples while whisper.cpp ranged 7.0–8.8 s, so the
ratio moves between 1.3× and 1.6× depending on the run. An earlier median of
three runs put them level, which was simply too small a sample. Treat this row
as "somewhat faster", not as a number.

**The 3.4× is the GPU, and the GPU is the whole point.** faster-whisper could
not touch this machine's GPU at all: CTranslate2 speaks CUDA, and there is no
CUDA here. whisper.cpp used Metal and took less than a third of the time. That
is the Vulkan argument in miniature — the win is not a better engine, it is
*being allowed to use hardware that is already present*. The CPU row is a
tuning difference; this row is a capability difference, and only capability
differences justify the disruption of changing engines.

**Nothing produced a wrong callsign.** Zero across every configuration; the
losses were all lines left unmatched. The prefer-unmatched bias is a property of
the matcher rather than of Whisper, so it survives an engine swap — which is
what makes swapping engines a reasonable thing to consider at all.

Two cautions about the numbers. The audio is TTS, which enunciates far better
than a handheld into a repeater, so every accuracy figure here is optimistic and
only the *ranking* is worth anything. And the roster prompt matters more than
the engine did: dropping it cost faster-whisper 15→12 and whisper.cpp 14→10. On
whisper.cpp `-mc 0`, the apparent analogue of `condition_on_previous_text=False`,
silently discards the initial prompt as well — worth knowing, since it looks
like a fair comparison and is not. Whatever engine ends up in front, the prompt
budget in `stt_worker.py` is doing more work than the model choice.

### What to run it on

Ranked for *this* use — a trailer, mains that may be a generator, gear that
travels — rather than on price per teraflop.

| | Real CUDA | Power | Setup | Notes |
| --- | --- | --- | --- | --- |
| **Used RTX laptop** | yes | 60–150 W | easy | The battery is a UPS. Most compute per pound, screen included |
| **RTX A2000 (6/12 GB)** | yes | 70 W | easy | Single-slot, low-profile, no aux power, its own fan. The best card for a fixed install; the 12 GB runs `large-v3` without thinking |
| RTX A1000 / A400 | yes | ~50–70 W | easy | Newer Ada equivalents in the same envelope, less VRAM |
| **Jetson Orin Nano Super** | yes | 7–25 W | fiddly | Lowest power with real CUDA. Needs an aarch64 CUDA build of CTranslate2 |
| Tesla T4 / A2 | yes | 60–70 W | easy | 16 GB, but **passively cooled** — needs forced airflow outside a server |
| x86 mini PC (N100) | no | 15–30 W | easy | Cheapest reliable CPU-only path; `base`/`small` comfortably |
| Raspberry Pi 5 | no | 10–15 W | easy | `tiny`/`base`, no headroom |
| A phone | no | — | no | The accelerator is not reachable from this stack |

Everything above the mini PC is NVIDIA because CTranslate2 speaks CUDA and
nothing else. Intel Arc and AMD cards become options only if the transcription
engine changes — which is item 2 on the list below, not a config setting.

**Laptop for kit that moves, Jetson for kit that is bolted in.** The Jetson's
8 GB is shared between CPU and GPU, so `small` live plus `large-v3` escalation
is tighter than it looks — `int8_float16` halves both.

Used models worth searching for, if it comes to that:

- **Ex-corporate mobile workstations** — built for sustained load, cheap off
  lease, better cooling and dust tolerance than gaming machines. Dell Precision
  7550/7560 (RTX A3000 6 GB, A4000 8 GB), Lenovo ThinkPad P15 or P1 Gen 3–4
  (Quadro RTX/A-series), older Precision 7540 or ThinkPad P53 (Quadro RTX 3000).
- **Gaming laptops** — more GPU per pound, louder, cooling varies. Lenovo
  Legion 5 / 5 Pro (RTX 3060 6 GB / 3070 8 GB), ASUS TUF A15/F15, Dell G15,
  Acer Nitro 5, HP Omen 15/16, ASUS Zephyrus G14/G15.

Two things that decide it more than the model name:

- **TGP varies wildly.** The same "RTX 3060" ships between 60 W and 130 W
  depending on chassis, and a thin one can be half the speed of a thick one.
  Look it up for the specific machine.
- **Hybrid graphics on Linux.** Optimus can make CUDA hard to reach; a machine
  with a MUX switch avoids the problem. The check is
  `python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"`
  — a 0 there on a machine with an NVIDIA GPU is the Optimus setup, not the
  card.

Prices and availability move; that part is worth checking rather than taking
from here.

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

   **Worth doing before buying hardware, not after.** The engine decides which
   accelerators are even candidates: CTranslate2 is CUDA-only, so today the
   answer is NVIDIA or nothing. Choosing the card first forecloses everything
   else. The measurement above is what this looks like in practice — on the
   same CPU the two engines were close, and the interesting 3.4× came from whisper.cpp
   being able to use a GPU that faster-whisper could not address, while the
   CPU-only difference was a far less interesting 1.4×.

   **What a different engine would unlock.** `whisper.cpp` has CUDA, ROCm,
   Vulkan, SYCL and Metal backends, and the Vulkan one makes almost any modern
   GPU usable. That changes the shopping list entirely:

   | Part | Why it becomes a candidate |
   | --- | --- |
   | Intel Arc A310 / A380 | 75 W, cheap, 4–6 GB, low-profile variants exist |
   | AMD RX 6600 / 7600 | 8 GB, plentiful used, ROCm or Vulkan |
   | Intel Core Ultra NPU | Via OpenVINO — no card at all, interesting for a mini PC |

   The Arc A310 in particular is the interesting one for a trailer: single
   slot, no aux power, and enough memory for `small` or `medium` — which is
   all the live model ever needs to be when escalation is handling the hard
   lines. `whisper.cpp` already exposes an OpenVINO encode path (`-oved`), so
   the Core Ultra route needs no new engine, just a different build.

   The build also ships a `parakeet-cli`, so the Parakeet half of this item can
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

5. **Review after the net, not during it.** Everything self-supervised is in
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
