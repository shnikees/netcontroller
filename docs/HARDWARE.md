# Hardware: what to run it on, and whether to buy anything

Split out of [STATUS.md](STATUS.md) once it grew past the point where it was a
note and became an argument.

The short version, and it is genuinely the conclusion rather than a hedge:
**measure before buying, because the measurement keeps saying you do not need
to buy anything.** On synthetic audio `base` matches `medium`. On the one real
net measured so far it *beats* both `small` and `medium`, recovering more than
twice the callsigns `medium` did for a third of the compute. Either way `base`
runs comfortably on hardware you already own, and the case for spending money
on a bigger model has got weaker every time it has been measured.

## Measure, then buy

There is no deadline here, which makes the sensible order obvious. The app
already reports the number that decides it: the realtime factor on the status
strip, transcription time over audio length.

1. **Start with the machine you have.** Run a recording through `--file` with
   `base` and read `speed` off the strip. Under about 0.5× there is headroom
   for a bigger model; near 1.00× there is not. That measurement costs nothing
   and answers the question better than any table here.
2. **Remember what escalation changes** -- and check that it helps at all. The
   big model only handles the lines the fast one was unsure about, in the gaps,
   so "can it run `large-v3` on every transmission" is the wrong question. But
   on real audio neither `small` nor `medium` recovered more callsigns than
   `base`, so a second pass with a bigger model is currently a cost with no
   demonstrated benefit. Measure it on your own net before turning it on.
3. **Only then buy**, and buy for the measurement rather than the spec sheet.

`tools/bench_engines.py` is the tool for step 1. It cuts a recording into clips
with the project's own VAD, so it measures the real workload rather than
flattering an engine with one long file:

```bash
python tools/bench_engines.py --audio net.wav --roster roster.csv --repeat 5
```

## Threads, and why the app was slower than these numbers until 2026-08-26

Every measurement in this file passes `cpu_threads` explicitly -- 8 unless
stated -- and `whisper.cpp` is always given `-t 8`, so the engine comparisons
below are like for like.

The app was not. `SttWorker` built its `WhisperModel` with no `cpu_threads` at
all and took the library default, which in practice keeps about one core busy.
So anyone reading these figures and then running `app.py` got worse throughput
than the doc promised, on the same machine. The setting is now
`whisper.cpu_threads`, still defaulting to 0 -- during a live net the SDR
software and the dashboard need cores too, and a batch replay can ask for more.

Worth knowing what threads actually buy, because it is much less than the core
count suggests. On an 8-core Ryzen 7700X, one 75-minute net:

| Threads | Time |
| --- | --- |
| library default | 292 s |
| 16 | 238 s |

That is 18%, and the process was using 0.93 cores throughout -- 73 OS threads,
222 CPU-seconds over 238 s of wall clock. Beam-search decoding is sequential, so
a single transcription uses about one core however many it is handed; the
threads only help the encoder.

**Parallelism comes from processing several recordings at once, not from
threads.** Four at a time with two threads each did four 75-minute nets in 247 s
against 238 s for one, close to four times the throughput --
`tools/batch_process.py --jobs 4`. That does not help a live net, where clips
arrive one at a time, but it is the difference between a folder of 32 recordings
taking two hours and taking eight.

## Biasing is the biggest lever, and half of what it appears to give is fake

Measured 2026-08-17 over 919 transmissions from three live nets, `base`, with
three callsigns confirmed by a member of the net. "Raw" counts every roster
match; "trusted" counts what survives `hallucination.py`, which drops a
transcript once it looks like the model reciting the prompt rather than a
transmission.

| Biasing | Raw | Trusted | Discarded |
| --- | --- | --- | --- |
| none (control) | 22 | 22 | 0% |
| `initial_prompt` | 163 | 75 | 54% |
| **`hotwords`** | 420 | **203** | 52% |
| both together | 680 | 138 | **80%** |

**The two rankings disagree, and that is the whole point.** By raw count,
prompt-and-hotwords wins by a mile at 680. By trusted count it is beaten by
hotwords alone, 203 to 138. Measuring recoveries without a precision check
would have selected the *worst* of the biased configurations while appearing to
select the best.

**Stacking both mechanisms does not compound, it over-biases.** Four fifths of
that configuration's output is the prompt read back. And because the filter is
all-or-nothing -- once a transcript is the prompt coming back there is no
principled way to pick which of its callsigns were real -- clips where a
genuine callsign sat next to a fabricated one are discarded too. Piling on bias
made the model less able to hear.

**What to do with this:** `hotwords` alone is worth about 9x the control, and
`stt_worker.py` does not currently use it at all -- it passes `initial_prompt`,
which measures at 75 against hotwords' 203. That is a code change rather than a
purchase, and it is the largest single improvement measured anywhere in this
file.

Caveats worth keeping attached: one model, one roster of three callsigns, and
"trusted" is defined by a heuristic written the same day. The control
discarding nothing is reassuring -- with no bias terms there is nothing to
echo -- but it is not proof the detector is right.

## A transducer model beats Whisper on real audio, and invents nothing

Measured 2026-08-26 on 1,525 clips -- 307.8 minutes of speech from six recorded
PSRG nets, segmented by the app's own VAD. Three engines over the identical
clips: **Parakeet TDT 0.6b**, and `base` Whisper with and without the roster in
the prompt. Both run under whisper.cpp with Metal on an M-series laptop, so the
timings are comparable to each other and to nothing else in this file.

| Engine | Compute | Realtime | Silent | Roster hits (raw) | Trusted | Discarded |
| --- | --- | --- | --- | --- | --- | --- |
| **Parakeet TDT 0.6b** | 306 s | **0.017×** | 60 | 31 | **31** | **0%** |
| Whisper `base`, no bias | 334 s | 0.018× | 0 | 19 | 19 | 0% |
| Whisper `base` + prompt | 335 s | 0.018× | 0 | 734 | 156 | 79% |

**Parakeet recovers 63% more callsigns than unbiased Whisper, at the same
speed, with nothing discarded.** 31 against 19. It is the first change measured
in this project that improves recovery without buying precision from somewhere
else.

**It also stays quiet.** 60 clips produced no transcript at all, against
Whisper's zero. That is not a failure -- those are squelch tails and dead
carrier, and Whisper's habit of writing *something* for them is where a good
deal of the fabrication starts. Refusing to guess is the behaviour the whole
matcher is built around, and this model does it at the acoustic layer.

**Repetition tells the two apart more sharply than any accuracy count.**
Counting every callsign-shaped string, not just roster ones:

| Engine | Clips with a callsign | Distinct | Total | Repetition |
| --- | --- | --- | --- | --- |
| Parakeet | 278 | 181 | 311 | 1.7× |
| Whisper, no bias | 157 | 150 | 185 | 1.2× |
| Whisper + prompt | 496 | 144 | 1,065 | **7.4×** |

Whisper-plus-prompt emits seven callsigns for every distinct one it knows. It is
not hearing more stations, it is saying the same handful over and over.

### Two fabrication tests that did not work, and one that did

Whisper+prompt's three most frequent outputs were the three prompted callsigns,
followed by `KJ7RMU` and `KI7RAB` -- which look exactly like recombinations of
them (`KJ7`+`RMU`, `KI7`+`RAB`). Two obvious ways to prove that failed, and both
are worth recording so they are not tried again:

- **Licence status proves nothing.** Both are issued. So is `KI7JXM`, the third
  possible recombination. Sampling 40 random `KJ7`/`KI7`-shaped callsigns,
  **37 were issued** -- the space is 92% full, so a plausible fabrication is
  almost always somebody's real callsign. This is a real limit on
  `mine_roster.py --validate`: dropping unissued candidates removes noise, but
  surviving the check is close to no evidence at all.
- **Geography proves nothing either, on this net.** The two sit in Oregon while
  the confirmed three are Puget Sound, which looked like a discriminator until
  the obvious objection: PSRG is linked and effectively global, so an Oregon or
  Idaho check-in is entirely ordinary. `mine_roster.py` already refuses to count
  distance against a candidate for this exact reason, and that decision is
  correct. Do not add a distance filter.

What does work is **cross-engine agreement on the same clip.** An echo of the
prompt has no acoustic support, so the engines that were never told the
callsigns will not go near it. For each callsign whisper+prompt reported, ask
whether the matcher can find that callsign in Parakeet's or unbiased Whisper's
transcript of *the same clip*:

    whisper+prompt callsign extractions:  876
      found in an unprompted transcript:  110  (13%)
      no support from either engine:      766  (87%)

**87% of what prompting appears to recover has no acoustic support whatsoever.**
The clips make it plain:

| Clip | whisper+prompt | Parakeet | Whisper, no bias |
| --- | --- | --- | --- |
| `c0016` | "Welcome to KJ7RAB, KJ7JXM, KI7RMU." | "Around the" | "around the" |
| `c0008` | "With KI7RAB, KJ7RAB, KJ7JXM." | "Whiskey Seven Oscar Hotel." | "with these seven-hats, go tell." |

And where a callsign was really said, the unprompted engines get it:

| Clip | Parakeet | Whisper, no bias |
| --- | --- | --- |
| `c0003` | "This is Cammy Kilo Juliet 7, Romeo Alpha Bravo" | "This is Kevin Erojoli at 7 Romeo Alfa Bravo" |

Note that Parakeet spells it out phonetically and the normalizer collapses it to
`KJ7RAB` correctly, unchanged. A first version of this test used a regex for
already-collapsed callsigns and reported 4% support instead of 13%, because it
could not see that line -- the matcher has to do the judging, not a pattern.

**A gap this exposes in `hallucination.py`.** It catches multi-callsign echo,
which is why 79% of whisper+prompt's raw hits are discarded above. It cannot
catch a *single* fabricated callsign in an otherwise plausible sentence, because
nothing in one clip distinguishes that from a real check-in. So the 156
"trusted" figure is still an overcount. Cross-engine agreement is the only
technique measured here that addresses it, and it costs a second transcription
pass.

**What this does and does not settle.** Six recordings, one net, three confirmed
callsigns, no ground truth for how many times those stations actually
identified -- so this is relative recovery again, not recall. The corpus is
larger than anything else in this file by an order of magnitude, and the
fabrication result does not depend on knowing the roster, but Parakeet's 31
against 19 is still 31 against 19 and not a precision measurement.

**What to do with this:** Parakeet is the strongest candidate to replace Whisper
in the pipeline, and it arrives with an argument against the biasing work rather
than for it -- an engine that does not need the prompt cannot echo it. It is
also CTranslate2-free, which removes the CUDA-only constraint that shapes the
whole buying section below.

### Running it: what the integration costs

`parakeet_worker.py` drives `parakeet-cli`, selected with
`whisper.engine: parakeet` or `--engine parakeet`. Off by default; see
docs/STATUS.md for why.

There is no server mode and no Python binding, so each clip is one subprocess:

| | Wall clock |
| --- | --- |
| one clip, cold spawn | 1.35 s |
| 20 clips, one spawn | 3.18 s |

which is about **1.25 s of model loading per clip against 0.1 s of inference**.
Wasteful, and still fast enough -- a 10-second transmission costs roughly 1.35 s
end to end, comparable to `base` Whisper on the same laptop and far inside
realtime. Measured per clip through the adapter: 0.58 s for a 0.7 s clip, 2.13 s
for a 54 s one.

Two things do not transfer, and both matter before switching a working setup:

- **Confidence is a different scale.** Mean per-token probability, against
  Whisper's duration-weighted `exp(avg_logprob)`. Both 0-1, both monotonic,
  neither calibrated, and *not* comparable -- so escalation thresholds and
  anything `tools/calibrate.py` derived from Whisper transcripts have to be
  re-derived.
- **`no_speech_prob` is binary.** The engine returns tokens or nothing, so this
  is 0.0 or 1.0. That is the honest report of what it can distinguish.

One concrete case from the end-to-end check, on the clip that opens the
2026-08-16 net:

| Engine | Transcript | Matched |
| --- | --- | --- |
| **Parakeet** | "Good afternoon. This is Cammy Kilo Juliet 7, Romeo Alfa Brasso" | **`KJ7RAB`** |
| Whisper `base` + prompt | "Good afternoon, this is Kevin, KI7RMU Alpha Bravo" | `KI7RMU` |

Same audio, and the biased engine put the wrong station on the log. Note also
that it named a *prompted* callsign, so nothing downstream could have caught it:
`hallucination.py` sees one plausible callsign in one plausible sentence, which
is exactly the case it cannot judge.

## Measured on real audio, which inverts the result below

Everything in the next section is synthetic speech. On 2026-08-16 the same
question was put to a real repeater: a 30-minute slice of a live net, 105
transmissions, 24.3 minutes of speech, segmented by the app's own VAD. Three
callsigns on that net are confirmed by a member of it, so "recovered" counts
transmissions where the matcher returned one of those three. All three were in
the prompt, as they would be in service.

| Model | Compute | Realtime | Callsigns recovered | Near-misses refused |
| --- | --- | --- | --- | --- |
| **`base`** | 200 s | 0.137× | **16** | 11 |
| `small` | 316 s | 0.217× | 11 | 5 |
| `medium` | 686 s | 0.470× | 7 | 6 |

**Recovery falls as the model grows, and the fall is steep** -- `medium`
recovers under half what `base` does for three and a half times the compute.
That is the opposite of the synthetic result below, where `medium` scored
full marks.

The likely mechanism is the one already suspected of `small`: a larger model is
a more confident language model, and a callsign is not language. Given noisy
audio and an unusual string, `base` writes something literal and
callsign-shaped, while `medium` writes fluent English -- and fluent English is
where the callsign goes to die. That the effect scales with model size is
consistent with it being a property of the language model rather than of the
acoustics.

**What this does and does not settle.** One net, one slice, three callsigns, no
ground truth for how many times those stations actually identified -- so this
measures *relative* recovery, not recall. It is nowhere near enough to conclude
that bigger models are worse in general, and `medium` may well transcribe the
*conversation* better while losing the callsign. But it is real audio, and it
is enough to stop the synthetic table below being used to justify buying
hardware to run a bigger model.

**Consequences worth acting on:**

- The advice elsewhere to escalate to `medium` rather than `small` is not
  supported by this. On real audio `base` beat both.
- A GPU bought to run `large-v3` live would, on this evidence, buy worse
  callsign recovery. Re-run this measurement before spending anything.
- The roster prompt matters more than the model -- and `hotwords` matters more
  than the prompt. See the biasing table above, which measures both against a
  control and against hallucination.

## Measured on synthetic speech: engines and model sizes

**Read the real-audio section above first -- these numbers point the other
way.** 18 synthetic event-net transmissions (86 s of speech from
`tools/make_test_audio.py`), cut into clips by the VAD, beam 5, roster prompt.
Apple M1 Pro, 10 core, median of five runs. `ok` counts transmissions where the
project's own matcher recovered the correct roster callsign.

| Model | Engine | Device | Compute | Realtime | Callsigns |
| --- | --- | --- | --- | --- | --- |
| `base` | faster-whisper int8 | CPU | 11.2 s | 0.130× | 17/18 |
| `base` | whisper.cpp fp16 | CPU | 8.2 s | 0.095× | **18/18** |
| `base` | whisper.cpp fp16 | GPU (Metal) | **3.3 s** | **0.039×** | **18/18** |
| `small` | faster-whisper int8 | CPU | 18.3 s | 0.213× | 12/18 |
| `small` | whisper.cpp fp16 | CPU | 19.3 s | 0.224× | 14/18 |
| `small` | whisper.cpp fp16 | GPU | 5.6 s | 0.065× | 13/18 |
| `medium` | faster-whisper int8 | CPU | 48.3 s | 0.561× | 18/18 |
| `medium` | whisper.cpp fp16 | CPU | 41.4 s | 0.481× | 18/18 |
| `medium` | whisper.cpp fp16 | GPU | 13.0 s | 0.151× | 18/18 |

Callsign figures are after the normalizer fixes described below; the timings
are from the median-of-five runs taken before them, since fixing the matcher
cannot change how long the model takes.

### `base` is as accurate as `medium`, at a twelfth of the cost

That is the headline, and it only became true after four normalizer bugs were
fixed. Before the fix `base` scored 15/18 and `medium` 18/18, which read as a
straightforward argument for a bigger model and therefore for a bigger machine.
It was not. Four of the eight losses never reached the model at all:

| Whisper wrote | Extracted | The bug |
| --- | --- | --- |
| "Kilo Delta **9er** Mike November Oscar" | *nothing* | `niner` was in the vocabulary, `9er` was not |
| "Kilo Delta **9 or** Mike November Oscar" | `KD9` | the orphaned "or" ended the run, discarding the suffix |
| "Victor Echo **III** Zulu Quebec Romeo" | *nothing* | Roman numeral, unhandled |
| "Alpha**4PQ**" | *nothing* | phonetic welded to digits and letters, unsplit |

Fixing those took `base` from 15/18 to 17–18/18. **The model was never the
bottleneck; the text handling after it was.** The lesson generalises past this
one bug: before concluding that accuracy needs more compute, check whether the
transcript already contains the answer.

### `small` is worse than `base`, consistently

12–14/18 against `base`'s 17–18, while costing roughly twice the compute. All
three engine configurations agree, so it is the model rather than the plumbing.
The plausible reason is that a mid-sized model is confident enough to "correct"
spelled phonetics toward ordinary English — "papa quebec" became "pop
equipment" — which is precisely the wrong instinct for this input, and a
failure `base` is too literal to make.

**This matters for a default:** `escalation.model_size` is currently `small`,
which on this evidence is the worst available choice. A real net has now
reproduced the `small` result -- and gone further, putting `medium` below it
too. Escalating to a bigger model is not currently supported by any real-audio
measurement.

### Nothing produced a wrong callsign

Zero, in every configuration and at every model size. The losses were all lines
left unmatched. The prefer-unmatched bias lives in the matcher rather than in
Whisper, so it survives an engine change — which is what makes changing engines
thinkable at all.

### Two cautions

The audio is TTS. It enunciates better than a handheld into a repeater, but it
also runs phonetic words together without the beats a human puts between them,
so it is not simply "easier" — only the *ranking* here is worth much.

And the roster prompt matters more than the engine did: dropping it cost
`base` 15→12 on faster-whisper and 14→10 on whisper.cpp. Note that
whisper.cpp's `-mc 0` looks like the analogue of
`condition_on_previous_text=False` and **silently discards the initial prompt
as well** — it looks like a fair comparison and is not.

## What a different engine would unlock

CTranslate2 speaks CUDA and nothing else, which is the only reason the buying
list below is all NVIDIA. The benchmark shows what that costs: on the same CPU
the two engines were within about 1.4× of each other, and the entire 3.4× came
from whisper.cpp being able to address a GPU that faster-whisper could not see.
A tuning difference does not justify changing engines; a capability difference
might.

`whisper.cpp` has CUDA, ROCm, Vulkan, SYCL and Metal backends, and the Vulkan
one makes almost any modern GPU usable.

### Integrated graphics

whisper.cpp 1.8.3 added proper iGPU support over Vulkan, reporting **3–4×
better realtime factor than CPU-only** on a Radeon 680M and a Core Ultra 7
155H. (Phoronix's "12×" headline multiplies that by the CPU baseline; the
figure to use is 3–4×.) That is the same order as the 2.4–3.4× measured here on
Metal, which is a reasonable sanity check — and both test parts are now two
generations old.

| Part | Why it becomes a candidate |
| --- | --- |
| Intel Arc A310 / A380 | 75 W, cheap, 4–6 GB, low-profile variants exist |
| AMD RX 6600 / 7600 | 8 GB, plentiful used, ROCm or Vulkan |
| Radeon 780M / 890M iGPU | No card at all — a $520–610 mini PC gets an accelerator |
| Intel Arc 140V / 140T iGPU | Same, via Lunar Lake or Arrow Lake |
| Intel Core Ultra NPU | Via OpenVINO; whisper.cpp already exposes an `-oved` encode path |
| Ryzen AI Max+ 395 "Strix Halo" | Radeon 8060S, up to 128 GB unified with 96 GB assignable as VRAM |

The mini PC line is the interesting one. The table further down lists "x86 mini
PC, no CUDA" as the cheap CPU-only tier — but with a Vulkan engine a Beelink
SER8 or Minisforum UM880 (Radeon 780M, 32 GB) at **$520–610** has an
accelerator after all. Strix Halo is the technically impressive option and
starts around **$2,000**, which buys capability this workload cannot consume:
`base` on a GPU already runs at 0.039× realtime, roughly 25× faster than
required.

### Used AMD laptops with discrete GPUs

**Do not plan on ROCm** — official support for mobile RDNA2 is effectively
absent. Vulkan is the path, and it is vendor-neutral.

| Model | GPU | VRAM | Notes |
| --- | --- | --- | --- |
| **ASUS ROG Strix G15 Advantage (G513QY)** | RX 6800M | **12 GB** | Most VRAM in an all-AMD laptop; 2021 vintage, so depreciated hard |
| ASUS ROG Zephyrus G14 GA402RK | RX 6800S | 8 GB | Same era, far more portable |
| Dell G15 5525 | RX 6700M | 10 GB | Common and cheap, heavy and loud |
| ASUS TUF A15 / A17 (2021–22) | RX 6600M | 8 GB | The budget floor |
| HP Omen 16 | RX 6600M / 7600M XT | 8 GB | The 7600M XT is RDNA3, newer drivers |

Current used prices are not recorded here because they could not be verified —
searches returned 2021 launch pricing and nothing reliable since. Check **sold**
eBay listings for "G513QY", "GA402RK", "6800M" rather than trusting a number
from this file.

## What to run it on

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
| AMD mini PC (780M) | not CUDA | 35–65 W | easy | Same price class, but a Vulkan engine turns the iGPU into an accelerator |
| Raspberry Pi 5 | no | 10–15 W | easy | `tiny`/`base`, no headroom |
| A phone | no | — | no | The accelerator is not reachable from this stack |

**Laptop for kit that moves, Jetson for kit that is bolted in.** The Jetson's
8 GB is shared between CPU and GPU, so `small` live plus `large-v3` escalation
is tighter than it looks — `int8_float16` halves both.

Used NVIDIA models worth searching for:

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

## Where the money actually goes

Cheapest first, with what each tier buys given the measurements above:

| Spend | What it buys |
| --- | --- |
| **Nothing** | `base` on the existing machine. Full accuracy on the synthetic net, and the best of the three models on the real one — start here, and note that a real net has now been asked and did not say otherwise |
| ~$520–610 | AMD mini PC (780M). Quiet, low power, and an accelerator if the engine changes |
| ~$700–1,000 | Used RTX or AMD gaming laptop. Adds a built-in UPS, which in a trailer on a generator is worth more than the speed |
| ~$1,000+ | RTX A2000 12 GB in a box that already exists, at 70 W |
| ~$2,000+ | Strix Halo, Jetson, or new hardware. Capability beyond what this workload consumes |

**A GPU is a convenience here, not a requirement.** What matters is whether the
live model keeps up while escalation handles the hard lines behind it — and on
current evidence `base` keeps up on almost anything, provided the normalizer is
not throwing the answers away.

## Sources

- [Phoronix — whisper.cpp 1.8.3 iGPU support](https://www.phoronix.com/news/Whisper-cpp-1.8.3-12x-Perf)
- [AMD Strix Halo laptop list](https://www.ultrabookreview.com/70442-amd-strix-halo-laptops/)
- [AMD Ryzen AI Max+ 395](https://www.amd.com/en/blogs/2025/amd-ryzen-ai-max-395-processor-breakthrough-ai-.html)
- [Intel Lunar Lake laptops](https://www.ultrabookreview.com/69679-intel-lunar-lake-laptops/)
- [Intel Panther Lake laptops](https://www.ultrabookreview.com/74624-intel-panther-lake-laptops/)
- [Radeon 890M vs Arc 140V](https://videocardz.com/newz/amd-radeon-890m-rdna3-5-and-intel-arc-140v-xe2-integrated-graphics-compared-in-geekbench-test)
- [Beelink SER8](https://www.newegg.com/beelink-ser8-8845hs-amd-ryzen-7-8845hs/p/2SW-0012-001W3)
- [ASUS ROG Strix G15 Advantage review](https://www.tomshardware.com/reviews/asus-rog-strix-g15-advantage-edition-rx-6800m)
