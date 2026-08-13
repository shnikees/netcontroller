# Hardware: what to run it on, and whether to buy anything

Split out of [STATUS.md](STATUS.md) once it grew past the point where it was a
note and became an argument.

The short version, and it is genuinely the conclusion rather than a hedge:
**measure before buying, because the measurement keeps saying you do not need
to buy anything.** On the test net below, `base` with a fixed normalizer scores
the same as `medium` — and `base` runs in real time on hardware you already
own.

## Measure, then buy

There is no deadline here, which makes the sensible order obvious. The app
already reports the number that decides it: the realtime factor on the status
strip, transcription time over audio length.

1. **Start with the machine you have.** Run a recording through `--file` with
   `base` and read `speed` off the strip. Under about 0.5× there is headroom
   for a bigger model; near 1.00× there is not. That measurement costs nothing
   and answers the question better than any table here.
2. **Remember what escalation changes.** The big model only handles the lines
   the fast one was unsure about, in the gaps — so "can it run `large-v3` on
   every transmission" is the wrong question. `base` live with `small` or
   `medium` escalating is a much cheaper target.
3. **Only then buy**, and buy for the measurement rather than the spec sheet.

`tools/bench_engines.py` is the tool for step 1. It cuts a recording into clips
with the project's own VAD, so it measures the real workload rather than
flattering an engine with one long file:

```bash
python tools/bench_engines.py --audio net.wav --roster roster.csv --repeat 5
```

## Measured: engines and model sizes

18 synthetic event-net transmissions (86 s of speech from
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
which on this evidence is the worst available choice. If a real net reproduces
this, escalation should go straight to `medium`.

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
| **Nothing** | `base` on the existing machine. On this test net that is already full accuracy — start here and stay unless a real net says otherwise |
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
