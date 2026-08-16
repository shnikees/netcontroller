# Ham Radio Net Speech-to-Text

Live transcription for an amateur radio net. Audio comes in from an SDR, a
radio's line output, or a microphone; each transmission is transcribed, the
speaker is matched against a roster of known callsigns, and everything lands in
a browser dashboard that net control can read from across the table.

Built for a **high-traffic event or race net** — the kind run from a trailer
with several people talking at once, where the question is *"what was that last
transmission?"* and the answer has to be on screen rather than in somebody's
memory. On an event net the callsign is also a *location*: operators are posted
around the course, so knowing who transmitted is knowing where it came from —
which is why this app would rather log a line as unmatched than attach a
plausible wrong callsign to it. It works for an orderly weekly check-in too,
but that is the easy case.

Fully offline — no cloud services, nothing leaves the machine. Whisper runs
locally.

```
SDR / radio ──▶ loopback, line in, or mic ──▶ capture ──▶ VAD ──▶ Whisper ──▶ roster match ──▶ dashboard
```

![The dashboard during an event net: each line shows the callsign, the
position that station is posted at, and the operator's name, with traffic
declarations badged, lines re-transcribed by the second pass marked, and a
roster sidebar acting as a who-is-where board](docs/images/dashboard.png)

<sub>Deliberately an unflattering run: `tiny` live with `base` on escalation, so
the **2nd pass** marks are visible and a couple of lines are left as `tiny`
produced them. `base` on its own transcribes this recording cleanly — see
[docs/HARDWARE.md](docs/HARDWARE.md).</sub>

Each line carries the callsign, **where that station is posted**, and the
operator's name. On an event net the position is the actionable half: "need
medical at my location" is only useful once the line says Mile 8. The sidebar
doubles as a who-is-where board and lights up as stations are heard from.

Lines that declared traffic are badged, and the stations holding it are marked
in the sidebar — the header count is also a filter. Stations the roster cannot
match are flagged amber rather than attached to the nearest plausible callsign — a wrong callsign is a wrong location. Clicking one
sets the right station, and the app
[learns the correction](docs/MATCHING.md#corrections-and-what-the-app-learns-from-them) for
next time. (Screenshot uses the example roster and synthesised audio from
`tools/make_test_audio.py`.)

## Status

Everything downstream of the audio device works and is tested offline:
segmentation, transcription, callsign matching, corrections and learning, voice
suggestions, multiple receivers, buffering, crash-safe transcripts, the
watchdog, export, the container image.

**No part of it has run against a real radio yet**, and every tuning constant
is a reasoned guess rather than a measurement. [docs/STATUS.md](docs/STATUS.md)
is the honest account of what is proven, what is not, and what to do next;
[docs/FIELD-BRINGUP.md](docs/FIELD-BRINGUP.md) is the checklist for the first
session at the hardware.

## What it does

- **Listens** to an SDR loopback, a radio's line output, a microphone, or a
  linked system over the internet (EchoLink/AllStar) — one
  receiver or several, each with its own level, VAD settings and health.
- **Transcribes** each transmission with Whisper, conditioning the audio and
  biasing decoding toward the stations most likely to speak.
- **Identifies** the station: phonetic normalisation, fuzzy roster matching,
  aliases learned from corrections, and voice recognition for transmissions
  that carry no callsign at all. It refuses to guess rather than log a wrong
  callsign.
- **Shows** the transcript with each station's *position* on the course, marks
  traffic and lets you clear it, and reports its own health continuously.
- **Keeps** everything: transcripts written as the net runs and survive a power
  cut, a session that can be resumed, and exports for the net report.
- **Learns** between events — aliases, voices, attendance and thresholds all
  improve from nets nobody was watching.

## Documentation

The README is deliberately short. Each of these is the full account of one
thing:

| | |
| --- | --- |
| [docs/AUDIO-INPUT.md](docs/AUDIO-INPUT.md) | What to plug in — SDR loopback, line in, or mic — and running several receivers at once |
| [docs/ACCURACY.md](docs/ACCURACY.md) | Choosing a model, the four things that buy accuracy, and tuning against your own recordings |
| [docs/MATCHING.md](docs/MATCHING.md) | The roster, how a spoken callsign becomes a match, corrections, and voice identification |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | Dashboard, settings panel, transcripts on disk, watchdog and alerting, config reference, container |
| [docs/FIELD-BRINGUP.md](docs/FIELD-BRINGUP.md) | First run against real hardware, in order, with the failure modes at each step |
| [docs/HARDWARE.md](docs/HARDWARE.md) | Engine and model-size benchmarks, and whether anything needs buying (usually not) |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How the pieces fit, the threading model, and why things are the way they are |
| [docs/TESTING.md](docs/TESTING.md) | Test suites, generating audio without an SDR, and adding a regression |
| [docs/STATUS.md](docs/STATUS.md) | What is proven, what is guessed, and the work worth doing next |

Deployment files live in [deploy/](deploy/) (a systemd unit) and at the repo
root (`Containerfile`, `docker-compose.yml`).

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
cp config.yaml.example config.yaml && cp roster.example.csv roster.csv
```

Edit `roster.csv` with the stations you expect, then find the audio device
SDR++/GQRX is feeding:

```bash
python app.py --list-devices
```

Put that device name (a substring is enough) in `config.yaml` under
`audio.device`, and run:

```bash
python app.py
```

Open <http://localhost:8080>. The first run downloads the Whisper model; after
that it works with no network at all.

## Where to go next

Three paths, depending on what you have in front of you:

- **No radio yet** — [docs/ACCURACY.md](docs/ACCURACY.md) covers replaying a
  recording through `--file`, which exercises everything except audio capture.
- **A radio and an afternoon** — [docs/FIELD-BRINGUP.md](docs/FIELD-BRINGUP.md)
  starts at getting the receiver producing audio and ends at a real net.
- **Wondering what to run it on** — [docs/HARDWARE.md](docs/HARDWARE.md). The
  short answer is that the machine you already have is probably enough.

Before the first real net, the two things worth setting deliberately are the
**roster** ([docs/MATCHING.md](docs/MATCHING.md)) and the **squelch on the
receiver** ([docs/FIELD-BRINGUP.md](docs/FIELD-BRINGUP.md)) — an open squelch
is the single most common way to get a stream of junk lines.

## Tests

```bash
.venv/bin/python -m pytest
```

428 tests, all offline, no audio hardware needed — CI runs them on every push
across Python 3.11–3.13, plus a job with the optional libraries removed so the
Raspberry Pi fallback paths are exercised too. `test_callsign_match.py`
covers the normalizer and matcher, including verbatim Whisper output;
`test_vad_segmenter.py` pins the clip boundaries with scripted speech patterns.

See [docs/TESTING.md](docs/TESTING.md) for the workflow when a net turns up a
mis-transcription the matcher does not handle — it is a two-line change plus a
test.

## What to run it on

The short version, with the benchmarks, specific machines and the buying
argument in [docs/HARDWARE.md](docs/HARDWARE.md). **Start by assuming you need
nothing** — on the synthetic test net `base` matches `medium` on callsign
recovery, and runs at a fraction of realtime on an ordinary CPU.

| | Runs | Notes |
| --- | --- | --- |
| Used RTX laptop | anything, on CUDA | The battery is a UPS, which matters more in a trailer than the extra speed |
| Jetson Orin Nano | anything, on CUDA | 7–25 W, for a permanent install. Needs an aarch64 CUDA build of CTranslate2 |
| x86 mini PC | `base`, `small` on CPU | Cheapest reliable path with no GPU |
| Raspberry Pi 5 | `tiny`, `base` | Works, no headroom. `small` only on a quiet net |
| Raspberry Pi 4 | `tiny`, `base` | Do not target `medium` |

INT8 is already the default on CPU (`whisper.compute_type: null`).

If SDR++/GQRX runs on the same box, remember it has real CPU cost of its own —
watch the status strip during an actual net before assuming one machine covers
both. And escalation lowers the bar a long way: the big model only handles the
lines the fast one was unsure about, so `base` live with `large-v3` on
escalation is a comfortable target for hardware that could not run `large-v3`
on every transmission.

## License

GPL-3.0. Copyright (C) 2026 Michelle Michaels. See [LICENSE](LICENSE).

The same license as GQRX, GNU Radio, and fldigi, so this composes with the rest
of the SDR stack it sits alongside. If you modify it and distribute your
version, your changes have to be available under the GPL too.

## Limitations

See [docs/STATUS.md](docs/STATUS.md) for the full account, including which
settings are still guesses. In short:

- Voice only (FM/SSB). No CW, no digital modes.
- One transmission is assumed to be one speaker, which holds for a half-duplex
  net and not much else.
- The confidence figure comes from Whisper's `avg_logprob`. It is a useful
  relative cue and not a calibrated probability.
- One Whisper model is shared across receivers, so two busy frequencies
  serialise rather than transcribe in parallel.
- Traffic is detected from what was said; whether it was *passed* is your
  click, not something the app infers from later transmissions.
- Voice suggestions and attendance both need history, so the first event with
  a new roster offers neither. Both improve every time the app runs.
- Settings changed from the dashboard apply to the running process; saving to
  `config.yaml` is a separate click, so an unsaved change is gone at the next
  restart.
