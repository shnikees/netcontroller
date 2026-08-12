# Ham Radio Net Speech-to-Text

Live transcription for an amateur radio net. Audio comes in from an SDR, a
radio's line output, or a microphone; each
transmission is transcribed, the speaker is matched against a roster of known
callsigns, and everything lands in a browser dashboard that net control can
read from across the table.

Fully offline — no cloud services, nothing leaves the machine. Whisper runs
locally.

```
SDR / radio ──▶ loopback, line in, or mic ──▶ capture ──▶ VAD ──▶ Whisper ──▶ roster match ──▶ dashboard
```

![The dashboard during a net: matched stations in green with operator names, an
unmatched transmission flagged in amber showing the callsign that was heard, and
a roster sidebar doubling as a check-in list](docs/images/dashboard.png)

Matched stations show in green with the operator's name; the roster sidebar
lights up as stations check in. The amber line is the interesting case — an
off-roster visitor, flagged rather than force-matched to the nearest roster
entry, with the callsign the app heard shown so net control can resolve it by
ear. Clicking that cell sets the right station, and the app
[learns the correction](#corrections-and-what-the-app-learns-from-them) for next
time. (Screenshot uses the example roster and synthesized audio from
`tools/make_test_audio.py`.)

## Status

Working end to end on recorded and synthesized audio. **The live SDR capture
path has not yet run against real hardware** — see
[docs/FIELD-BRINGUP.md](docs/FIELD-BRINGUP.md) for the bring-up checklist and
the list of specifically unverified pieces.

Everything downstream of the audio device — segmentation, transcription,
callsign matching, dashboard, export, container image — is tested and works.

## Documentation

- [docs/FIELD-BRINGUP.md](docs/FIELD-BRINGUP.md) — first-time setup against
  real hardware, in order, with the failure modes at each step
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how the pieces fit, the
  threading model, and why things are the way they are
- [docs/TESTING.md](docs/TESTING.md) — test suites, generating test audio
  without an SDR, and how to add a regression when a net surfaces a new
  mis-transcription

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
cp config.yaml.example config.yaml && cp roster.example.csv roster.csv
```

Edit `roster.csv` with the stations you expect (`callsign,name`), then find the
audio device SDR++/GQRX is feeding:

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

## Before you go live: replay a recording

Do this first. It lets you tune the VAD and see how the matcher handles your
net's actual callsigns without a live net waiting on you.

```bash
python app.py --file recorded-net.wav --model tiny
```

No recording handy? Generate one — this needs no radio at all:

```bash
python tools/make_test_audio.py && python app.py --file test-net.wav --model tiny
```

The WAV must be 16-bit PCM; any sample rate works (44.1 kHz from a phone
recording is resampled automatically). The
dashboard populates as if the audio were live. If transmissions are being split
in half, raise `vad.silence_ms`; if two stations are being merged into one clip,
lower it.

## Choosing an audio input

Any input works — the app does not care where the audio comes from:

| Source | What it is | Trade-off |
| --- | --- | --- |
| **SDR loopback** | A monitor source fed by SDR++/GQRX | Best quality; audio never leaves the machine. Most setup. |
| **Line in** | USB sound card or line input, fed from a radio's speaker or headphone jack | The practical choice for a handheld or mobile rig. Needs a cable and a level check. |
| **Microphone** | Pointed at the radio's speaker | Fastest to set up, and works. The room is in the recording too, so expect more junk lines. |

```bash
python app.py --list-devices
```

That labels which devices look like a loopback, a line in, or a mic. Put the
name (a substring is enough) or the index in `audio.device`.

Two settings matter for line in and mic, and not at all for a loopback:

- **`audio.channel`** — `mix`, `left`, `right`, or an index. If a stereo input
  carries the radio on one side only, mixing in the dead channel halves your
  level.
- **`audio.gain`** — a line output into a mic input is usually far too quiet
  (try 4–10); a speaker output into a line input is usually hot (try 0.3–0.5).
  The dashboard warns when the level is too low to work with, so set this by
  watching rather than guessing.

Sample rate is handled for you. Mics and USB sound cards typically run at
44.1 kHz, which is resampled to the 16 kHz Whisper wants (via `soxr`, with a
built-in fallback if it is not installed).

### Multiple receivers

A net often runs on more than one frequency at once — the repeater plus a
simplex staging channel. List them under `sources:` and both land in one log:

```yaml
sources:
  - name: Repeater
    device: repeater_sink.monitor
    priority: 10          # served first when the transcriber is behind
  - name: Simplex
    device: simplex_sink.monitor
    channel: left
    gain: 4.0             # weaker signal, brought up
    aggressiveness: 1     # gentler VAD than the repeater needs
    silence_ms: 1200
```

**Receivers are not interchangeable, so weight them.** The repeater carries the
net and arrives strong; a staging channel may be weak and slow. Anything under
`vad:` can be overridden per source (`aggressiveness`, `silence_ms`,
`min_clip_ms`, `preroll_ms`, `trigger_ratio`), and `gain` compensates for level
differences at the input. `priority` decides who goes first when there is a
backlog — put the frequency the net actually runs on above the side traffic, so
a slow moment delays the staging channel rather than the main log.

Every line is tagged with the receiver that heard it, and the sidebar grows a
per-source health panel — click a source to filter the log to it, which
composes with the callsign filter. Sources are captured independently, so a
receiver that drops does not take the net down with it; the banner names the
one that broke.

They share a single Whisper model on purpose. It is the memory-hungry part,
and two on a Pi would thrash rather than go faster; serialising costs a little
latency on a busy net and nothing at all on a quiet one. The single `audio:`
block still works and stays the common case.

To tune a two-receiver setup without radios, give each source a `file:` instead
of a `device:`.

### Feeding audio in from an SDR

Create a null sink and point SDR++/GQRX's output at it, then have this app read
the sink's monitor source:

```bash
pactl load-module module-null-sink sink_name=net_sink sink_properties=device.description=NetAudio
```

Set the SDR app's output device to `NetAudio`, and set `audio.device:
net_sink.monitor` in `config.yaml`. Use `pactl list sources short` to confirm
the name.

To hear the net yourself at the same time, use a combined sink or set the SDR
app to duplicate its output — the monitor source does not consume the audio.

## Configuration

Everything lives in `config.yaml` (see `config.yaml.example` for the annotated
version). Any key can also be set by an environment variable named
`NETSTT_<SECTION>_<KEY>`, e.g. `NETSTT_WHISPER_MODEL_SIZE=small` — which is how
the container is configured.

The settings that actually matter in the field:

| Setting | Default | Why you would change it |
| --- | --- | --- |
| `vad.silence_ms` | 800 | Lower if stations run together; raise if one check-in splits into several lines. 800 ms survives the pauses in a phonetically spelled callsign. |
| `vad.min_clip_ms` | 400 | Raise if squelch tails and kerchunks are producing junk lines. |
| `vad.aggressiveness` | 3 | Lower to 1–2 if quiet stations are being missed entirely. |
| `roster.threshold` | 78 | Raise for fewer wrong matches, lower for fewer "unmatched" lines. See below. |
| `whisper.model_size` | base | See the table below. |

### Choosing a model

Rough figures for one ~5 second transmission on CPU. Measure on your own
hardware; this is a guide, not a benchmark.

| Model | Laptop CPU | Pi 4 | Pi 5 | Accuracy |
| --- | --- | --- | --- | --- |
| `tiny` | ~0.2 s | ~2 s | ~1 s | Rough. Drops words when an operator rattles a callsign off quickly; leans hard on the roster to recover |
| `base` | ~0.4 s | ~4 s | ~2 s | Good default for an unhurried net |
| `small` | ~1.5 s | too slow | ~6 s | Where to go if transcripts are wrong on fast, run-together speech |
| `medium` | ~4 s | no | no | Best accuracy; needs a GPU to be worth it |

**If the problem is fast-paced traffic, model size is the main lever.** A
station who gives their callsign as one run-on phrase is harder for a small
model than a quiet one who spells it out, and no amount of VAD tuning fixes a
word the model never heard. `small` is the first thing to try; `beam_size: 5`
(the default) is already doing what it can.

Two settings interact with fast nets and are worth knowing about:

- **`vad.silence_ms`** (default 800). On a net where stations key up on top of
  each other, an 800 ms gap may never appear, and two stations end up in one
  clip — where only the first callsign gets logged. Lower it to 400–500 for a
  fast net, and check the log for lines containing two check-ins.
- **`whisper.vocabulary`** — adding the phrases your net actually uses biases
  decoding, which helps most exactly when speech is quick and clipped.

CUDA is detected automatically (`whisper.device: auto`); set it to `cpu` or
`cuda` to force one.

## How callsign matching works

This is the part worth understanding, because it is where the app earns its
keep. Whisper does not know your net; it hears "kilo delta niner mike november
oscar" and writes something approximate. Four stages fix that:

1. **Normalize** — phonetics become letters, spoken digits become numerals,
   filler ("this is", "net control", "over") is stripped. The tables in
   `callsign_match.py` include the ways Whisper actually mangles phonetics, and
   handle words it runs together (`alfabravo` → `A B`).
2. **Extract** — pull tokens shaped like a US callsign (1–2 letters, a digit,
   1–3 letters).
3. **Match** — fuzzy-match against the roster with `rapidfuzz`.
4. **Accept or flag** — a match is accepted only if it clears
   `roster.threshold` *and* no other roster entry is within
   `roster.ambiguity_margin` of it.

That last rule is deliberate. One wrong character in a five-character callsign
scores 80, so the default threshold of 78 forgives a single Whisper slip. But
if two roster entries are equally plausible, the app flags the line
**unmatched** rather than guessing — a wrong callsign in the log is worse than
a blank one, because net control will not notice the wrong one. Unmatched lines
show the callsign the app *heard*, so it can be resolved by ear.

### Homophones

"For" is a preposition in "standing by for traffic" and a 4 in "alpha for
bravo". The normalizer converts an ambiguous word to a digit only when it sits
between two spelling tokens, which is where a callsign's digit always sits.
Ordinals are handled the same way, because Whisper reliably turns a spoken
digit between two phonetics into one ("november five delta" → "november fifth
delta").

### Corrections, and what the app learns from them

Click any callsign in the dashboard and pick the right station. Three things
happen:

1. The log line is fixed, and marked `✓ corrected` — the export keeps a note of
   what the matcher originally said, so the record still shows where it was
   wrong.
2. The correction is appended to `feedback.jsonl` (transcript, what was heard,
   what it actually was).
3. The matcher **learns the alias**, so the next time that station is mangled
   the same way, it matches on its own.

That third step is the point. If Whisper reliably hears `KJ6TUV` as `E3Z` on
your repeater, you fix it once and the app has it from then on — including
after a restart, since aliases are replayed from the log at startup.

A learned match shows `✓ learned` rather than `✓ corrected`, so you can always
tell which lines the machine got by itself and which came from an alias.

Aliases only ever point at stations on your roster, are dropped automatically if
you remove a station from `roster.csv`, and override even an "ambiguous"
refusal — an operator's word beats the fuzzy matcher's. Set
`roster.learn_aliases: false` to keep logging corrections without applying them.

`feedback.jsonl` is also your labelled dataset: each line pairs Whisper's
output with a human-confirmed callsign, which is exactly what a fine-tuning run
would need later. Back it up alongside `roster.csv`; it is gitignored because
it contains real net traffic.

### Tuning it for your net

`test_callsign_match.py` has a "Regressions from real transcripts" section
holding verbatim Whisper output. When your net turns up a new mis-transcription:

1. Add the exact string as a test case.
2. Add the missing spelling to `PHONETIC_MAP` / `AMBIGUOUS_DIGIT_MAP`.
3. Run `pytest` and confirm nothing else broke.

## Watchdog, logging, and alerting

The failure worth guarding against is not a crash — it is the pipeline that
keeps running while logging nothing, because the SDR app was closed, the
squelch stayed shut, or the sink got repointed. Forty minutes of the net go
missing and the dashboard looks fine the whole time.

So the app watches three things: whether audio frames are arriving, whether
there is any *signal* in them, and whether the machine is keeping up.

**On the dashboard** — a red or amber banner naming the problem, the status dot
changes colour, and it beeps once on the transition (toggle with the
**Alerts** button; the setting sticks). The beep matters because the operator
is usually looking at the radio, not the screen.

**In the log** — console plus a rotating file in `logs/`, with a heartbeat line
every minute so you can see the net progressing:

```
Heartbeat: ok | up 12m | 24000 frames, 18 clips, 18 transcripts | level 412 RMS | last transcribe 0.4s | 0 dropped
```

**For anything external** — `GET /api/health` returns the same data as JSON, and
**503** when the pipeline is in error, so a container healthcheck or a one-line
`curl -f` in cron can act on it.

**Keeping up** — transcription runs on its own thread, so a slow machine never
costs you audio. Clips queue in memory; if that fills, they spill to disk and
are transcribed during the next lull or after the net. Those lines are marked
**late** on the dashboard but sit in their correct place in the log, so the
exported net report still reads in transmission order. If clips are spilling
every net, the model is a size too big for the hardware and the banner says so.

**Recovery** — if the audio device drops (a USB SDR replugged mid-net), capture
reopens automatically with exponential backoff, so it costs seconds rather
than the rest of the net. For an unattended install, `deploy/net-stt.service`
adds systemd `Restart=always` on top, and the container image has a
`HEALTHCHECK` wired to the same endpoint.

Thresholds live under `health:` in the config — most usefully
`silence_after_s` (default 5 minutes), which is how long the audio can be dead
quiet before it is worth telling you about.

## Dashboard

- Matched stations in bold green; unmatched lines flagged amber with the
  callsign that was heard.
- Roster sidebar doubles as a check-in list — stations light up as they check
  in, with a count. Click one to filter the log to that station.
- **Auto-scroll** toggle for reviewing history mid-net without fighting the log.
- **Export log** writes a CSV and a text log to `export_dir`. The text log is
  also written automatically on exit, so a Ctrl-C never loses the session.

The page reconnects on its own if the app restarts.

## Tests

```bash
.venv/bin/python -m pytest
```

167 tests, all offline, no audio hardware needed. `test_callsign_match.py`
covers the normalizer and matcher, including verbatim Whisper output;
`test_vad_segmenter.py` pins the clip boundaries with scripted speech patterns.

See [docs/TESTING.md](docs/TESTING.md) for the workflow when a net turns up a
mis-transcription the matcher does not handle — it is a two-line change plus a
test.

## Running in a container

The app image contains the STT/dashboard only. Keep SDR++/GQRX on the host —
passing a USB SDR into a container is the most fragile part of this setup, and
there is no benefit to it here.

```bash
UID=$(id -u) AUDIO_DEVICE=net_sink.monitor docker compose up --build
```

The Pulse socket is bind-mounted from `$XDG_RUNTIME_DIR/pulse`, and the
container runs as a user whose UID matches the host's so the socket is
readable. Mismatched UIDs are the usual cause of "no audio" in this setup.

The Whisper model is baked into the image at build time
(`MODEL_SIZE=tiny docker compose build`), so a deployed container never needs
the network.

## Raspberry Pi notes

- Use `tiny` or `base` on a Pi 4, `small` on a Pi 5. Do not target `medium`.
- INT8 is the default on CPU already (`whisper.compute_type: null`).
- SDR++/GQRX has real CPU cost of its own. If you are running both on one Pi,
  watch headroom during an actual net — it may be worth splitting them across
  two boxes.

## License

GPL-3.0. Copyright (C) 2026 Michelle Michaels. See [LICENSE](LICENSE).

The same license as GQRX, GNU Radio, and fldigi, so this composes with the rest
of the SDR stack it sits alongside. If you modify it and distribute your
version, your changes have to be available under the GPL too.

## Limitations

- Voice only (FM/SSB). No CW, no digital modes.
- One transmission is assumed to be one speaker, which holds for a half-duplex
  net and not much else.
- The confidence figure comes from Whisper's `avg_logprob`. It is a useful
  relative cue and not a calibrated probability.
- Transcripts live in memory for the session; export before you close it.
