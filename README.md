# Ham Radio Net Speech-to-Text

Live transcription for an amateur radio net. Audio comes in from an SDR, each
transmission is transcribed, the speaker is matched against a roster of known
callsigns, and everything lands in a browser dashboard that net control can
read from across the table.

Fully offline — no cloud services, nothing leaves the machine. Whisper runs
locally.

```
SDR++ / GQRX ──▶ loopback sink ──▶ capture ──▶ VAD ──▶ Whisper ──▶ roster match ──▶ dashboard
```

![The dashboard during a net: matched stations in green with operator names, an
unmatched transmission flagged in amber showing the callsign that was heard, and
a roster sidebar doubling as a check-in list](docs/images/dashboard.png)

Matched stations show in green with the operator's name; the roster sidebar
lights up as stations check in. The amber line is the interesting case — an
off-roster visitor, flagged rather than force-matched to the nearest roster
entry, with the callsign the app heard shown so net control can resolve it by
ear. (Screenshot uses the example roster and synthesized audio from
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

The WAV must be 16-bit PCM at 16 kHz (or a multiple: 48 kHz works). The
dashboard populates as if the audio were live. If transmissions are being split
in half, raise `vad.silence_ms`; if two stations are being merged into one clip,
lower it.

## Feeding audio in from an SDR

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

| Model | Laptop CPU | Pi 4 | Pi 5 | Accuracy on phonetics |
| --- | --- | --- | --- | --- |
| `tiny` | ~0.2 s | ~2 s | ~1 s | Rough; workable with a roster to correct against |
| `base` | ~0.4 s | ~4 s | ~2 s | Good default |
| `small` | ~1.5 s | too slow | ~6 s | Noticeably better on weak signals |
| `medium` | ~4 s | no | no | Only if you have a GPU |

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

### Tuning it for your net

`test_callsign_match.py` has a "Regressions from real transcripts" section
holding verbatim Whisper output. When your net turns up a new mis-transcription:

1. Add the exact string as a test case.
2. Add the missing spelling to `PHONETIC_MAP` / `AMBIGUOUS_DIGIT_MAP`.
3. Run `pytest` and confirm nothing else broke.

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

52 tests, all offline, no audio hardware needed. `test_callsign_match.py`
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
- Single audio input. Two repeaters would need two instances on different ports.
- The confidence figure comes from Whisper's `avg_logprob`. It is a useful
  relative cue and not a calibrated probability.
- Transcripts live in memory for the session; export before you close it.
