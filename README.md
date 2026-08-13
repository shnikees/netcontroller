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
declarations badged and a roster sidebar acting as a who-is-where
board](docs/images/dashboard.png)

Each line carries the callsign, **where that station is posted**, and the
operator's name. On an event net the position is the actionable half: "need
medical at my location" is only useful once the line says Mile 8. The sidebar
doubles as a who-is-where board and lights up as stations are heard from.

Lines that declared traffic are badged, and the stations holding it are marked
in the sidebar — the header count is also a filter. Stations the roster cannot
match are flagged amber rather than attached to the nearest plausible callsign — a wrong callsign is a wrong location. Clicking one
sets the right station, and the app
[learns the correction](#corrections-and-what-the-app-learns-from-them) for
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

- **Listens** to an SDR loopback, a radio's line output, or a microphone — one
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

- [docs/FIELD-BRINGUP.md](docs/FIELD-BRINGUP.md) — first-time setup against
  real hardware, in order, with the failure modes at each step
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how the pieces fit, the
  threading model, and why things are the way they are
- [docs/TESTING.md](docs/TESTING.md) — test suites, generating test audio
  without an SDR, and how to add a regression when a net surfaces a new
  mis-transcription
- [docs/STATUS.md](docs/STATUS.md) — what is proven, what is guessed, and the
  work worth doing next
- [docs/HARDWARE.md](docs/HARDWARE.md) — engine and model-size benchmarks,
  accelerator options, and whether anything needs buying (usually not)

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

## Training it on your net

Two commands, and neither needs you at the keyboard during a net.

**After each net**, from the transcripts the app already wrote:

```bash
python tools/calibrate.py            # see what the data says
python tools/calibrate.py --apply    # write it into config.yaml (keeps a backup)
```

It sets `escalation.min_confidence` from the confidence of matched versus
unmatched lines, and `voice.min_similarity` from how alike two clips of one
operator turned out to be. It refuses to suggest anything it cannot support
yet, and says what it is still short of — no number is better than a number
that looks measured and is not.

**Once, from a recording**, for the timing thresholds. Every one of them ships
as a reasoned guess about how a net sounds; ten minutes of your own traffic
replaces the guesses with measurements:

```bash
python tools/tune.py --audio net-recording.wav --roster roster.csv
```

It sweeps the VAD and split thresholds, checks your audio level, and prints a
config block. Where two candidates are indistinguishable on your recording it
says so, rather than picking one and pretending.

Replaying recordings to build the history is scriptable — `--batch` processes a
file and exits instead of serving the dashboard:

```bash
python app.py --file last-tuesday.wav --batch
```

**None of this needs an operator present.** The roster is the supervision: a
clean roster match is a labelled example, so voice profiles, attendance and
calibration data all accumulate on nets nobody was watching. Corrections make
it better and faster, but they are not what makes it work.

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

### The roster

Columns are read by name when the file has a header, so the order does not
matter and anything but `callsign` can be left out:

```csv
callsign,name,position,sources
W6ABC,Alice,Net Control,Repeater
K7XYZ,Bob,Turn 7,Repeater
N5DEF,Carol,Mile 8,Repeater;Simplex
KJ6TUV,Frank,Sweep,
```

**`position`** is where the operator is posted. On an event net this is the
point of identifying them at all — the callsign is how net control knows where
a transmission came from — so it shows on every line and in the exported log:

```
[19:04:12] N5DEF (Mile 8 / Carol): rider down, need medical at my location
```

**`sources`** is which receivers to expect them on, separated by `;` or `|`
(a comma would break the CSV). Lines starting with `#` are comments, so a
station who is away can be commented out rather than deleted.

Files without a header still work and are read as `callsign, name, sources,
position`.

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

**Each receiver gets its own tab**, and the first one listed is the default —
put the repeater there, since that is the frequency actually being monitored.
Each tab carries a health dot, so a dead receiver is visible while you are
looking at a different frequency, and an unread count, so a check-in on the
other tab does not go unnoticed. An **All** tab shows everything interleaved by
time.

Sources are captured independently, so a receiver that drops does not take the
net down with it; the banner names the one that broke.

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

## Settings, without leaving the net

The dashboard has a **Settings** panel for the handful of things somebody
reaches for mid-event, when walking to a terminal costs transmissions:

| Group | What it covers |
| --- | --- |
| Transcription | Model size, beam size, second-pass escalation |
| Segmentation | Pause that ends a transmission, squelch rejection, shortest transmission, gap between two stations |
| Matching | Match confidence, ambiguity margin |
| Dashboard | Traffic marking and clearing, voice suggestions |
| Audio | Input level, one control per receiver |

Changes **apply immediately and in memory**. Writing them back to
`config.yaml` is a separate button, because a change made during a net is
often a change for tonight only, and quietly rewriting the config would make
every experiment permanent. Saving patches the values in place, so the
comments explaining what each threshold is for survive, and it leaves a
`.bak` beside the file.

Switching the live model is the only setting that costs anything, and it costs
latency rather than audio: the swap happens between clips while capture keeps
buffering. Verified mid-net on a six-transmission recording — the model changed
partway and all six lines still landed.

Bounds live with each setting rather than in the browser, so a hand-made
request cannot put the pipeline somewhere it will not come back from.

Everything else stays in the file. Device names, ports, buffer depths and
paths are set once at install, and a UI for them would be a text editor with
extra steps.

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
| `split.min_gap_ms` | 500 | Raise if a station naming another is logged as two check-ins; lower if back-to-back stations are merged into one line. |
| `audio.gain` | 1.0 | Line in and mic only — a level that is too low costs accuracy. Per source when you have several. |
| `transcripts.live` | true | Leave on. It is what makes a crash cost nothing. |

Everything under `vad:` can also be set **per source**, which is what you want
when a strong repeater and a weak simplex receiver share one app — see
[Multiple receivers](#multiple-receivers).

### Choosing a model

Rough figures for one ~5 second transmission on CPU. Measure on your own
hardware; this is a guide, not a benchmark.

| Model | Laptop CPU | Pi 4 | Pi 5 | Accuracy |
| --- | --- | --- | --- | --- |
| `tiny` | ~0.2 s | ~2 s | ~1 s | Rough. Drops words when an operator rattles a callsign off quickly; leans hard on the roster to recover |
| `base` | ~0.4 s | ~4 s | ~2 s | Good default for an unhurried net |
| `small` | ~1.5 s | too slow | ~6 s | **Measured worse than `base`** at callsign recovery — see below |
| `medium` | ~4 s | no | no | Best accuracy; needs a GPU to be worth it |

On the synthetic test net in [docs/HARDWARE.md](docs/HARDWARE.md), `small`
recovered fewer callsigns than `base` in every engine configuration, seemingly
because a mid-sized model is confident enough to "correct" spelled phonetics
into ordinary English. That is one synthetic recording rather than a verdict,
but check it against your own audio before reaching for a bigger model.

**Model size is a lever for fast traffic, but try it last rather than first.**
A station who rattles a callsign off as one run-on phrase is genuinely harder
than one who spells it out, and no amount of VAD tuning fixes a word the model
never heard. But on the one net measured so far, a bigger model was not the
fix: `base` matched `medium` on callsign recovery once the normalizer stopped
discarding callsigns, and `small` scored *worse* than either. Check the
transcript before the model — if the callsign is legible in `raw_text` and the
line still came back unmatched, the problem is downstream of Whisper and a
bigger model will not touch it. `beam_size: 5` (the default) is already doing
what it can.

Two settings interact with fast nets and are worth knowing about:

- **`vad.silence_ms`** (default 800). On a net where stations key up on top of
  each other, an 800 ms gap may never appear, and two stations end up in one
  clip — where only the first callsign gets logged. Lower it to 400–500 for a
  fast net, and check the log for lines containing two check-ins.
- **`whisper.vocabulary`** — adding the phrases your net actually uses biases
  decoding, which helps most exactly when speech is quick and clipped.

### Accuracy without losing speed

Four things, in the order they cost you time (the first three cost none):

**1. The prompt actually fits.** Whisper's prompt window is 224 tokens and
anything past it is silently discarded — a written callsign costs about four
tokens, so roughly 48 fit *in total*. A roster of 50+ was overflowing and being
truncated at an arbitrary point, which is worse than a short prompt chosen on
purpose. Now the prompt carries the phonetic alphabet (26 words that bias every
spelled callsign, rather than seven tokens per station), the net vocabulary,
and as many callsigns as the budget allows — ordered by who is most likely to
speak next.

**1a. Who actually turns up.** Roster order says nothing about who is likely
to speak next. The transcripts already record who was on the last few nets, so
that is used to order the prompt instead — weighted by how often a station
appears and how recently, since crews change. It needs no setup: it reads the
sessions in `transcripts/` at startup and says what it found.

A callsign that shows up in the logs but not the roster is *reported*, never
adopted. Promoting a mis-transcription to a station would bias decoding toward
its own mistake.

**2. Per-frequency rosters.** With 20–50 stations on the repeater and another
20–50 on simplex, no single prompt can cover both. The optional third column in
`roster.csv` says which receivers a station is expected on:

```csv
callsign,name,sources
W6ABC,Alice,Repeater
KD9MNO,Dave,Simplex
N5DEF,Carol,Repeater;Simplex
KJ6TUV,Frank,
```

Each receiver then gets a prompt biased toward *its* stations, which is what
makes 20–50 per frequency fit where 100 never would. Stations who have not
checked in yet sort first — on a check-in net they are the ones about to speak.
Matching still runs against the whole roster, so a station who misses the
prompt is still matched; they just lose the decoding hint.

**3. Conditioned audio.** Every clip is high-passed and peak-normalised before
decoding — 0.8 ms for a five-second clip. Whisper was trained on normalised
audio, and this matters most on line-in and mic input where the level is
whatever the radio's volume knob happened to be.

**4. Escalation — the one that buys real accuracy.** The live line comes from a
fast model. Anything that comes back **unmatched or low-confidence** is queued
for a second pass with a bigger model, run only when nothing live is waiting,
and the line is updated in place. The re-transcribed line carries `escalated`
in the log and in the export — but **the dashboard does not show it yet**, and
nothing anywhere marks a line as *waiting* for its second pass, so between
queueing and completion an escalated line is indistinguishable from a finished
one. Both are on the list in [docs/STATUS.md](docs/STATUS.md). Since only the
hard clips are
escalated, you pay a fraction of the big model's cost while getting its
accuracy exactly where the fast one failed:

```yaml
whisper:
  model_size: base      # the live line
escalation:
  enabled: true
  model_size: medium    # or large-v3 on a GPU
```

**Skip `small` for the second pass.** It is still the shipped default, but on
the benchmark in [docs/HARDWARE.md](docs/HARDWARE.md) it recovered *fewer*
callsigns than `base` in every engine configuration — so escalating to it can
cost accuracy while spending twice the compute. `medium` scored full marks.
One synthetic net is not proof, but it is the only evidence there is, and it
points away from the default.

The second pass is also *targeted*: it biases toward the handful of roster
callsigns nearest to what was actually heard, which is a short list that fits
easily — so this scales to any roster size. An operator correction always wins;
a re-run never overwrites a human.

### Recognising a station by voice

Everything above works on what was *said*. This works on *who said it*, for the
case nothing else can reach: **a transmission with no usable callsign in it**.
"Back to you, net control" stays unmatched no matter how good transcription
gets — but it is still Frank's voice, and Frank checked in ten minutes ago.

```yaml
voice:
  enabled: true
```

The training data is already being produced. Every clean roster match, and
every line you correct by hand, is a labelled (audio, callsign) pair. Profiles
persist in `voices.json`, so the system knows more voices every week without
anyone doing anything extra.

The clips behind each profile are kept too (`voice.keep_audio`, on by
default). Embeddings from two different models are not comparable, so without
the audio, changing the embedder later would void every profile and enrolment
would start from nothing. With it, `python tools/rebuild_voices.py` re-embeds
everything in a single pass — and `--compare` scores two embedders on
identical clips, which is the only fair way to test one. Budget about a
megabyte per station.

**Suggestions only, and only on unmatched lines.** An unmatched row shows
`sounds like KJ6TUV?` — one click confirms it, through the same correction path
as any manual fix, which also learns the alias *and* the voice. A voice match
never overrides a callsign that was actually heard and never fills one in
silently, because the failure modes are real: FM narrowband flattens the
features that distinguish speakers, two operators share one radio, and a
relayed transmission is somebody else's voice entirely.

`min_similarity` (0.82) is the number to tune against your own net; it is the
one thing no amount of synthetic testing can set correctly. Start high — a
suggestion you have to think about is worse than no suggestion.

### Which embedder

Two, chosen with `voice.backend`:

**`builtin`** (default) — log-mel cepstral statistics in numpy. Nothing to
install, runs on a Pi, and honest about being weak: 24 hand-picked numbers that
were never trained to tell one speaker from another. It also partly identifies
the *radio* rather than the operator, so a profile built from somebody's HT
breaks when they check in mobile.

**`onnx`** — a trained speaker model (ECAPA-TDNN, TitaNet, or similar) run
through `onnxruntime`: an 18 MB wheel plus a model of about the same, against
hundreds of megabytes for PyTorch or NeMo. Discriminatively trained on
thousands of speakers and augmented specifically for changes of microphone.
Download an ONNX export, point `voice.model_path` at it, and set the backend:

```yaml
voice:
  enabled: true
  backend: onnx
  model_path: models/speaker.onnx
```

It runs on the **CPU on purpose**. A speaker embedding is a small model over a
few seconds of audio — milliseconds either way — and any GPU present belongs
to Whisper.

Exported speaker models disagree about their inputs: waveform or features,
which way round the feature axes go, whether a length is passed alongside. The
model is inspected and the input built to match, so a model downloaded later
has a fair chance of working without a code change. A missing or unreadable
model logs one line and falls back to `builtin` rather than stopping the net.

**Switching backends invalidates every stored profile** — vectors from two
models mean nothing to each other. That is what the kept enrolment audio is
for:

```bash
python tools/rebuild_voices.py --compare
```

re-embeds the stored clips in one pass and scores the new embedder on the same
audio, so the two can be compared honestly rather than across two different
months of traffic.

### Traffic

A net exists partly to move traffic, and "who still has something to pass" is
otherwise a list somebody keeps in their head. Each line is read for a traffic
declaration and marked:

```
[19:04:12] K7XYZ (Turn 7 / Bob): [TRAFFIC] checking in with traffic for net control
```

The dashboard badges those lines, counts them in the header, marks the holding
stations in the sidebar, and the count doubles as a filter — one click shows
only the traffic.

**Click the badge when the traffic has been passed** and it clears: the badge
turns to `passed`, the station drops off the outstanding list, and the header
count goes down. That makes it a working list that empties rather than a tally
that only grows. Clicking again puts it back, because a mis-click during a
busy net should cost a second click and not a restart — and the declaration is
never erased, so the exported log separates what is outstanding from what was
handled:

```
Traffic outstanding: KD9MNO
Traffic passed: K7XYZ
```

Both halves are optional — `traffic.detect: false` removes the badges
entirely, and `traffic.acknowledge: false` keeps the badges as a read-only
marker.

There are three states, not two: **declared traffic**, **explicitly none**
("no traffic", "nothing for the net"), and **did not say**. Only the first is
badged. Marking "no traffic" would put a badge on most of the net, which is
the same as marking nothing.

The trap this is built around is that the *negative* is far more common than
the positive, so the detector checks for negation before anything else — and
treats a question ("any traffic for the net?") as net control soliciting
rather than net control holding. Where the phrasing is genuinely ambiguous it
records nothing rather than guessing; a badge nobody trusts is worse than no
badge.

### When two stations land in one clip

If stations key up inside `vad.silence_ms` of each other, the VAD hands both to
Whisper as a single clip. The app splits those back into separate log lines —
but only on evidence, because these two transcripts look identical:

```
"W6ABC checking in"  …pause…  "K7XYZ also checking in"    two stations
"W6ABC here, I have traffic for K7XYZ"                    one station
```

What tells them apart is the **pause**: two transmissions have dead air where
one operator unkeys and another keys up, and a sentence naming somebody else
does not. So the split is decided on Whisper's word timings, never on the text.
`split.min_gap_ms` (default 500) is how much dead air it takes; raise it if a
station naming another is being logged as two check-ins, lower it if genuine
back-to-back stations are being merged. `split.enabled: false` turns it off.

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

## Transcripts are written as the net runs

The session does not live only in memory waiting for a clean shutdown. Every
line goes to disk as it is produced, into `transcripts/`:

| File | What it is |
| --- | --- |
| `net-<stamp>.jsonl` | Append-only, one JSON object per line, fsynced. The durable record, and an auditable history — corrections are appended rather than overwriting what the machine originally said. |
| `net-<stamp>.txt` | The human-readable log, rewritten in transmission order after every change. Always current; this is the one that gets pasted into a net report. |

Verified the way it matters: `kill -9` mid-net, no cleanup of any kind, and the
complete log was on disk. A power cut costs at most the last line.

Set `transcripts.live: false` to turn it off, or `fsync: false` if you would
rather have the few milliseconds back than the guarantee.

**Export log** still works for an on-demand copy, and a clean exit still writes
a final one — but neither is now the thing standing between you and a lost net.

If the app does go down mid-net, restart it with `--resume`:

```bash
python app.py --resume
```

It reloads the interrupted log, remembers who had already checked in, and keeps
writing to the *same* files — so the net ends with one record rather than two
halves to staple together. Pass a filename to resume a specific session.

## Watchdog, logging, and alerting

The failure worth guarding against is not a crash — it is the pipeline that
keeps running while logging nothing, because the SDR app was closed, the
squelch stayed shut, or the sink got repointed. Forty minutes of the net go
missing and the dashboard looks fine the whole time.

So the app watches three things: whether audio frames are arriving, whether
there is any *signal* in them, and whether the machine is keeping up.

**On the dashboard** — two halves. A **status strip** is always on, showing a
level meter per receiver, how full the audio buffer is, the queue depth, the
realtime factor, system load, memory and uptime. That is the half you watch to
see a level sagging or the machine running out of headroom *before* anything
has gone wrong; hide it with the **Status** button.

The other half only speaks when something is already wrong: a red or amber
**banner** naming the problem, a status dot that changes colour, and one beep
on the transition (toggle with **Alerts**; the setting sticks). The beep
matters because the operator is usually looking at the radio, not the screen.

The number worth knowing on the strip is **speed** — the realtime factor.
Above 1.00× a clip took longer to transcribe than it took to say, which is the
point at which a busy net will start arriving late. Drop a model size from the
Settings panel and watch it fall.

With an NVIDIA GPU present the strip also shows its utilisation and VRAM, and
**which device inference actually resolved to** — `cuda/float16` against
`cpu/int8`. That last one exists because `device: auto` silently falling back
to the CPU is a thing to notice before an event rather than during one; the
compute reading turns amber when a GPU is sitting there unused. `nvidia-smi` is
enough for the numbers, and installing `nvidia-ml-py` makes the polling
cheaper.

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
  callsign that was heard. Click any callsign to fix it — the app
  [learns the correction](#corrections-and-what-the-app-learns-from-them).
- **A tab per receiver** when you run more than one, with the first configured
  source as the default view, a health dot per tab, and an unread count so a
  check-in on another frequency is not missed. An **All** tab interleaves them.
- Roster sidebar doubles as a check-in list — stations light up as they check
  in, with a count. Click one to filter the log to that station.
- Lines recovered from the backlog are marked **late**, and sit in their correct
  place in time rather than at the bottom.
- **Alerts** toggle for the beep that accompanies the health banner; **Auto-
  scroll** toggle for reviewing history mid-net without fighting the log.
- **Export log** writes a CSV and a text log to `export_dir` on demand — but you
  do not have to remember to press it, because the session is
  [written continuously](#transcripts-are-written-as-the-net-runs).

The page reconnects on its own if the app restarts, and says so plainly while
it is disconnected rather than showing a stale log as though it were live.

## Tests

```bash
.venv/bin/python -m pytest
```

373 tests, all offline, no audio hardware needed — CI runs them on every push
across Python 3.11–3.13, plus a job with the optional libraries removed so the
Raspberry Pi fallback paths are exercised too. `test_callsign_match.py`
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
