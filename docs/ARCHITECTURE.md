# Architecture

Written for the version of you that comes back to this in six months.

## Data flow

Everything above the clip queue runs **once per receiver**; everything below it
is shared.

```
  ┌─ Receiver: "Repeater" ────────┐   ┌─ Receiver: "Simplex" ─────────┐
  │ SDR loopback / line in / mic  │   │ …one of these per source      │
  │            │                  │   │                               │
  │   audio_capture.py            │   │                               │
  │   + resample.py   any rate → 16 kHz, pick/mix channel, apply gain  │
  │            │                  │   │                               │
  │   ring_buffer.py  pre-allocated; the callback never allocates      │
  │            │  30 ms int16 frames                                   │
  │   vad_segmenter.py  webrtcvad → one Clip per transmission          │
  └────────────┼──────────────────┘   └───────────────┬───────────────┘
               │  Clip(audio, start_offset_ms, duration_ms, source, sequence)
               └───────────────┬───────────────────────┘
                               ▼
                    clip queue  (shared, bounded)
                               │                    ╲ full?
                               │                     ╲
                               │              clip_spill.py  WAVs on disk,
                               │                     ╱       replayed in lulls
                               ▼                    ╱
                    stt_worker.py   audio_prep → faster-whisper → text
                               │  Transcription(text, confidence, words)
                               │  unsure? → second pass with a bigger model,
                               │            biased toward the nearest roster
                               │            entries, run only when idle
                               ▼
                    callsign_match.py   aliases → normalize → extract → fuzzy match
                               │  MatchResult(matched, callsign, score, …)
                    clip_split.py   two stations in one clip? split on the pause
                               │  one Segment per transmission
                               ▼
                    transcript_store.py   in-memory log, ordered by timestamp
                               │      │  every line, as it happens
                               │      ├─► session_writer.py  .jsonl + .txt on disk
                               │      ▲
                               │      │ corrections (POST /api/correct)
                               │  feedback.py   append-only log → learned aliases
                               ▼
                    server.py   FastAPI + websocket
                               │  /api/history · /api/correct · /api/health
                               │  /api/aliases · /api/export · WS /ws
                               ▼
                    static/   plain-JS dashboard, no build step
```

Alongside all of it, `health.py` watches each source and `app.py`'s watchdog
turns that into the dashboard banner, the log, and the `/api/health` status
code. `app.py` wires everything together and owns the process lifecycle.

## File map

| File | What it does |
| --- | --- |
| `app.py` | Entrypoint, `SourceCapture` and `Pipeline`, watchdog task |
| `audio_capture.py` | Device selection, channel/gain, → 16 kHz frames |
| `resample.py` | Any sample rate → 16 kHz (soxr, or a built-in fallback) |
| `audio_prep.py` | High-pass and normalise a clip before decoding |
| `ring_buffer.py` | Pre-allocated audio buffer between callback and VAD |
| `vad_segmenter.py` | One clip per transmission |
| `clip_spill.py` | Disk overflow when the transcriber is behind |
| `stt_worker.py` | faster-whisper wrapper |
| `callsign_match.py` | Normalizer + roster matcher — the domain logic |
| `clip_split.py` | Splits a clip that caught two stations into separate lines |
| `traffic.py` | Reads a traffic declaration off a transmission |
| `feedback.py` | Operator corrections, and the aliases learned from them |
| `voice_id.py` | Voice profiles, kept enrolment audio, suggestions |
| `calibrate.py` | Thresholds derived from collected sessions and voices |
| `transcript_store.py` | Session log, ordering, CSV/text export |
| `session_writer.py` | Streams the session to disk as it happens |
| `health.py` | Per-source health, and the combined verdict |
| `logging_setup.py` | Console + rotating file logging |
| `config.py` | YAML + `NETSTT_*` env overrides |
| `server.py` | HTTP + websocket |
| `static/` | Dashboard |

## Threading model

Four contexts, and the boundaries between them are the design:

| Context | Work | Speed |
| --- | --- | --- |
| PortAudio callback (one per source) | resample, write to ring buffer | real-time, never allocates |
| Capture thread (one per source, `SourceCapture._run`) | ring → frames → VAD → clip queue | cheap, always keeps up |
| STT thread (`_transcribe_loop`) | Whisper → matcher → store | slow, allowed to fall behind |
| Main thread | asyncio + uvicorn: HTTP, websockets, watchdog | must stay responsive |

The STT split is the one that took a bug to find. Originally VAD and Whisper
shared a thread, so **nothing drained the audio buffer during a transcription**.
On a Pi, a 4-second inference meant the next transmission was dropped mid-word
— the failure was invisible, because the missing line looked like a station
that never checked in.

Now backpressure has three stages, each degrading further than the last:

1. **Ring buffer** (`ring_buffer.py`, 30 s default) — pre-allocated once, so the
   audio callback never allocates and never blocks. Overrun drops the *oldest*
   audio, because old frames belong to a transmission that was already
   truncated.
2. **Clip queue** (32 clips) — finished clips waiting for Whisper.
3. **Disk spill** (`clip_spill.py`) — past that, clips become WAVs on disk and
   are transcribed in the next lull. Late, flagged, and in the right place in
   the log, because `TranscriptStore` inserts by timestamp rather than arrival.

The rule the whole chain enforces: **a slow machine makes transcripts late, not
missing.** `test_pipeline.py` asserts exactly that against a transcriber
slower than real time.

## Module notes

### `callsign_match.py`

The one module worth reading closely, and the only one with real domain logic.
Four stages, each separately testable:

1. `normalize()` — phonetics → letters, spoken digits → numerals, filler
   dropped, single characters glued into words. Also splits tokens Whisper ran
   together (`alfabravo` → `A B`) using a dynamic-programming word split that
   only fires when the *whole* token decomposes into known vocabulary, so
   ordinary English is untouched.
2. `extract_candidates()` — regex for US callsign structure. Strict matches
   (1–2 letters, digit, 1–3 letters) rank ahead of loose ones, so a clean read
   wins over a mangled one in the same transmission.
3. `CallsignMatcher._match_candidate()` — `rapidfuzz` against the roster.
4. Accept/reject — threshold plus an ambiguity margin.

Dropped filler leaves a `BREAK` marker rather than nothing, so gluing does not
run across the gap. Without it, "W6ABC no traffic K7XYZ" — two stations landing
in one clip, which is what a fast net produces — welded into `W6ABCK7XYZ` and
*both* callsigns were lost.

The threshold and margin encode a deliberate bias: **prefer "unmatched" over a
wrong callsign.** Net control will notice a blank and resolve it by ear; they
will not notice a plausible wrong callsign in a scrolling log.

The vocabulary tables (`PHONETIC_MAP`, `DIGIT_MAP`, `AMBIGUOUS_DIGIT_MAP`) are
the main tuning surface. They are meant to grow as real nets surface new
mis-transcriptions — see [TESTING.md](TESTING.md).

### `stt_worker.py` and the prompt budget

Whisper's prompt window is 224 tokens, and overflow is discarded without a
word. A written callsign costs about four tokens, so about 48 fit — against a
roster of 40-100 across two frequencies. Listing the roster therefore does not
work: most of it would be dropped at an arbitrary point, which is strictly
worse than a shorter prompt chosen deliberately.

`build_prompt` counts real tokens with the model's own tokenizer and stops at
the budget. What survives is decided by `CallsignMatcher.bias_terms`, ordered
by how likely each station is to be the next voice *on this receiver*: the
phonetic alphabet first (26 words that bias every spelled callsign, where
per-station spellings would cost seven tokens each), then net vocabulary, then
this source's stations who have not checked in yet, then the rest.

**Escalation** is the other half. A clip that came back unmatched or unsure is
queued for a second pass with a larger model, run only when no live clip and no
spilled clip is waiting — improving an old line must never delay the current
one. That pass biases toward `matcher.nearest()`: the handful of roster entries
the first pass was already close to, which is short enough to fit whatever the
roster size. The result replaces the line only if it genuinely matched better,
and never if an operator has already corrected it by hand.

### `voice_id.py`

The only part of the app that works on *who* spoke rather than *what* was said,
and it exists for the one failure the rest cannot touch: a transmission with no
usable callsign in it. No transcription improvement reaches that line; a voice
does.

Enrolment is free, because the labels are already being produced — a clean
roster match, or a line an operator corrected, is a labelled (audio, callsign)
pair. Profiles are a running mean per station, persisted between nets.

**It only ever suggests, and only on unmatched lines.** The reasons are the
same ones that make the matcher prefer "unmatched" to a guess: FM narrowband
flattens speaker features, two operators share one radio, and a relay carries
somebody else's voice. So a suggestion is offered for the operator to accept
with the click that already exists, and `TranscriptStore.suggest` refuses a
line that is matched or has been corrected.

The embedding is log-mel cepstral statistics in numpy — mean and deviation
pooled over the clip, which makes it independent of what was said. Weaker than
a trained speaker network and chosen so the app still installs on a Pi without
a deep-learning runtime; with a high threshold and suggestion-only output, the
weakness costs recall rather than correctness.

### `clip_split.py`

On a fast net, stations key up inside the VAD's silence window and arrive as
one clip. Finding the second callsign is easy — the matcher already does. The
hard part is telling these apart:

```
"W6ABC checking in"  …pause…  "K7XYZ also checking in"    two stations
"W6ABC here, I have traffic for K7XYZ"                    one station
```

The transcripts are indistinguishable. What separates them is the **pause**, so
the decision is made on Whisper's word timings and never on the text — which is
why `stt_worker.py` asks for `word_timestamps`.

The bias matches the matcher's: when the evidence is weak, keep it as one line.
An over-eager split invents a check-in that never happened, and nobody reading
the log will know to doubt it — the same reasoning behind preferring
"unmatched" to a guess. Splits are refused when there is no qualifying gap, no
word timings at all, fewer than two distinct roster stations, or when a segment
would come out too short to be a transmission.

### `traffic.py`

Reads "has traffic", "has none" or "did not say" off a transmission. Three
states rather than two, because a station saying "nothing for the net" and a
station not mentioning it are different facts, and only the first means the
list can stop watching them.

Built around one hazard: **the negative is far more common than the positive**.
Most stations say "no traffic", so a detector that keyed on the word alone
would flag the whole net, and net control would learn to ignore the column
within one session. Negation is checked first, questions ("any traffic for the
net?") are treated as soliciting rather than declaring, and ambiguity resolves
to "did not say" — a badge nobody trusts is worse than no badge.

Clearing is separate from declaring, and kept on its own field. A cleared line
still records that traffic was declared: what was handled is part of the
account of the net, not something to erase. The clear is a toggle in both
directions because a mis-click on a busy net should cost a second click.

### `feedback.py`

The learning loop, such as it is. An operator correction does three things: fix
the log line, append to `feedback.jsonl`, and teach the matcher that
`candidate -> callsign`. The next transmission Whisper mangles the same way then
matches on its own.

One file is both the audit log and the alias source — aliases are derived by
replaying the log at startup rather than stored separately, so there is no
second file to drift out of sync with the record of what the operator said.
JSON Lines, append-only: a write interrupted by a power cut costs one
correction, not the whole history.

Deliberate constraints, all tested:

- Aliases may only point at roster callsigns, and stale ones are dropped when a
  station leaves `roster.csv`. Otherwise last month's correction quietly
  resurrects a station that is no longer on the net.
- Candidates shorter than three characters are refused — `K7` would mis-fire
  constantly.
- An alias beats an `ambiguous` refusal. The operator heard the transmission;
  the matcher only saw a string.

This is also the labelled dataset. Each line pairs Whisper's output with a
human-confirmed callsign, which is what a supervised fine-tuning run would
consume. Nothing in the pipeline trains today — faster-whisper runs on
CTranslate2, an inference-only runtime — so "learning" here means the alias
table, not model weights.

### `audio_capture.py` and `resample.py`

One `AudioCapture` per source, and it does not care whether the audio comes
from an SDR loopback, a USB sound card fed by a radio's speaker jack, or a
microphone. The differences between those are three settings — sample rate,
channel, and level — so they are handled here rather than pushed onto the
operator.

Sample rate is the one with real substance. A loopback sink can be told to run
at 16 or 48 kHz; a microphone usually cannot, and offers 44.1 kHz, which is not
an integer multiple of 16 kHz. `resample.py` picks a path automatically:

| Case | Path |
| --- | --- |
| Already 16 kHz | passthrough |
| Integer ratio (48 kHz) | boxcar decimation — the content above 8 kHz on a narrowband voice channel is noise anyway |
| Anything else (44.1 kHz) | `soxr`, or a windowed-sinc low-pass plus linear interpolation if it is not installed |

The low-pass is not a nicety. Fold 15 kHz content down into the voice band and
it lands on top of the speech, costing accuracy on exactly the fast,
run-together delivery that is already hardest to transcribe. `test_resample.py` runs its
signal checks against **both** engines, because otherwise the fallback would be
dead code everywhere except a Pi in the field.

### `ring_buffer.py`

Storage allocated once and reused, sitting between the PortAudio callback and
the VAD. The callback previously built a `bytes` object per block — 33
allocations a second, plus the garbage behind them, which on a loaded Pi
invites xruns that arrive as clicks the VAD then mistakes for speech.

Honest caveat: this is Python, the callback still takes the GIL, and nothing
here is hard real-time. Removing the per-block allocation removes the part that
was ours to remove.

Overrun drops the **oldest** audio. If the reader has fallen behind, old frames
belong to a transmission that was already truncated, so overwriting them loses
nothing salvageable, while the newest audio is a transmission that might still
be logged intact.

### `clip_spill.py`

The last line of defence, and the stage that makes the whole design's promise
true: **a slow machine makes transcripts late, never missing.**

Clips become WAVs with a JSON sidecar, written oldest-first and read back
oldest-first, so a backlog drains in transmission order. The STT thread takes
live clips first and only touches the spill when the queue runs dry — during a
net, the line that matters is the one being spoken now.

The trade is explicit and visible: a spilled line appears late, flagged `late`
on the dashboard, but sits in its correct place in the log because
`TranscriptStore` inserts by timestamp rather than arrival. An exported net
report that reads out of order would be worse than a late line.

### `health.py`

Built around the failure that actually bites: not a crash, which is obvious,
but the pipeline that stays up while producing nothing — the SDR app closed,
the sink repointed, the squelch shut. The dashboard looks fine and forty
minutes of the net go unlogged.

So it watches for *silence where there should be sound*, at three levels:
frames arriving at all, signal within those frames, and whether the machine is
keeping up. Times come from an injectable clock, so a five-minute silence is
tested in microseconds and the suite never sleeps.

`HealthFleet` holds one monitor per source and reduces them to a single
verdict: worst state wins, and issues are prefixed with the source name
whenever there is more than one. With two radios in the room, "the pipeline is
unhealthy" is not an instruction anybody can act on.

### `vad_segmenter.py`

A state machine with pre-roll and hangover:

- **Pre-roll** keeps ~300 ms from *before* the trigger, so clips do not start
  mid-syllable. Without it, Whisper loses the first phonetic — which is the
  callsign's prefix letter, the character you can least afford to lose.
- **Trigger ratio** requires most of the pre-roll window to be voiced, which
  debounces single-frame noise bursts and squelch crashes.
- **Hangover** (`silence_ms`) is the parameter that will need tuning on real
  audio. Too short splits one check-in across several lines; too long merges
  two stations into one clip. It is trimmed back off before the clip is sent to
  Whisper, so the hangover costs nothing in inference time.

### `stt_worker.py`

Thin wrapper. Two settings are load-bearing:

- `condition_on_previous_text=False` — transmissions are independent. Left on,
  Whisper carries context between unrelated stations and hallucinates
  continuity that is not there.
- `initial_prompt` — built from the roster by `CallsignMatcher.hotwords()`, in
  both written (`W6ABC`) and spoken (`whiskey six alpha bravo charlie`) form,
  which biases decoding toward callsigns that actually exist on this net.

`vad_filter=False` because the segmenter already did that work.

The confidence number is `exp(avg_logprob)`, duration-weighted. It is a
monotonic proxy, not a calibrated probability — fine for colouring a cell,
not for making decisions.

### `transcript_store.py`

An in-memory list, with one rule that matters: entries are inserted **by
timestamp**, not by arrival. A clip recovered from the disk backlog arrives
after later ones but was spoken earlier, and belongs where it was spoken.

Corrections keep `original_callsign`, so the log still records where the
machine was wrong — that record is the point, not an embarrassment to hide.

### `session_writer.py`

Transcripts used to reach disk only on a clean exit or an Export click, which
holds right up until the Pi loses power and two hours of net vanish — the one
outcome this app exists to prevent.

Now every line is written as it is produced, in two files because they answer
different questions. The `.jsonl` is append-only and fsynced: the durable
record, which survives a power cut minus at most its last line, and which keeps
corrections as their own entries so the history stays auditable. The `.txt` is
rewritten in transmission order after each change, so the readable copy is
never behind — rewriting is cheap at net scale and removes the "remember to
export" step.

Failures degrade rather than propagate: a full or read-only disk disables
writing, logs once, and lets the net carry on.

### `config.py`

YAML plus `NETSTT_<SECTION>_<KEY>` env overrides, with env winning. The env
layer exists so the container image is configurable without baking a config
file into it. Unknown YAML keys are ignored rather than fatal.

`sources:` (a list) and the single `audio:` block are both valid;
`audio_sources()` resolves whichever is present. Breaking every existing config
to add a feature most operators will not use would be a poor trade.

### `logging_setup.py`

Console for the operator during the net, rotating file for the morning after,
when someone asks why a station is missing from the log — so the file keeps
more detail than the console shows. A file that cannot be opened (read-only
media, wrong owner in a container) is reported and skipped rather than fatal:
losing the log is not a reason to lose the net.

## Design decisions

**Multiple sources, one transcriber.** Each receiver gets its own capture
thread, ring buffer, VAD, and health monitor — a dead simplex rig must not make
the repeater look broken, and the operator needs to know which one to go and
fix. But they share one Whisper model: it is the memory-hungry component, and
two instances on a Pi would thrash rather than parallelise. Clips from all
sources queue together and are transcribed in arrival order.

Entries are tagged with their source only when more than one is configured —
a source column on every line of a single-receiver net is noise.

**Receivers are weighted, not uniform.** A repeater arrives strong and carries
the net; a staging channel may be weak and slow. So VAD settings are per-source
(falling back to the global block), `gain` levels the inputs, and `priority`
orders the shared clip queue — the frequency the net runs on should not wait
behind side traffic when the transcriber is behind.

**In-memory transcripts.** This is a session tool, not an archive. A database
would add a dependency and a migration story for something that gets exported
to a text file at the end of the night and pasted into a net report.

**Roster is a flat CSV.** It gets edited by a human, in a spreadsheet, before a
net, so columns are read by name and order does not matter. It carries three
things beyond the callsign, each earning its place: `name`, `position` — where
the operator is posted, which on an event net is what the callsign is *for* —
and `sources`, which receivers to expect them on. Anything further should be
justified by a real need.
