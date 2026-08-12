# Architecture

Written for the version of you that comes back to this in six months.

## Data flow

```
SDR++ / GQRX  (separate app, on the host)
     │  audio out → PulseAudio/PipeWire loopback sink
     ▼
audio_capture.py    reads the sink's monitor source, 16 kHz mono int16 frames
     │  30 ms frames over a bounded queue
     ▼
vad_segmenter.py    webrtcvad state machine → one Clip per transmission
     │  Clip(float32 audio, start_offset_ms, duration_ms)
     ▼
stt_worker.py       faster-whisper → text + confidence
     │  Transcription(text, confidence, ...)
     ▼
callsign_match.py   learned aliases → normalize → extract → fuzzy match roster
     │  MatchResult(matched, callsign, score, candidate, reason, via_alias)
     ▼
transcript_store.py in-memory session log + CSV/text export
     │                        ▲
     │                        │ operator corrections (POST /api/correct)
     │                  feedback.py -- append-only log, replayed into aliases
     │
     ▼
server.py           FastAPI: GET /api/history, POST /api/export, WS /ws
     │
     ▼
static/             plain-JS dashboard, no build step
```

`app.py` wires it together and owns the process lifecycle.

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

The threshold and margin encode a deliberate bias: **prefer "unmatched" over a
wrong callsign.** Net control will notice a blank and resolve it by ear; they
will not notice a plausible wrong callsign in a scrolling log.

The vocabulary tables (`PHONETIC_MAP`, `DIGIT_MAP`, `AMBIGUOUS_DIGIT_MAP`) are
the main tuning surface. They are meant to grow as real nets surface new
mis-transcriptions — see [TESTING.md](TESTING.md).

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

### `config.py`

YAML plus `NETSTT_<SECTION>_<KEY>` env overrides, with env winning. The env
layer exists so the container image is configurable without baking a config
file into it. Unknown YAML keys are ignored rather than fatal.

## Design decisions

**Multiple sources, one transcriber.** Each receiver gets its own capture
thread, ring buffer, VAD, and health monitor — a dead simplex rig must not make
the repeater look broken, and the operator needs to know which one to go and
fix. But they share one Whisper model: it is the memory-hungry component, and
two instances on a Pi would thrash rather than parallelise. Clips from all
sources queue together and are transcribed in arrival order.

Entries are tagged with their source only when more than one is configured —
a source column on every line of a single-receiver net is noise.

**In-memory transcripts.** This is a session tool, not an archive. A database
would add a dependency and a migration story for something that gets exported
to a text file at the end of the night and pasted into a net report.

**Roster is a flat CSV.** It gets edited by a human, in a spreadsheet, before a
net. Anything richer than `callsign,name` should be justified by a real need.
