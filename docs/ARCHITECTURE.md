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
callsign_match.py   normalize → extract candidates → fuzzy match roster
     │  MatchResult(matched, callsign, score, candidate, reason)
     ▼
transcript_store.py in-memory session log + CSV/text export
     │
     ▼
server.py           FastAPI: GET /api/history, POST /api/export, WS /ws
     │
     ▼
static/             plain-JS dashboard, no build step
```

`app.py` wires it together and owns the process lifecycle.

## Threading model

There are exactly two threads, and the split matters:

- **Main thread**: asyncio loop running uvicorn. Serves HTTP and websockets.
- **Capture thread** (`Pipeline._run`): audio → VAD → Whisper → matcher → store.

Whisper inference blocks for hundreds of milliseconds to seconds. If it ran on
the event loop, every transcription would freeze the dashboard. Instead the
capture thread pushes finished entries across with
`asyncio.run_coroutine_threadsafe`, which is the only point where the two
threads touch.

Backpressure is handled by dropping, not queueing: `AudioCapture` uses a bounded
queue and discards frames when the consumer falls behind. On a machine too slow
for the chosen model, you lose audio rather than accumulating an ever-growing
backlog that puts the dashboard minutes behind the net. If you see
`AudioCapture.overflows` climbing, the model is too big for the hardware.

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

**Single input.** Two repeaters would mean two instances on two ports. Sharing
one process would mean interleaving transmissions from different nets in one
log, which is not what a net control operator wants to read.

**In-memory transcripts.** This is a session tool, not an archive. A database
would add a dependency and a migration story for something that gets exported
to a text file at the end of the night and pasted into a net report.

**Roster is a flat CSV.** It gets edited by a human, in a spreadsheet, before a
net. Anything richer than `callsign,name` should be justified by a real need.
