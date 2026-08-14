<!-- Split out of README.md, which had grown to a thirty-minute read. -->

# Callsign matching

The roster, how a spoken callsign becomes a match, and the two things that
identify a station when the callsign itself is unusable.

## The roster

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

## Recognising a station by voice

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

## Which embedder

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
