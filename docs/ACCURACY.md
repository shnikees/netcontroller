<!-- Split out of README.md, which had grown to a thirty-minute read. -->

# Getting the transcript right

Model choice, the four things that buy accuracy, and how to tune against
your own recordings. Measured engine and model-size comparisons live in
[HARDWARE.md](HARDWARE.md).

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

## Choosing a model

Rough figures for one ~5 second transmission on CPU. Measure on your own
hardware; this is a guide, not a benchmark.

| Model | Laptop CPU | Pi 4 | Pi 5 | Accuracy |
| --- | --- | --- | --- | --- |
| `tiny` | ~0.2 s | ~2 s | ~1 s | Rough. Drops words when an operator rattles a callsign off quickly; leans hard on the roster to recover |
| `base` | ~0.4 s | ~4 s | ~2 s | Good default for an unhurried net |
| `small` | ~1.5 s | too slow | ~6 s | **Measured worse than `base`** at callsign recovery — see below |
| `medium` | ~4 s | no | no | Best accuracy; needs a GPU to be worth it |

On the synthetic test net in [HARDWARE.md](HARDWARE.md), `small`
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

## Accuracy without losing speed

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
and the line is updated in place and marked. The dashboard says which of the
two states a line is in, because they mean opposite things to a reader: a
queued line shows **waiting** — do not settle on this yet, it may still
change — and a re-transcribed one shows **2nd pass**, meaning the better model
has already had it. The strip counts what is outstanding. Since only the
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
the benchmark in [HARDWARE.md](HARDWARE.md) it recovered *fewer*
callsigns than `base` in every engine configuration — so escalating to it can
cost accuracy while spending twice the compute. `medium` scored full marks.
One synthetic net is not proof, but it is the only evidence there is, and it
points away from the default.

The second pass is also *targeted*: it biases toward the handful of roster
callsigns nearest to what was actually heard, which is a short list that fits
easily — so this scales to any roster size. An operator correction always wins;
a re-run never overwrites a human.

## When two stations land in one clip

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
