# Field bring-up

The whole app has been developed and tested against recorded and synthesized
audio. **The live SDR audio path has never run against real hardware.** This is
the checklist for the first time it does. Work down it in order — each step
isolates one thing, so when something fails you know what.

Budget an hour, and do it on a day when there is *no* net you care about.

## 0. Before you travel

Do these at home, they need no radio:

> The clips behind each voice profile are kept by default
> (`voice.keep_audio`), so a better embedder can be tried later without
> throwing away enrolment. Leave it on unless SD card space is tight.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest
```

```bash
python tools/make_test_audio.py && python app.py --file test-net.wav --model tiny
```

If the dashboard fills in with six transmissions, everything except audio
capture is working. Also pre-download the model you plan to use in the field,
since the model cache is the only thing that needs the network:

```bash
python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"
```

## 1. Confirm the OS sees the audio

With SDR++/GQRX running and receiving:

```bash
pactl list sources short
```

You want the `.monitor` source belonging to whatever sink the SDR app outputs
to. If you have not set up a dedicated sink yet:

```bash
pactl load-module module-null-sink sink_name=net_sink sink_properties=device.description=NetAudio
```

Set the SDR app's output device to `NetAudio`. This survives until reboot; make
it permanent in `/etc/pulse/default.pa` once it works.

Then confirm the app sees it too:

```bash
python app.py --list-devices
```

**If the device is missing here but present in `pactl`:** PortAudio is not
seeing PulseAudio. Check that `libportaudio2` is installed and that you are not
inside a container without the socket mounted.

## 2. Confirm audio is actually flowing

The most common failure is a device that exists but carries silence — the SDR
app's output is going somewhere else. Record five seconds and look at it:

```bash
parecord --device=net_sink.monitor --file-format=wav --rate=16000 --channels=1 test.wav & sleep 5; kill %1
```

```bash
python -c "
import wave, numpy as np
w = wave.open('test.wav'); a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
print(f'peak={abs(a).max()} rms={np.sqrt((a.astype(float)**2).mean()):.0f}')"
```

- `peak=0` — no audio at all. The SDR app is not outputting to this sink.
- `peak` under ~500 with the squelch open — level is too low; raise the SDR
  app's output volume. Whisper degrades badly on quiet audio.
- `peak` pinned at 32767 — clipping; lower it.

Aim for peaks around a third to half of full scale on normal speech.

## 3. First live capture

```bash
python app.py -v
```

Key someone up, or wait for a transmission. Watch the log, not the dashboard,
for this step. You are looking for one `INFO net-stt:` line per transmission.

**Nothing appears at all:**
- Squelch open with hiss reaching the app? `vad.aggressiveness: 3` is
  deliberately harsh — try 1 or 2.
- Check the level from step 2 again.

**Every squelch tail produces a junk line:** raise `vad.min_clip_ms` to 600–800.

**One check-in becomes three log lines:** raise `vad.silence_ms`. Operators who
spell their callsign slowly need more; 1200 is not unreasonable.

**Two stations merge into one line:** the splitter should catch this (see step
6), but you can also lower `vad.silence_ms` toward 500 so the VAD separates
them in the first place.

Tune this against a *recording* of your net rather than live traffic if you can
— record 10 minutes with `parecord`, then iterate with `--file`. Each pass is
seconds instead of a week.

## 4. Check latency

Time from end-of-transmission to the line appearing. A few seconds is fine; a
net moves slower than that.

If it lags, the banner will tell you before you notice it yourself: clips
spilling to disk means the machine cannot keep up. Drop a model size. On a Pi,
`base` is roughly the ceiling for a Pi 4 and `small` for a Pi 5 — and note that
two receivers share one model, so a second source roughly doubles the load.

Lateness is no longer data loss: a slow machine makes lines arrive late and
flagged, not missing. But a net that is permanently behind is still worth
fixing.

## 5. If you run more than one receiver

Bring each one up **on its own first**, using the steps above with only that
source enabled. Two receivers failing at once is much harder to diagnose than
one, and `enabled: false` lets you park the second while you sort out the first.

Once both are up, the dashboard gives each a tab, and the first source listed
is the default view — put the repeater there. Watch for:

- The level on each source. They will not match; `gain` is per source for
  exactly this reason.
- Whether the weaker receiver needs a gentler `aggressiveness` or a longer
  `silence_ms` than the repeater. Both are per-source overrides.
- The health dot on each tab. A source that never opened shows there, not just
  in the banner.

## 6. Check the matcher against your real roster

This is where the interesting failures are, and the reason the vocabulary
tables exist. Have a few known stations check in and watch the callsign column.

For each miss, capture the *exact* transcript text from the log line — the
`raw_text` field, verbatim — and follow the workflow in
[TESTING.md](TESTING.md#adding-a-regression). It is a two-line change plus a
test, and it makes that mis-transcription permanently handled.

Expect the first real net to surface several. That is the system working as
intended: the roster is what corrects Whisper, and the tables are what teach it
your net's specific failure modes.

**If a station matches the *wrong* roster entry**, that is more serious than an
unmatched line. Raise `roster.threshold` or `roster.ambiguity_margin` and file
the transcript as a test case.

**If two stations key up back to back**, they can land in one clip. The app
splits those into separate lines when it can hear real dead air between them
(`split.min_gap_ms`, default 500 ms). Two things to check on a fast net:

- A station who merely *names* another station ("W6ABC here, traffic for
  K7XYZ") must stay one line. If those are being logged as two check-ins,
  raise `split.min_gap_ms`.
- Two stations that were logged as one line mean the pause between them was
  shorter than the threshold. Lower it, or lower `vad.silence_ms` so the VAD
  separates them before the splitter has to.

## 7. Run a real net

Keep a paper log the first time, and compare afterwards.

**You do not have to remember to save anything.** The session is written to
`transcripts/` as it happens — a `.jsonl` fsynced line by line, and a `.txt`
kept in transmission order. If the app crashes, the Pi loses power, or somebody
closes the laptop, the log is already on disk; a power cut costs at most the
last line. This was tested with `kill -9` mid-net and no cleanup at all.

For an extra copy at a specific moment:

```bash
curl -X POST localhost:8080/api/export
```

If something does go wrong, restart with `--resume`: it reloads the
interrupted log, remembers who had already checked in, and keeps writing to the
same files, so the net ends with one record rather than two halves.

```bash
python app.py --resume
```

## Known-unverified list

Things no test has ever exercised, in rough order of how likely they are to
bite:

- **Device selection by name substring** (`audio_capture.find_device`) against
  real PulseAudio device names.
- **Overflow handling** under sustained load — the ring buffer's drop path is
  unit-tested but has never been hit by a real machine falling behind.
- **A device that disappears mid-net** (USB SDR unplugged). The restart-with-
  backoff path is exercised against a device that never existed, not one that
  vanished while streaming.
- **`vad.*` defaults** against real receiver audio. They were tuned on
  synthesized speech, which is cleaner and more consistent than anything off
  the air.
- **`split.min_gap_ms`** against real back-to-back transmissions. The logic is
  tested, but the 500 ms threshold is a guess about how your net actually
  sounds.
- **Two receivers running at once on real hardware.** Tested with two recorded
  files, never with two sound cards.
- **Whisper accuracy on weak/noisy signals.** TTS test audio makes the pipeline
  look considerably better than it will be on a marginal simplex signal.
- **CUDA auto-detection**, if you plan to use a GPU.
- **The container's Pulse socket mount.** The image builds and serves, but has
  only run without audio.
