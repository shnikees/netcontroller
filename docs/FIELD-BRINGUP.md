# Field bring-up

The whole app has been developed and tested against recorded and synthesized
audio. **The live SDR audio path has never run against real hardware.** This is
the checklist for the first time it does. Work down it in order — each step
isolates one thing, so when something fails you know what.

Budget an hour, and do it on a day when there is *no* net you care about.

## 0. Before you travel

Do these at home, they need no radio:

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

**Two stations merge into one line:** lower `vad.silence_ms` toward 500.

Tune this against a *recording* of your net rather than live traffic if you can
— record 10 minutes with `parecord`, then iterate with `--file`. Each pass is
seconds instead of a week.

## 4. Check latency

Time from end-of-transmission to the line appearing. A few seconds is fine; a
net moves slower than that.

If it lags: drop to a smaller model, or watch for `AudioCapture.overflows`
climbing, which means the machine cannot keep up and is dropping audio. On a
Pi, `base` is roughly the ceiling for a Pi 4 and `small` for a Pi 5.

## 5. Check the matcher against your real roster

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

## 6. Run a real net

Keep a paper log the first time. Compare afterwards, and export the session:

```bash
curl -X POST localhost:8080/api/export
```

The text log is also written automatically on Ctrl-C, so a clean exit never
loses the session.

## Known-unverified list

Things no test has ever exercised, in rough order of how likely they are to
bite:

- **Device selection by name substring** (`audio_capture.find_device`) against
  real PulseAudio device names.
- **48 kHz → 16 kHz decimation** (`AudioCapture._downsample`). The logic is
  simple and unit-testable in principle, but it has only ever run on audio that
  was already 16 kHz.
- **Overflow handling** under sustained load — the drop path has never been hit.
- **`vad.*` defaults** against real receiver audio. They were tuned on
  synthesized speech, which is cleaner and more consistent than anything off
  the air.
- **Whisper accuracy on weak/noisy signals.** TTS test audio makes the pipeline
  look considerably better than it will be on a marginal simplex signal.
- **CUDA auto-detection**, if you plan to use a GPU.
- **The container's Pulse socket mount.** The image builds and serves, but has
  only run without audio.
