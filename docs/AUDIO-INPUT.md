<!-- Split out of README.md, which had grown to a thirty-minute read. -->

# Audio input

What to plug in, how the app finds it, and how to run more than one
receiver at once. Field bring-up against real hardware is a separate
checklist: [FIELD-BRINGUP.md](FIELD-BRINGUP.md).

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

## Multiple receivers

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
