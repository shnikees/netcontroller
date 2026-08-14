<!-- Split out of README.md, which had grown to a thirty-minute read. -->

# Running a net

The dashboard, the settings panel, what gets written to disk, and what
happens when something breaks mid-net.

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

## Traffic

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
