# netcontroller -- live speech-to-text and callsign matching for ham radio nets
# Copyright (C) 2026 Michelle Michaels
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

"""Ham radio net speech-to-text pipeline.

    python app.py --config config.yaml
    python app.py --list-devices
    python app.py --file recorded-net.wav      # offline replay, no SDR needed

The audio -> VAD -> STT chain runs on a worker thread; finished entries are
handed to the asyncio loop, which broadcasts them to the dashboard. Keeping the
blocking work off the loop is what stops a slow transcription from stalling the
websocket.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import uvicorn

from audio_capture import TARGET_RATE, AudioCapture, list_devices
from callsign_match import CallsignMatcher, load_roster
from config import Config, load_config
from feedback import FeedbackLog
from health import ERROR, OK, WARNING, HealthMonitor
from logging_setup import setup_logging
from resample import Resampler, describe
from server import Broadcaster, create_app
from stt_worker import SttWorker
from transcript_store import TranscriptStore
from vad_segmenter import VadSegmenter

log = logging.getLogger("net-stt")


class AudioUnavailable(RuntimeError):
    """The capture device could not be opened; the message is operator-facing."""


class Pipeline:
    """Owns the capture thread and turns clips into transcript entries."""

    def __init__(
        self,
        config: Config,
        store: TranscriptStore,
        matcher: CallsignMatcher,
        broadcaster: Broadcaster,
        loop: asyncio.AbstractEventLoop,
        health: HealthMonitor,
        wav_path: str | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.matcher = matcher
        self.broadcaster = broadcaster
        self.loop = loop
        self.health = health
        self.wav_path = wav_path
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture: AudioCapture | None = None

        self.stt = SttWorker(
            model_size=config.whisper.model_size,
            device=config.whisper.device,
            compute_type=config.whisper.compute_type,
            beam_size=config.whisper.beam_size,
            language=config.whisper.language,
            initial_prompt=matcher.hotwords(config.whisper.vocabulary),
        )
        self.segmenter = VadSegmenter(
            frame_ms=config.audio.frame_ms,
            aggressiveness=config.vad.aggressiveness,
            silence_ms=config.vad.silence_ms,
            min_clip_ms=config.vad.min_clip_ms,
            max_clip_ms=config.vad.max_clip_ms,
            preroll_ms=config.vad.preroll_ms,
            trigger_ratio=config.vad.trigger_ratio,
        )

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self.stt.load()
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._capture is not None:
            self._capture.stop()
        if self._thread is not None:
            self._thread.join(timeout=5)

    # -- worker thread -----------------------------------------------------

    def _run(self) -> None:
        """Supervise the capture chain, restarting it when the device drops.

        A USB SDR replugged mid-net, or a loopback sink that disappears when the
        SDR app restarts, should cost a few seconds of the net -- not the rest
        of it. File replay never restarts: reaching the end is success, and
        retrying would loop the recording forever.
        """
        session_start = datetime.now()
        delay = self.config.health.restart_delay_s

        while not self._stop.is_set():
            try:
                frames = (
                    self._wav_frames(self.wav_path)
                    if self.wav_path
                    else self._live_frames()
                )
                for clip in self.segmenter.segment(self._watch(frames)):
                    if self._stop.is_set():
                        break
                    self._handle_clip(clip, session_start)
                self.health.capture_stopped()
                if self.wav_path or self._stop.is_set():
                    return
                log.warning("Audio stream ended unexpectedly")
            except AudioUnavailable as exc:
                # The dashboard stays up so the operator can read the error and
                # the session log so far; only capture is down.
                self.health.capture_failed(str(exc))
                log.error("%s", exc)
                if self.wav_path:
                    return
            except Exception as exc:
                self.health.capture_failed(f"{type(exc).__name__}: {exc}")
                log.exception("Capture pipeline failed")
                if self.wav_path:
                    return

            if not self.config.health.restart_capture:
                log.error("Capture restart disabled; audio will stay down")
                return
            self._close_capture()
            if self._stop.wait(delay):
                return
            log.info("Restarting audio capture (retry in %.0fs if it fails)", delay)
            delay = min(delay * 2, self.config.health.restart_max_delay_s)

    def _watch(self, frames):
        """Pass frames through, reporting level and liveness to the monitor."""
        import numpy as np

        for index, frame in enumerate(frames):
            samples = np.frombuffer(frame, dtype=np.int16)
            rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))) if len(
                samples
            ) else 0.0
            self.health.note_frame(rms)
            if self._capture is not None and index % 100 == 0:
                self.health.note_overflows(self._capture.overflows)
            yield frame

    def _close_capture(self) -> None:
        if self._capture is not None:
            try:
                self._capture.stop()
            except Exception:  # pragma: no cover - best effort on a dying device
                log.debug("Error closing audio device", exc_info=True)
            self._capture = None

    def _live_frames(self):
        try:
            self._capture = AudioCapture(
                device=self.config.audio.device,
                frame_ms=self.config.audio.frame_ms,
                channel=self.config.audio.channel,
                gain=self.config.audio.gain,
            )
            self._capture.start()
        except Exception as exc:
            # Almost always the audio source, not the code: a wrong device name,
            # or in a container a Pulse socket that is missing or owned by a
            # different UID. Say so instead of printing a PortAudio traceback.
            raise AudioUnavailable(
                f"Could not open audio input "
                f"{self.config.audio.device or '(system default)'}: {exc}\n"
                "  - `python app.py --list-devices` shows what this host can see.\n"
                "  - Check SDR++/GQRX is running and feeding the loopback sink.\n"
                "  - In a container, check $XDG_RUNTIME_DIR/pulse is mounted and "
                "the container UID matches the host user's."
            ) from exc
        self.health.capture_started()
        log.info("Capturing from %s", self._capture.describe())
        return self._capture.frames()

    def _wav_frames(self, path: str):
        """Replay a recording, for tuning the VAD without going live."""
        import wave

        import numpy as np

        self.health.capture_started()
        with wave.open(path, "rb") as wav:
            if wav.getsampwidth() != 2:
                raise ValueError("WAV must be 16-bit PCM")
            rate = wav.getframerate()
            channels = wav.getnchannels()
            # Any rate: a net recorded on a phone or handheld recorder is 44.1
            # or 48 kHz, and asking the operator to convert it first is a good
            # way to have the tuning step skipped.
            resampler = Resampler(rate, TARGET_RATE)
            log.info(
                "Replaying %s: %s, %d ch", path, describe(rate, TARGET_RATE), channels
            )

            frame_bytes = int(TARGET_RATE * self.config.audio.frame_ms / 1000) * 2
            buffer = bytearray()
            finished = False

            while not self._stop.is_set():
                raw = wav.readframes(4096)
                if raw:
                    samples = np.frombuffer(raw, dtype=np.int16)
                    if channels > 1:
                        samples = samples.reshape(-1, channels)[:, 0]
                    buffer.extend(resampler.process(samples).tobytes())
                elif not finished:
                    buffer.extend(resampler.flush().tobytes())
                    finished = True

                while len(buffer) >= frame_bytes:
                    frame = bytes(buffer[:frame_bytes])
                    del buffer[:frame_bytes]
                    yield frame

                if finished:
                    return

    def _handle_clip(self, clip, session_start: datetime) -> None:
        started_at = session_start + timedelta(milliseconds=clip.start_offset_ms)
        self.health.note_clip()
        began = time.monotonic()
        try:
            transcription = self.stt.transcribe(clip.audio)
        except Exception as exc:
            # One bad clip must not take the pipeline down mid-net.
            self.health.note_error(f"transcription failed: {exc}")
            log.exception("Transcription failed for %.1fs clip", clip.duration_ms / 1000)
            return
        self.health.note_transcription(time.monotonic() - began)
        if not transcription.text:
            log.debug("Empty transcription for %.1fs clip", clip.duration_ms / 1000)
            return

        result = self.matcher.match(transcription.text)
        entry = self.store.add(
            started_at=started_at,
            matched=result.matched,
            matched_callsign=result.callsign,
            operator_name=result.name,
            raw_text=transcription.text,
            confidence=transcription.confidence,
            match_score=result.score,
            clip_duration=clip.duration_ms / 1000,
            candidate=result.candidate,
            unmatched_reason=result.reason,
            via_alias=result.via_alias,
        )
        log.info(
            "%s | %s | %s",
            entry.timestamp,
            entry.matched_callsign or f"unmatched({entry.candidate or '-'})",
            entry.raw_text,
        )
        asyncio.run_coroutine_threadsafe(
            self.broadcaster.broadcast({"type": "entry", "entry": entry.to_dict()}),
            self.loop,
        )


async def watchdog(
    config: Config,
    health: HealthMonitor,
    broadcaster: Broadcaster,
) -> None:
    """Poll health, announce changes, and log a periodic heartbeat.

    Alerting for this app means the dashboard and the log, not a pager: it runs
    offline, in a room with the one person who can act on it. So a state change
    goes to the log at a level matching its severity, and to every connected
    dashboard, which shows a banner and (optionally) beeps.
    """
    previous = OK
    last_heartbeat = time.monotonic()

    while True:
        await asyncio.sleep(config.health.check_interval_s)
        snapshot = health.snapshot()

        if snapshot.state != previous:
            if snapshot.state == ERROR:
                log.error("Pipeline unhealthy: %s", "; ".join(snapshot.issues))
            elif snapshot.state == WARNING:
                log.warning("Pipeline degraded: %s", "; ".join(snapshot.issues))
            else:
                log.info("Pipeline healthy again")
            previous = snapshot.state

        await broadcaster.broadcast({"type": "health", "health": snapshot.to_dict()})

        now = time.monotonic()
        if config.health.heartbeat_s and now - last_heartbeat >= config.health.heartbeat_s:
            last_heartbeat = now
            log.info(
                "Heartbeat: %s | up %.0fm | %d frames, %d clips, %d transcripts "
                "| level %.0f RMS | last transcribe %.2fs | %d dropped",
                snapshot.state,
                snapshot.uptime_s / 60,
                snapshot.frames,
                snapshot.clips,
                snapshot.transcriptions,
                snapshot.signal_rms,
                snapshot.last_transcribe_s,
                snapshot.overflows,
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ham radio net STT pipeline")
    parser.add_argument("--config", default="config.yaml", help="path to config YAML")
    parser.add_argument(
        "--list-devices", action="store_true", help="print audio devices and exit"
    )
    parser.add_argument(
        "--file", help="replay a 16-bit PCM WAV instead of capturing live audio"
    )
    parser.add_argument("--roster", help="override the roster CSV path")
    parser.add_argument("--port", type=int, help="override the web server port")
    parser.add_argument("--model", help="override the Whisper model size")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--no-log-file", action="store_true", help="console logging only"
    )
    return parser.parse_args(argv)


async def run(config: Config, wav_path: str | None) -> None:
    roster = load_roster(config.roster.path)
    log.info("Loaded %d roster entries from %s", len(roster), config.roster.path)

    feedback = FeedbackLog(config.roster.feedback_path)
    aliases = feedback.aliases() if config.roster.learn_aliases else {}
    matcher = CallsignMatcher(
        roster=roster,
        threshold=config.roster.threshold,
        ambiguity_margin=config.roster.ambiguity_margin,
        aliases=aliases,
    )
    if matcher.aliases:
        log.info(
            "Replayed %d learned alias(es) from %s",
            len(matcher.aliases),
            config.roster.feedback_path,
        )

    store = TranscriptStore()
    broadcaster = Broadcaster()
    health = HealthMonitor(
        stall_after_s=config.health.stall_after_s,
        silence_after_s=config.health.silence_after_s,
        silence_rms=config.health.silence_rms,
    )
    app = create_app(
        store,
        roster,
        broadcaster,
        export_dir=config.export_dir,
        matcher=matcher,
        feedback=feedback,
        health=health,
    )

    pipeline = Pipeline(
        config,
        store,
        matcher,
        broadcaster,
        asyncio.get_running_loop(),
        health,
        wav_path,
    )
    pipeline.start()
    watch = asyncio.create_task(watchdog(config, health, broadcaster))

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=config.server.host,
            port=config.server.port,
            log_level="warning",
            access_log=False,
        )
    )
    log.info(
        "Dashboard on http://%s:%d",
        "localhost" if config.server.host == "0.0.0.0" else config.server.host,
        config.server.port,
    )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: setattr(server, "should_exit", True))

    try:
        await server.serve()
    finally:
        watch.cancel()
        pipeline.stop()
        if store.entries:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            path = store.export_text(Path(config.export_dir) / f"net-log-{stamp}.txt")
            log.info("Session log written to %s", path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list_devices:
        logging.basicConfig(level=logging.INFO)
        print(list_devices())
        return 0

    config_path = args.config if Path(args.config).exists() else None
    if config_path is None and args.config != "config.yaml":
        raise SystemExit(f"Config file not found: {args.config}")
    config = load_config(config_path)

    if args.roster:
        config.roster.path = args.roster
    if args.port:
        config.server.port = args.port
    if args.model:
        config.whisper.model_size = args.model

    log_path = setup_logging(
        log_dir=None if args.no_log_file else config.logging.dir,
        level=config.logging.level,
        file_level=config.logging.file_level,
        max_bytes=config.logging.max_bytes,
        backups=config.logging.backups,
        verbose=args.verbose,
    )
    if log_path:
        log.info("Logging to %s (rotating, %d backups)", log_path, config.logging.backups)

    if not Path(config.roster.path).exists():
        raise SystemExit(
            f"Roster file not found: {config.roster.path}\n"
            "Copy roster.example.csv to roster.csv, or pass --roster."
        )

    try:
        asyncio.run(run(config, args.file))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
