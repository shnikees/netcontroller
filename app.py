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
from datetime import datetime, timedelta
from pathlib import Path

import uvicorn

from audio_capture import TARGET_RATE, AudioCapture, list_devices
from callsign_match import CallsignMatcher, load_roster
from config import Config, load_config
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
        wav_path: str | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.matcher = matcher
        self.broadcaster = broadcaster
        self.loop = loop
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
        try:
            frames = (
                self._wav_frames(self.wav_path)
                if self.wav_path
                else self._live_frames()
            )
            session_start = datetime.now()
            for clip in self.segmenter.segment(frames):
                if self._stop.is_set():
                    break
                self._handle_clip(clip, session_start)
        except AudioUnavailable as exc:
            # The dashboard stays up so the operator can read the error and the
            # session log so far; only capture is dead.
            log.error("%s", exc)
        except Exception:
            log.exception("Capture pipeline stopped")

    def _live_frames(self):
        try:
            self._capture = AudioCapture(
                device=self.config.audio.device, frame_ms=self.config.audio.frame_ms
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
        log.info(
            "Capturing from %s at %d Hz",
            self.config.audio.device or "system default",
            self._capture.samplerate,
        )
        return self._capture.frames()

    def _wav_frames(self, path: str):
        """Replay a recording, for tuning the VAD without going live."""
        import wave

        import numpy as np

        with wave.open(path, "rb") as wav:
            if wav.getsampwidth() != 2:
                raise ValueError("WAV must be 16-bit PCM")
            rate = wav.getframerate()
            if rate % TARGET_RATE != 0:
                raise ValueError(f"WAV rate {rate} is not a multiple of {TARGET_RATE}")
            ratio = rate // TARGET_RATE
            channels = wav.getnchannels()
            chunk = int(TARGET_RATE * self.config.audio.frame_ms / 1000)
            while not self._stop.is_set():
                raw = wav.readframes(chunk * ratio)
                if len(raw) < chunk * ratio * 2 * channels:
                    return
                samples = np.frombuffer(raw, dtype=np.int16)
                if channels > 1:
                    samples = samples.reshape(-1, channels)[:, 0]
                if ratio > 1:
                    samples = (
                        samples.reshape(-1, ratio).mean(axis=1).astype(np.int16)
                    )
                yield samples.tobytes()

    def _handle_clip(self, clip, session_start: datetime) -> None:
        started_at = session_start + timedelta(milliseconds=clip.start_offset_ms)
        transcription = self.stt.transcribe(clip.audio)
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
    return parser.parse_args(argv)


async def run(config: Config, wav_path: str | None) -> None:
    roster = load_roster(config.roster.path)
    log.info("Loaded %d roster entries from %s", len(roster), config.roster.path)
    matcher = CallsignMatcher(
        roster=roster,
        threshold=config.roster.threshold,
        ambiguity_margin=config.roster.ambiguity_margin,
    )
    store = TranscriptStore()
    broadcaster = Broadcaster()
    app = create_app(store, roster, broadcaster, export_dir=config.export_dir)

    pipeline = Pipeline(
        config, store, matcher, broadcaster, asyncio.get_running_loop(), wav_path
    )
    pipeline.start()

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
        pipeline.stop()
        if store.entries:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            path = store.export_text(Path(config.export_dir) / f"net-log-{stamp}.txt")
            log.info("Session log written to %s", path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.list_devices:
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
