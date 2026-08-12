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

Three threads, deliberately:

    PortAudio callback  -> ring buffer        (never allocates, never blocks)
    capture thread      -> VAD -> clip queue  (cheap; always keeps up)
    STT thread          -> Whisper -> store   (slow; allowed to fall behind)

The split is what keeps audio from being lost on a slow box. With VAD and
Whisper on one thread, nothing drained the audio buffer during a transcription,
so a long clip on a Pi meant the *next* transmission was dropped mid-word.
Now a slow transcriber makes transcripts arrive late -- and if even the clip
queue fills, clips spill to disk rather than being discarded.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import queue
import signal
import sys
import threading
import time
from datetime import datetime, timedelta
from types import SimpleNamespace
from pathlib import Path

import uvicorn

from audio_capture import TARGET_RATE, AudioCapture, list_devices
import collections

from callsign_match import CallsignMatcher, load_roster
from clip_split import Segment, split_transmissions
from clip_spill import SpillStore
from config import Config, SourceConfig, audio_sources, load_config
from feedback import FeedbackLog
from health import ERROR, OK, WARNING, HealthFleet, HealthMonitor
from logging_setup import setup_logging
from resample import Resampler, describe
import attendance as attendance_history
import settings as settings_registry
from server import Broadcaster, create_app
from session_writer import SessionWriter, latest_session, read_session
from stt_worker import SttWorker
import traffic as traffic_detector
from transcript_store import TranscriptStore
from vad_segmenter import VadSegmenter
from voice_id import EnrolmentAudio, VoiceProfiles

log = logging.getLogger("net-stt")


def _segment_audio(clip, segment):
    """The slice of clip audio belonging to one segment.

    A second pass has to re-hear the same audio, and after a split that is a
    portion of the clip rather than all of it -- feeding the whole thing back
    would re-transcribe the other station too.
    """
    audio = getattr(clip, "audio", None)
    if audio is None:
        return None
    samples_per_ms = TARGET_RATE // 1000
    start = max(0, segment.start_offset_ms * samples_per_ms)
    end = min(len(audio), start + segment.duration_ms * samples_per_ms)
    if end <= start:
        return audio
    return audio[start:end]


STOP_SENTINEL = (-(10**9), -1, None)
"""Ordered ahead of every real clip, so stop() is never queued behind a backlog."""


class AudioUnavailable(RuntimeError):
    """The capture device could not be opened; the message is operator-facing."""


class SourceCapture:
    """One receiver: device -> ring buffer -> VAD -> the shared clip queue.

    Each source runs its own thread and its own health, so a dead simplex
    receiver does not make the repeater look broken -- and the operator can see
    which one to go and fix.
    """

    def __init__(
        self,
        source: SourceConfig,
        config: Config,
        health: HealthMonitor,
        submit,
        session_start: datetime,
        wav_path: str | None = None,
    ) -> None:
        self.source = source
        self.config = config
        self.health = health
        self.submit = submit
        self.session_start = session_start
        # A per-source recording wins over the global --file, so a two-receiver
        # setup can be replayed from two files at once.
        self.wav_path = source.file or wav_path

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture: AudioCapture | None = None
        self.segmenter = VadSegmenter(
            frame_ms=config.audio.frame_ms, **source.vad_settings(config.vad)
        )

    @property
    def name(self) -> str:
        return self.source.name

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name=f"capture:{self.name}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._capture is not None:
            self._capture.stop()
        if self._thread is not None:
            self._thread.join(timeout=5)

    # -- worker thread -----------------------------------------------------

    def _run(self) -> None:
        """Supervise this source, restarting it when its device drops.

        A USB SDR replugged mid-net should cost a few seconds of one channel --
        not the rest of the net, and not the other receivers. File replay never
        restarts: reaching the end is success, and retrying would loop forever.
        """
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
                    clip.source = self.name
                    clip.priority = self.source.priority
                    self.submit(clip)
                if self.wav_path or self._stop.is_set():
                    self.health.capture_finished()
                    return
                self.health.capture_stopped()
                log.warning("[%s] Audio stream ended unexpectedly", self.name)
            except AudioUnavailable as exc:
                # The dashboard stays up so the operator can read the error and
                # the session log so far; only this source is down.
                self.health.capture_failed(str(exc))
                log.error("[%s] %s", self.name, exc)
                if self.wav_path:
                    return
            except Exception as exc:
                self.health.capture_failed(f"{type(exc).__name__}: {exc}")
                log.exception("[%s] Capture failed", self.name)
                if self.wav_path:
                    return

            if not self.config.health.restart_capture:
                log.error("[%s] Restart disabled; this source stays down", self.name)
                return
            self._close_capture()
            if self._stop.wait(delay):
                return
            log.info(
                "[%s] Restarting capture (retry in %.0fs if it fails)",
                self.name,
                delay,
            )
            delay = min(delay * 2, self.config.health.restart_max_delay_s)

    def _watch(self, frames):
        """Pass frames through, reporting level and liveness to the monitor."""
        import numpy as np

        for index, frame in enumerate(frames):
            samples = np.frombuffer(frame, dtype=np.int16)
            rms = (
                float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
                if len(samples)
                else 0.0
            )
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
                device=self.source.device,
                frame_ms=self.config.audio.frame_ms,
                channel=self.source.channel,
                gain=self.source.gain,
                buffer_seconds=self.config.buffering.ring_seconds,
            )
            self._capture.start()
        except Exception as exc:
            # Almost always the audio source, not the code: a wrong device name,
            # or in a container a Pulse socket that is missing or owned by a
            # different UID. Say so instead of printing a PortAudio traceback.
            raise AudioUnavailable(
                f"Could not open audio input "
                f"{self.source.device or '(system default)'}: {exc}\n"
                "  - `python app.py --list-devices` shows what this host can see.\n"
                "  - Check the receiver is running and feeding the loopback sink.\n"
                "  - In a container, check $XDG_RUNTIME_DIR/pulse is mounted and "
                "the container UID matches the host user's."
            ) from exc
        self.health.capture_started()
        log.info("[%s] Capturing from %s", self.name, self._capture.describe())
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
                "[%s] Replaying %s: %s, %d ch",
                self.name,
                path,
                describe(rate, TARGET_RATE),
                channels,
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


class Pipeline:
    """All configured sources, feeding one shared transcriber.

    One Whisper model, deliberately. It is the memory-hungry part, and two
    receivers on a Pi would thrash rather than go faster; serialising them
    costs a little latency on a busy net and nothing at all on a quiet one.
    """

    def __init__(
        self,
        config: Config,
        store: TranscriptStore,
        matcher: CallsignMatcher,
        broadcaster: Broadcaster,
        loop: asyncio.AbstractEventLoop,
        fleet: HealthFleet,
        wav_path: str | None = None,
        session: SessionWriter | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.matcher = matcher
        self.broadcaster = broadcaster
        self.loop = loop
        self.fleet = fleet
        self.session = session
        self._stop = threading.Event()
        self._stt_thread: threading.Thread | None = None
        self._sequence = 0
        self._sequence_lock = threading.Lock()
        self._session_start = datetime.now()

        # Bounded on purpose. Past this depth the backlog goes to disk, where it
        # is not competing for memory on a 2 GB Pi.
        # Priority queue: when the transcriber is behind, the repeater's traffic
        # should reach the log before a staging channel's. Ties break on
        # sequence, so within one priority it stays first-in-first-out.
        self._clips: queue.PriorityQueue = queue.PriorityQueue(
            maxsize=config.buffering.clip_queue_max
        )
        self.spill = SpillStore(
            config.buffering.spill_dir, max_clips=config.buffering.spill_max_clips
        )

        self.stt = SttWorker(
            model_size=config.whisper.model_size,
            device=config.whisper.device,
            compute_type=config.whisper.compute_type,
            beam_size=config.whisper.beam_size,
            language=config.whisper.language,
            condition_audio=config.whisper.condition_audio,
            prompt_token_budget=config.whisper.prompt_token_budget,
        )
        # A second, larger model for clips the first pass could not resolve.
        # Loaded lazily: it is real memory, and a net may never need it.
        self._escalator: SttWorker | None = None
        self._escalate: collections.deque = collections.deque(
            maxlen=config.escalation.max_pending
        )
        self._prompts: dict[str, str] = {}
        self._prompt_generation = -1
        self._pending_model: str | None = None
        """A model change requested from the dashboard, applied between clips."""

        # Who has actually turned up before, from the sessions already logged.
        self.attendance = attendance_history.Attendance()

        # Voices, learned from clean matches and from operator corrections.
        self.voices = VoiceProfiles(
            path=config.voice.path,
            min_similarity=config.voice.min_similarity,
            margin=config.voice.margin,
            min_enrolments=config.voice.min_enrolments,
            audio=(
                EnrolmentAudio(
                    config.voice.audio_dir,
                    per_station=config.voice.audio_per_station,
                    max_seconds=config.voice.audio_max_seconds,
                )
                if config.voice.keep_audio
                else None
            ),
        )
        # Recent clip audio, so a correction arriving a minute later can still
        # enrol the voice it belongs to.
        self._recent_audio: collections.OrderedDict = collections.OrderedDict()

        self.sources = [
            SourceCapture(
                source=source,
                config=config,
                health=fleet.monitor(source.name),
                submit=self._enqueue,
                session_start=self._session_start,
                wav_path=wav_path,
            )
            for source in audio_sources(config)
        ]

    @property
    def multi_source(self) -> bool:
        return len(self.sources) > 1

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self.stt.load()
        self.attendance = attendance_history.load(
            self.config.transcripts.dir, {e.callsign for e in self.matcher.roster}
        )
        if self.attendance.records:
            log.info("Attendance: %s", attendance_history.summary(self.attendance))
        if self.attendance.unknown:
            # Reported, never adopted: a mis-transcription promoted to a
            # station would bias decoding toward its own mistake.
            log.info(
                "Heard before but not on the roster: %s",
                ", ".join(self.attendance.unknown[:8]),
            )
        if self.config.voice.enabled:
            known = self.voices.load()
            if known:
                log.info("Loaded %d voice profile(s) from %s", known, self.config.voice.path)
        if self.config.buffering.spill_enabled:
            cleared = self.spill.clear()
            if cleared:
                log.info("Cleared %d clip(s) left over from a previous run", cleared)
        self._stt_thread = threading.Thread(
            target=self._transcribe_loop, name="stt", daemon=True
        )
        self._stt_thread.start()
        for source in self.sources:
            source.start()
        log.info(
            "Listening on %d source(s): %s",
            len(self.sources),
            ", ".join(s.name for s in self.sources),
        )

    def stop(self) -> None:
        self._stop.set()
        for source in self.sources:
            source.stop()
        # Sentinel must be comparable with the (priority, sequence, clip)
        # tuples in the queue, and must sort ahead of them so shutdown is not
        # stuck behind a backlog.
        if self.config.voice.enabled and self.voices.profiles:
            if self.voices.save():
                log.info(
                    "Saved %d voice profile(s) to %s",
                    len(self.voices.profiles),
                    self.config.voice.path,
                )
        self._clips.put(STOP_SENTINEL)
        if self._stt_thread is not None:
            self._stt_thread.join(timeout=5)

    def drain(self, timeout: float = 30.0) -> int:
        """Wait for the backlog to finish. Used at shutdown so a spilled clip
        still makes it into the exported log."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._clips.empty() and self.spill.pending() == 0:
                return 0
            time.sleep(0.2)
        return self._clips.qsize() + self.spill.pending()

    # -- settings changed while running ------------------------------------

    def apply_setting(self, path: str, value) -> None:
        """Make a config change take effect now.

        Several components keep their own copy of a setting -- the matcher, each
        source's segmenter, the open audio device -- because reading through the
        config on every frame would be silly. So a live change has to be pushed
        to them, and this is the one place that knows where those copies are.
        """
        settings_registry.set_value(self.config, path, value)

        if path == "roster.threshold":
            self.matcher.threshold = float(value)
        elif path == "roster.ambiguity_margin":
            self.matcher.ambiguity_margin = float(value)
        elif path.startswith("vad."):
            key = path.split(".", 1)[1]
            for source in self.sources:
                # Per-source overrides win: a change to the global default must
                # not quietly overwrite a receiver that was set deliberately.
                if getattr(source.source, key, None) is None:
                    setattr(source.segmenter, key, value)
        elif path.startswith("sources.") and path.endswith(".gain"):
            name = path.split(".")[1]
            for source in self.sources:
                if source.name == name and source._capture is not None:
                    source._capture.gain = float(value)
        elif path == "whisper.model_size":
            # Handed to the STT thread rather than swapped here: another thread
            # is inside the model right now.
            self._pending_model = str(value)
        elif path == "whisper.beam_size":
            self.stt.beam_size = int(value)
            if self._escalator is not None:
                self._escalator.beam_size = int(value)
        elif path == "escalation.model_size":
            self._escalator = None  # reloaded lazily at its next use
        elif path == "voice.enabled" and value and not self.voices.profiles:
            self.voices.load()

        log.info("Setting changed: %s = %s", path, value)

    def _apply_pending_model(self) -> None:
        """Swap the live model, on the thread that owns it."""
        pending, self._pending_model = self._pending_model, None
        if not pending or pending == self.stt.model_size:
            return
        log.info("Switching live model to %s; audio buffers while it loads", pending)
        try:
            self.stt.reload(pending)
            self._prompts.clear()  # rebuilt against the new tokenizer
            log.info("Live model now %s", pending)
        except Exception as exc:
            self.fleet.note_error(f"could not load {pending}: {exc}")
            log.exception("Could not switch to %s; keeping the current model", pending)

    # -- prompts -----------------------------------------------------------

    def _prompt_for(self, source: str) -> str:
        """Whisper prompt for this receiver, rebuilt as the net progresses.

        Cached until the set of stations heard changes: the ordering depends on
        who has already checked in, and rebuilding it counts tokens.
        """
        heard = set(self.store.check_ins())
        if len(heard) != self._prompt_generation:
            self._prompts.clear()
            self._prompt_generation = len(heard)
        if source not in self._prompts:
            terms = self.matcher.bias_terms(
                self.config.whisper.vocabulary,
                source=source,
                heard=heard,
                attendance=self.attendance.for_source(source),
            )
            self._prompts[source] = self.stt.build_prompt(terms)
            log.debug(
                "[%s] prompt carries %d of %d bias terms",
                source or "main",
                self.stt.prompt_terms_used,
                self.stt.prompt_terms_offered,
            )
        return self._prompts[source]

    # -- clip queue, spill, and transcription ------------------------------

    def _enqueue(self, clip) -> None:
        """Hand a clip to the STT thread, spilling to disk if it is behind.

        Called from every source thread, so the sequence counter is locked.
        """
        with self._sequence_lock:
            self._sequence += 1
            clip.sequence = self._sequence

        try:
            self._clips.put_nowait((-clip.priority, clip.sequence, clip))
            return
        except queue.Full:
            pass

        if not self.config.buffering.spill_enabled:
            self.fleet.note_error("clip dropped: transcriber is too far behind")
            log.error(
                "[%s] Clip queue full and spilling disabled -- dropped %.1fs of audio",
                clip.source,
                clip.duration_ms / 1000,
            )
            return

        path = self.spill.write(
            clip.audio,
            clip.start_offset_ms,
            clip.duration_ms,
            clip.sequence,
            source=clip.source,
        )
        if path is None:
            self.fleet.note_error("clip lost: could not spill to disk")
            return
        self.fleet.note_spill(self.spill.spilled, self.spill.pending())
        log.warning(
            "[%s] Transcriber is behind; spilled clip %d (%.1fs) to %s",
            clip.source,
            clip.sequence,
            clip.duration_ms / 1000,
            path.name,
        )

    # -- voices ------------------------------------------------------------

    def _remember_audio(self, entry_id: int, audio) -> None:
        if not self.config.voice.enabled or audio is None:
            return
        self._recent_audio[entry_id] = audio
        while len(self._recent_audio) > self.config.voice.recent_audio:
            self._recent_audio.popitem(last=False)

    def _voice_pass(self, entry, audio) -> None:
        """Learn from a clean match, or suggest a name for an unmatched line."""
        if not self.config.voice.enabled or audio is None:
            return

        if entry.matched and entry.matched_callsign:
            # Only learn from matches good enough to be trusted as labels; a
            # profile built from a wrong match poisons every later suggestion.
            if entry.match_score >= self.config.voice.enrol_min_score:
                self.voices.enrol(entry.matched_callsign, audio)
            return

        suggestion = self.voices.identify(audio)
        if suggestion is None:
            return
        self.store.suggest(entry.id, suggestion.callsign, suggestion.score)
        log.info(
            "Voice suggests %s for entry %d (%.2f, next best %s %.2f)",
            suggestion.callsign,
            entry.id,
            suggestion.score,
            suggestion.runner_up or "-",
            suggestion.runner_up_score,
        )

    def enrol_from_correction(self, entry_id: int, callsign: str) -> bool:
        """Learn a voice from a line the operator just fixed.

        The best labels available: a human listened and said whose it was.
        """
        if not self.config.voice.enabled:
            return False
        audio = self._recent_audio.get(entry_id)
        if audio is None:
            return False
        learned = self.voices.enrol(callsign, audio)
        if learned:
            log.info("Learned %s's voice from correction of entry %d", callsign, entry_id)
        return learned

    def _maybe_escalate(self, entry, audio) -> None:
        """Queue a line for a second, better pass when the first was unsure."""
        settings = self.config.escalation
        if not settings.enabled or entry.escalated or audio is None:
            return
        unsure = (not entry.matched and settings.on_unmatched) or (
            entry.matched and entry.confidence < settings.min_confidence
        )
        if not unsure:
            return
        if len(self._escalate) == self._escalate.maxlen:
            log.warning("Escalation queue full; dropping the oldest waiting clip")
        self._escalate.append((entry.id, audio, entry.candidate, entry.source))

    def _escalate_one(self) -> bool:
        """Re-transcribe one queued clip. Returns whether there was work."""
        if not self._escalate:
            return False
        entry_id, audio, candidate, source = self._escalate.popleft()
        if audio is None:
            return False

        worker = self._escalation_worker()
        # The targeted part: bias toward the handful of roster entries the
        # first pass was already near, rather than a roster that cannot fit.
        terms = self.matcher.nearest(candidate or "") + self.matcher.bias_terms(
            self.config.whisper.vocabulary,
            source=source,
            heard=set(self.store.check_ins()),
            attendance=self.attendance.for_source(source),
        )
        try:
            better = worker.transcribe(audio, prompt=worker.build_prompt(terms))
        except Exception as exc:
            self.fleet.note_error(f"escalation failed: {exc}")
            log.exception("Escalation pass failed for entry %d", entry_id)
            return True
        if not better.text:
            return True

        result = self.matcher.match(better.text)
        entry = self.store.get(entry_id)
        if entry is None:
            return True
        # Only replace the line if the second pass actually did better; a
        # bigger model is not automatically right, and churning a line net
        # control has already read costs more than it gains.
        was = entry.matched_callsign or "unmatched"
        improved = (result.matched and not entry.matched) or (
            result.matched and result.score > entry.match_score
        )
        if not improved:
            log.debug("Escalation gained nothing for entry %d", entry_id)
            return True

        updated = self.store.improve(
            entry_id,
            raw_text=better.text,
            matched=result.matched,
            matched_callsign=result.callsign,
            operator_name=result.name,
            position=result.position,
            confidence=better.confidence,
            match_score=result.score,
            candidate=result.candidate,
            unmatched_reason=result.reason,
        )
        if updated is None:  # an operator already fixed it by hand
            return True

        log.info(
            "Second pass improved entry %d: %s -> %s",
            entry_id,
            was,  # captured before improve() mutated the entry in place
            updated.matched_callsign,
        )
        if self.session is not None:
            self.session.append(updated)
        asyncio.run_coroutine_threadsafe(
            self.broadcaster.broadcast(
                {"type": "correction", "entry": updated.to_dict(), "learned": False}
            ),
            self.loop,
        )
        return True

    def _escalation_worker(self) -> SttWorker:
        if self._escalator is None:
            settings = self.config.escalation
            log.info("Loading %s for second-pass transcription", settings.model_size)
            self._escalator = SttWorker(
                model_size=settings.model_size,
                device=settings.device,
                compute_type=settings.compute_type,
                beam_size=self.config.whisper.beam_size,
                language=self.config.whisper.language,
                condition_audio=self.config.whisper.condition_audio,
                prompt_token_budget=self.config.whisper.prompt_token_budget,
            )
            self._escalator.load()
        return self._escalator

    def _transcribe_loop(self) -> None:
        """Drain the clip queue, then any spilled backlog, forever.

        Live clips come first: during a net, the line that matters is the one
        being spoken now. Spilled clips are picked up whenever the queue runs
        dry -- a lull between check-ins, or after the net ends.
        """
        while not self._stop.is_set():
            if self._pending_model:
                self._apply_pending_model()
            try:
                item = self._clips.get(timeout=0.5)
            except queue.Empty:
                # Live traffic first, then the disk backlog, then second passes:
                # improving an old line must never delay the current one.
                if not self._drain_one_spilled():
                    self._escalate_one()
                continue
            if item[2] is None:
                return
            self._handle_clip(item[2], late=False)

    def _drain_one_spilled(self) -> bool:
        """Transcribe one spilled clip. Returns whether there was work."""
        if not self.config.buffering.spill_enabled:
            return False
        spilled = self.spill.read_oldest()
        if spilled is None:
            return False
        clip = SimpleNamespace(
            audio=spilled.audio,
            start_offset_ms=spilled.start_offset_ms,
            duration_ms=spilled.duration_ms,
            sequence=spilled.sequence,
            source=spilled.source,
        )
        log.info("Catching up: transcribing spilled clip %d", spilled.sequence)
        self._handle_clip(clip, late=True)
        self.fleet.note_spill(self.spill.spilled, self.spill.pending())
        return True

    def _handle_clip(self, clip, late: bool = False) -> None:
        started_at = self._session_start + timedelta(milliseconds=clip.start_offset_ms)
        health = self.fleet.monitor(clip.source or self.sources[0].name)
        health.note_clip()
        began = time.monotonic()
        try:
            transcription = self.stt.transcribe(
                clip.audio, prompt=self._prompt_for(getattr(clip, "source", ""))
            )
        except Exception as exc:
            # One bad clip must not take the pipeline down mid-net.
            health.note_error(f"transcription failed: {exc}")
            log.exception("Transcription failed for %.1fs clip", clip.duration_ms / 1000)
            return
        health.note_transcription(time.monotonic() - began, self._clips.qsize())
        if not transcription.text:
            log.debug("Empty transcription for %.1fs clip", clip.duration_ms / 1000)
            return

        # Usually one transmission per clip. On a fast net two stations key up
        # inside the VAD's silence window and land in the same one.
        try:
            segments = self._segments(clip, transcription)
        except Exception as exc:
            health.note_error(f"split failed: {exc}")
            log.exception("Could not split clip; logging it as one transmission")
            segments = [
                Segment(transcription.text, 0, clip.duration_ms)
            ]
        for segment in segments:
            self._log_transmission(clip, segment, started_at, transcription, late)

    def _segments(self, clip, transcription) -> list[Segment]:
        whole = [
            Segment(
                text=transcription.text,
                start_offset_ms=0,
                duration_ms=clip.duration_ms,
            )
        ]
        if not self.config.split.enabled:
            return whole
        return split_transmissions(
            transcription.text,
            transcription.words,
            self.matcher.match_all(transcription.text),
            clip.duration_ms,
            min_gap_ms=self.config.split.min_gap_ms,
            min_segment_ms=self.config.split.min_segment_ms,
        )

    def _log_transmission(
        self, clip, segment: Segment, clip_started_at, transcription, late: bool
    ) -> None:
        started_at = clip_started_at + timedelta(milliseconds=segment.start_offset_ms)
        result = self.matcher.match(segment.text)
        entry = self.store.add(
            started_at=started_at,
            matched=result.matched,
            matched_callsign=result.callsign,
            operator_name=result.name,
            position=result.position,
            traffic=(
                traffic_detector.detect(segment.text)
                if self.config.traffic.detect
                else ""
            ),
            raw_text=segment.text,
            confidence=transcription.confidence,
            match_score=result.score,
            clip_duration=segment.duration_ms / 1000,
            candidate=result.candidate,
            unmatched_reason=result.reason,
            via_alias=result.via_alias,
            late=late,
            source=clip.source if self.multi_source else "",
        )
        audio = _segment_audio(clip, segment)
        self._remember_audio(entry.id, audio)
        self._voice_pass(entry, audio)
        if self.session is not None:
            self.session.append(entry)
        self._maybe_escalate(entry, audio)
        log.info(
            "%s |%s %s | %s%s",
            entry.timestamp,
            f" [{clip.source}]" if self.multi_source else "",
            entry.matched_callsign or f"unmatched({entry.candidate or '-'})",
            entry.raw_text,
            " [late]" if late else "",
        )
        asyncio.run_coroutine_threadsafe(
            self.broadcaster.broadcast({"type": "entry", "entry": entry.to_dict()}),
            self.loop,
        )


async def watchdog(
    config: Config,
    fleet: HealthFleet,
    broadcaster: Broadcaster,
) -> None:
    """Poll health, announce changes, and log a periodic heartbeat.

    Alerting for this app means the dashboard and the log, not a pager: it runs
    offline, in a room with the one person who can act on it. So a state change
    goes to the log at a level matching its severity, and to every connected
    dashboard, which shows a banner and (optionally) beeps.
    """
    previous = OK
    started = time.monotonic()
    last_heartbeat = started

    while True:
        await asyncio.sleep(config.health.check_interval_s)
        snapshot = fleet.snapshot()

        if snapshot["state"] != previous:
            if snapshot["state"] == ERROR:
                log.error("Pipeline unhealthy: %s", "; ".join(snapshot["issues"]))
            elif snapshot["state"] == WARNING:
                log.warning("Pipeline degraded: %s", "; ".join(snapshot["issues"]))
            else:
                log.info("Pipeline healthy again")
            previous = snapshot["state"]

        await broadcaster.broadcast({"type": "health", "health": snapshot})

        now = time.monotonic()
        if config.health.heartbeat_s and now - last_heartbeat >= config.health.heartbeat_s:
            last_heartbeat = now
            log.info(
                "Heartbeat: %s | up %.0fm | %d frames, %d clips, %d transcripts "
                "| level %.0f RMS | last transcribe %.2fs | backlog %d"
                "%s | %d dropped",
                snapshot["state"],
                (time.monotonic() - started) / 60,
                snapshot["frames"],
                snapshot["clips"],
                snapshot["transcriptions"],
                snapshot["signal_rms"],
                max(
                    (s["last_transcribe_s"] for s in snapshot["sources"].values()),
                    default=0.0,
                ),
                max((s["backlog"] for s in snapshot["sources"].values()), default=0),
                f" (+{snapshot['spill_pending']} on disk)"
                if snapshot["spill_pending"]
                else "",
                snapshot["overflows"],
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
    parser.add_argument(
        "--batch",
        action="store_true",
        help="with --file: process the recording, write the transcript, exit",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        metavar="SESSION",
        help="continue an interrupted net: reload its log and keep writing to "
        "it. Defaults to the most recent session.",
    )
    return parser.parse_args(argv)


async def run(
    config: Config,
    wav_path: str | None,
    batch: bool = False,
    resume: str | None = None,
    config_path: str | None = None,
) -> None:
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

    # Resuming: reload the interrupted log and keep writing to the same files,
    # so a crash mid-net leaves one record rather than two halves.
    resume_path = None
    if resume:
        resume_path = (
            latest_session(config.transcripts.dir)
            if resume == "latest"
            else Path(resume)
        )
        if resume_path is None or not resume_path.exists():
            log.warning(
                "Nothing to resume from in %s; starting a new session",
                config.transcripts.dir,
            )
            resume_path = None
        else:
            restored = store.restore(read_session(resume_path))
            log.info("Resumed %d line(s) from %s", restored, resume_path.name)
            if restored:
                log.info(
                    "  %d station(s) already checked in: %s",
                    len(store.check_ins()),
                    ", ".join(store.check_ins()),
                )

    session = None
    if config.transcripts.live:
        session = SessionWriter(
            config.transcripts.dir,
            store,
            fsync=config.transcripts.fsync,
            resume=resume_path,
        )
        path = session.start()
        if path:
            verb = "Continuing" if resume_path else "Writing this session to"
            log.info("%s %s (and the .txt beside it)", verb, path)
    fleet = HealthFleet(
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
        health=fleet,
        sources=[s.name for s in audio_sources(config)],
        session=session,
        acknowledge_traffic=config.traffic.detect and config.traffic.acknowledge,
        config=config,
        config_path=str(config_path) if config_path else None,
    )

    pipeline = Pipeline(
        config,
        store,
        matcher,
        broadcaster,
        asyncio.get_running_loop(),
        fleet,
        wav_path,
        session=session,
    )
    # The correction endpoint learns the voice as well as the alias, and the
    # settings endpoint needs the pipeline to push changes into the components
    # that hold their own copies.
    app.state.enrol_voice = pipeline.enrol_from_correction
    app.state.apply_setting = pipeline.apply_setting
    pipeline.start()
    watch = asyncio.create_task(watchdog(config, fleet, broadcaster))

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

    if batch:
        # Replaying a recording to build up training data: finish the file,
        # drain the backlog, and get out. Without this the dashboard keeps
        # serving forever, which makes the whole loop unscriptable.
        async def finish() -> None:
            while any(source._thread and source._thread.is_alive()
                      for source in pipeline.sources):
                await asyncio.sleep(0.2)
            await asyncio.to_thread(pipeline.drain, config.buffering.drain_timeout_s)
            log.info("Batch run complete: %d line(s)", len(store.entries))
            server.should_exit = True

        asyncio.create_task(finish())

    try:
        await server.serve()
    finally:
        watch.cancel()
        remaining = pipeline.drain(timeout=config.buffering.drain_timeout_s)
        if remaining:
            log.warning(
                "Shutting down with %d clip(s) still untranscribed; they stay in %s",
                remaining,
                config.buffering.spill_dir,
            )
        pipeline.stop()
        if session is not None:
            session.close()
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
        asyncio.run(
            run(
                config,
                args.file,
                batch=args.batch,
                resume=args.resume,
                config_path=config_path,
            )
        )
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
