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

"""YAML config loading, with every key overridable by an env var.

Env vars are named NETSTT_<SECTION>_<KEY>, e.g. NETSTT_WHISPER_MODEL_SIZE=small.
This keeps the container deployment configurable without baking a config file
into the image.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

ENV_PREFIX = "NETSTT"


@dataclass
class AudioConfig:
    device: str | None = None
    """Name substring or index of the input; null uses the system default.

    Works with any input: an SDR loopback monitor source, a USB sound card or
    line-in fed from a radio's speaker output, or a microphone.
    """
    frame_ms: int = 30
    channel: str = "mix"
    """mix, left, right, or a 0-based index. Use one channel when a stereo
    line-in carries the radio on a single side."""
    gain: float = 1.0
    """Linear multiplier applied at capture, for inputs that are too quiet."""


@dataclass
class SourceConfig:
    """One receiver feeding the app.

    A net often runs on more than one frequency at once -- the repeater plus a
    simplex staging channel, say -- and net control needs both in one log.
    Each source is an independent receiver: its own device, its own level, its
    own health.
    """

    name: str = "Main"
    """Shown on every line this source produces. Keep it short: 'Repeater'."""
    device: str | None = None
    channel: str = "mix"
    gain: float = 1.0
    enabled: bool = True
    """Set false to keep a source configured but not opened tonight."""
    file: str | None = None
    """Replay a recording instead of opening a device. For tuning a
    multi-receiver setup offline, the way --file does for a single one."""
    priority: int = 0
    """Higher goes first when the transcriber is behind. Put the repeater --
    the frequency the net actually runs on -- above a staging channel, so a
    backlog delays the side traffic rather than the main log."""

    # VAD overrides; None inherits the global `vad:` block. Receivers differ:
    # a strong repeater can take an aggressive setting that would deafen a weak
    # simplex signal, which is the whole reason these are per-source.
    aggressiveness: int | None = None
    silence_ms: int | None = None
    min_clip_ms: int | None = None
    preroll_ms: int | None = None
    trigger_ratio: float | None = None

    def vad_settings(self, defaults: "VadConfig") -> dict:
        """This source's VAD settings, falling back to the global block."""
        return {
            "aggressiveness": _or(self.aggressiveness, defaults.aggressiveness),
            "silence_ms": _or(self.silence_ms, defaults.silence_ms),
            "min_clip_ms": _or(self.min_clip_ms, defaults.min_clip_ms),
            "max_clip_ms": defaults.max_clip_ms,
            "preroll_ms": _or(self.preroll_ms, defaults.preroll_ms),
            "trigger_ratio": _or(self.trigger_ratio, defaults.trigger_ratio),
        }


def _or(value, fallback):
    """Fall back when a per-source override is not set."""
    return fallback if value is None else value


@dataclass
class VadConfig:
    aggressiveness: int = 3
    silence_ms: int = 800
    min_clip_ms: int = 400
    max_clip_ms: int = 120_000
    preroll_ms: int = 300
    trigger_ratio: float = 0.7


@dataclass
class SplitConfig:
    """Splitting a clip that caught two stations keying up back to back."""

    enabled: bool = True
    min_gap_ms: int = 500
    """Dead air between two callsigns before they count as two transmissions.
    Must sit below vad.silence_ms (or the VAD would have split them already)
    and above the pauses inside one person's speech. Raise it if a station
    naming another station is being logged as two check-ins."""
    min_segment_ms: int = 400
    """Discard a split that would produce a sliver; more likely a mis-timing."""


@dataclass
class WhisperConfig:
    model_size: str = "base"
    condition_audio: bool = True
    """High-pass and normalise each clip before decoding. Sub-millisecond."""
    prompt_token_budget: int = 200
    """Whisper's prompt window is 224 tokens; overflow is silently discarded."""
    device: str = "auto"
    compute_type: str | None = None
    beam_size: int = 5
    language: str | None = "en"
    vocabulary: list[str] = field(
        default_factory=lambda: [
            "net control",
            "QNI",
            "QRZ",
            "QSL",
            "QTH",
            "check in",
            "traffic",
            "over",
            "roger",
        ]
    )


@dataclass
class EscalationConfig:
    """Re-transcribe the hard clips with a bigger model, in the gaps.

    The live line comes from a model chosen for speed. Anything that came back
    unmatched or unsure is queued for a second pass with a larger model, run
    only when nothing live is waiting -- so the dashboard keeps up while the
    accuracy of a bigger model lands on exactly the clips that needed it.
    """

    enabled: bool = False
    """Off by default: it loads a second model, which is real memory."""
    model_size: str = "small"
    device: str = "auto"
    compute_type: str | None = None
    on_unmatched: bool = True
    """Escalate anything the roster could not match -- the clearest failures."""
    min_confidence: float = 0.55
    """Also escalate matched lines below this confidence."""
    max_pending: int = 50
    """Ceiling on the queue; past it the oldest waiting clip is dropped."""


@dataclass
class VoiceConfig:
    """Recognising a station by voice, to help when the callsign is not usable.

    Suggestions only, and only on lines the roster could not match. A voice
    match never overrides a callsign that was actually heard.
    """

    enabled: bool = False
    path: str = "voices.json"
    """Profiles persist here, so next week's net starts knowing these voices."""
    min_similarity: float = 0.82
    """How close a voice must be before it is worth suggesting. Expect to tune
    this against your own net: it is the one number here that no amount of
    synthetic testing can set correctly."""
    margin: float = 0.06
    """The best match must beat the runner-up by this much, or say nothing."""
    min_enrolments: int = 2
    """Clips needed before a profile is trusted at all."""
    enrol_min_score: float = 95.0
    """Only learn a voice from a roster match this clean. A profile built from
    a wrong match poisons every later suggestion."""
    recent_audio: int = 60
    """Clips kept in memory so an operator correction can enrol its audio --
    corrections are the best labels there are."""


@dataclass
class RosterConfig:
    path: str = "roster.csv"
    threshold: float = 78.0
    ambiguity_margin: float = 5.0
    feedback_path: str = "feedback.jsonl"
    """Operator corrections. Replayed at startup so learned aliases persist."""
    learn_aliases: bool = True
    """Set false to log corrections without applying them to future matching."""


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8080


@dataclass
class HealthConfig:
    stall_after_s: float = 5.0
    """No audio frames for this long is an error."""
    silence_after_s: float = 300.0
    """Frames arriving but no signal in them for this long is a warning."""
    silence_rms: float = 15.0
    """Below this level (int16 units) a frame counts as dead air."""
    check_interval_s: float = 1.0
    heartbeat_s: float = 60.0
    """How often to log a stats line, so the log shows the net progressing."""
    restart_capture: bool = True
    """Reopen the audio device automatically when capture dies."""
    restart_delay_s: float = 2.0
    restart_max_delay_s: float = 30.0
    """Backoff ceiling; a device that is gone should not be retried in a spin."""


@dataclass
class BufferingConfig:
    """How much slack the pipeline has when transcription falls behind."""

    ring_seconds: float = 30.0
    """Pre-allocated audio buffer between the device and the VAD."""
    clip_queue_max: int = 32
    """Clips held in memory waiting for Whisper. Past this they go to disk."""
    spill_enabled: bool = True
    """Write the overflow to disk instead of dropping it."""
    spill_dir: str = "spill"
    spill_max_clips: int = 500
    """Ceiling on the disk backlog; past it the oldest spilled clip is dropped."""
    drain_timeout_s: float = 30.0
    """How long to keep transcribing the backlog at shutdown."""


@dataclass
class TranscriptConfig:
    """Writing the session to disk as it happens, not only at the end."""

    live: bool = True
    dir: str = "transcripts"
    fsync: bool = True
    """Force each line to disk. The point is surviving a power cut, and a
    buffered write that never landed would defeat it."""


@dataclass
class LoggingConfig:
    dir: str | None = "logs"
    """Set null to disable file logging and use the console only."""
    level: str = "INFO"
    file_level: str = "DEBUG"
    max_bytes: int = 5_000_000
    backups: int = 5


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    sources: list[SourceConfig] = field(default_factory=list)
    """Multiple receivers. Empty means use the single `audio:` block."""
    vad: VadConfig = field(default_factory=VadConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    escalation: EscalationConfig = field(default_factory=EscalationConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    roster: RosterConfig = field(default_factory=RosterConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    buffering: BufferingConfig = field(default_factory=BufferingConfig)
    transcripts: TranscriptConfig = field(default_factory=TranscriptConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    export_dir: str = "."


def audio_sources(config: Config) -> list[SourceConfig]:
    """The sources to open, however they were configured.

    A single-input `audio:` block stays valid -- it is the common case, and
    breaking existing configs to add a feature most operators will not use
    would be a poor trade.
    """
    if config.sources:
        return [s for s in config.sources if s.enabled]
    return [
        SourceConfig(
            name="Main",
            device=config.audio.device,
            channel=config.audio.channel,
            gain=config.audio.gain,
        )
    ]


def load_config(path: str | Path | None) -> Config:
    data: dict = {}
    if path:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = _build(data)
    _apply_env(config)
    return config


_SECTIONS = {
    "audio": AudioConfig,
    "vad": VadConfig,
    "split": SplitConfig,
    "whisper": WhisperConfig,
    "escalation": EscalationConfig,
    "voice": VoiceConfig,
    "roster": RosterConfig,
    "server": ServerConfig,
    "health": HealthConfig,
    "buffering": BufferingConfig,
    "transcripts": TranscriptConfig,
    "logging": LoggingConfig,
}


def _build(data: dict) -> Config:
    """Build a Config from parsed YAML, ignoring keys the app does not know."""
    kwargs: dict = {}
    for f in fields(Config):
        if f.name not in data:
            continue
        value = data[f.name]
        if f.name == "sources":
            known = {sf.name for sf in fields(SourceConfig)}
            kwargs[f.name] = [
                SourceConfig(**{k: v for k, v in (item or {}).items() if k in known})
                for item in (value or [])
            ]
            continue
        section = _SECTIONS.get(f.name)
        if section is not None:
            known = {sf.name for sf in fields(section)}
            kwargs[f.name] = section(
                **{k: v for k, v in (value or {}).items() if k in known}
            )
        else:
            kwargs[f.name] = value
    return Config(**kwargs)


def _apply_env(config: Config) -> None:
    for section_name, section_cls in _SECTIONS.items():
        section = getattr(config, section_name)
        for f in fields(section_cls):
            env = f"{ENV_PREFIX}_{section_name.upper()}_{f.name.upper()}"
            raw = os.environ.get(env)
            if raw is None:
                continue
            setattr(section, f.name, _coerce(raw, getattr(section, f.name)))
    export = os.environ.get(f"{ENV_PREFIX}_EXPORT_DIR")
    if export:
        config.export_dir = export


def _coerce(raw: str, current):
    """Coerce an env string to the type of the value it replaces."""
    if isinstance(current, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        return [item.strip() for item in raw.split(",") if item.strip()]
    return raw
