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
class VadConfig:
    aggressiveness: int = 3
    silence_ms: int = 800
    min_clip_ms: int = 400
    max_clip_ms: int = 120_000
    preroll_ms: int = 300
    trigger_ratio: float = 0.7


@dataclass
class WhisperConfig:
    model_size: str = "base"
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
    vad: VadConfig = field(default_factory=VadConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    roster: RosterConfig = field(default_factory=RosterConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    buffering: BufferingConfig = field(default_factory=BufferingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    export_dir: str = "."


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
    "whisper": WhisperConfig,
    "roster": RosterConfig,
    "server": ServerConfig,
    "health": HealthConfig,
    "buffering": BufferingConfig,
    "logging": LoggingConfig,
}


def _build(data: dict) -> Config:
    """Build a Config from parsed YAML, ignoring keys the app does not know."""
    kwargs: dict = {}
    for f in fields(Config):
        if f.name not in data:
            continue
        value = data[f.name]
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
