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
    """Substring of the input device name; null uses the system default."""
    frame_ms: int = 30


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


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8080


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    roster: RosterConfig = field(default_factory=RosterConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
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
