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

"""The settings worth changing while a net is running.

Not everything in `config.yaml` belongs in a dashboard. Device names, ports and
buffer depths are set once at install, and a UI for them is a text editor with
extra steps. What belongs here is the handful of things somebody reaches for
*mid-event*, when walking to a terminal costs transmissions:

- a level that turns out to be wrong once traffic starts
- a model that cannot keep up on the night
- a threshold that is splitting or merging the wrong things
- a feature that is more noise than help on this particular net

Each setting names what it does in the language of the problem rather than the
implementation, carries its own bounds so a slip cannot wedge the pipeline, and
says when it costs something to change. Changes apply immediately and in
memory; writing them back to `config.yaml` is a separate, deliberate act.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Setting:
    """One knob, described well enough for a UI to render it unaided."""

    path: str
    """Dotted path into the config: "vad.silence_ms"."""
    label: str
    group: str
    help: str
    kind: str = "float"
    """bool, int, float, or choice."""
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[str, ...] = ()
    cost: str = ""
    """What changing it costs, if anything -- shown as a warning in the UI."""


SETTINGS: tuple[Setting, ...] = (
    # -- Transcription ----------------------------------------------------
    Setting(
        path="whisper.model_size",
        label="Model",
        group="Transcription",
        help="Bigger is more accurate on fast, run-together speech and slower "
        "per transmission. Drop a size if lines are arriving late.",
        kind="choice",
        choices=("tiny", "base", "small", "medium", "large-v3"),
        cost="Reloads the model. Audio keeps buffering meanwhile, so nothing "
        "is lost, but transcripts pause for a few seconds.",
    ),
    Setting(
        path="whisper.beam_size",
        label="Beam size",
        group="Transcription",
        help="How many decodings Whisper weighs before choosing. Higher is a "
        "little more accurate and proportionally slower.",
        kind="int",
        minimum=1,
        maximum=10,
        step=1,
    ),
    Setting(
        path="escalation.enabled",
        label="Second pass on unsure lines",
        group="Transcription",
        help="Re-transcribe lines that came back unmatched or low-confidence "
        "with a bigger model, in the gaps between transmissions.",
        kind="bool",
        cost="Loads a second model the first time it runs.",
    ),
    Setting(
        path="escalation.model_size",
        label="Second-pass model",
        group="Transcription",
        kind="choice",
        choices=("base", "small", "medium", "large-v3"),
        help="The model used for the second pass. It only runs when nothing "
        "live is waiting, so it can afford to be slow.",
        cost="Reloads the second-pass model.",
    ),
    # -- Segmentation -----------------------------------------------------
    Setting(
        path="vad.silence_ms",
        label="Pause that ends a transmission",
        group="Segmentation",
        help="Raise it if one station's check-in is being split across several "
        "lines. Lower it if two stations end up on one line.",
        kind="int",
        minimum=200,
        maximum=3000,
        step=50,
    ),
    Setting(
        path="vad.aggressiveness",
        label="Squelch rejection",
        group="Segmentation",
        help="How hard the detector works to reject non-speech. Lower it if "
        "quiet stations are being missed entirely.",
        kind="int",
        minimum=0,
        maximum=3,
        step=1,
    ),
    Setting(
        path="vad.min_clip_ms",
        label="Shortest transmission",
        group="Segmentation",
        help="Anything briefer is dropped. Raise it if squelch tails and "
        "kerchunks are producing junk lines.",
        kind="int",
        minimum=100,
        maximum=2000,
        step=50,
    ),
    Setting(
        path="split.min_gap_ms",
        label="Gap between two stations",
        group="Segmentation",
        help="Dead air before two callsigns in one clip count as two separate "
        "transmissions. Raise it if a station naming another is being logged "
        "twice.",
        kind="int",
        minimum=200,
        maximum=2000,
        step=50,
    ),
    # -- Matching ---------------------------------------------------------
    Setting(
        path="roster.threshold",
        label="Match confidence",
        group="Matching",
        help="How close a heard callsign must be to a roster entry. Higher "
        "means fewer wrong matches and more unmatched lines.",
        kind="int",
        minimum=50,
        maximum=95,
        step=1,
    ),
    Setting(
        path="roster.ambiguity_margin",
        label="Ambiguity margin",
        group="Matching",
        help="If two roster entries are this close, the line is left unmatched "
        "rather than guessed at.",
        kind="float",
        minimum=0,
        maximum=20,
        step=0.5,
    ),
    # -- On the dashboard -------------------------------------------------
    Setting(
        path="traffic.detect",
        label="Mark traffic",
        group="Dashboard",
        help="Read traffic declarations off each transmission and badge them.",
        kind="bool",
    ),
    Setting(
        path="traffic.acknowledge",
        label="Allow clearing traffic",
        group="Dashboard",
        help="Let the badge be clicked when traffic has been passed, so the "
        "outstanding list empties.",
        kind="bool",
    ),
    Setting(
        path="voice.enabled",
        label="Suggest stations by voice",
        group="Dashboard",
        help="On unmatched lines, offer the station whose voice it sounds "
        "like. Suggestions only -- never applied on its own.",
        kind="bool",
    ),
)

GAIN = Setting(
    path="",
    label="",
    group="Audio",
    help="Input level for this receiver. Raise it if the level reads low; "
    "lower it if the audio is clipping.",
    kind="float",
    minimum=0.1,
    maximum=20.0,
    step=0.1,
)


def for_config(config) -> list[Setting]:
    """Every setting, including one gain control per configured source."""
    from config import audio_sources

    out = list(SETTINGS)
    sources = audio_sources(config)
    for source in sources:
        label = "Input level" if len(sources) == 1 else f"{source.name} level"
        out.append(
            Setting(
                path=f"sources.{source.name}.gain",
                label=label,
                group=GAIN.group,
                help=GAIN.help,
                kind=GAIN.kind,
                minimum=GAIN.minimum,
                maximum=GAIN.maximum,
                step=GAIN.step,
            )
        )
    return out


# --------------------------------------------------------------------------
# Reading and writing values
# --------------------------------------------------------------------------


def get_value(config, path: str):
    """Current value at a dotted path, including the per-source gains."""
    if path.startswith("sources."):
        _, name, field_name = path.split(".", 2)
        for source in _configured_sources(config):
            if source.name == name:
                return getattr(source, field_name)
        return None
    section, key = path.split(".", 1)
    return getattr(getattr(config, section), key)


def set_value(config, path: str, value) -> None:
    """Write a validated value back into the config objects."""
    if path.startswith("sources."):
        _, name, field_name = path.split(".", 2)
        for source in _configured_sources(config):
            if source.name == name:
                setattr(source, field_name, value)
        return
    section, key = path.split(".", 1)
    setattr(getattr(config, section), key, value)


def _configured_sources(config):
    from config import audio_sources

    # The single-input case synthesises a SourceConfig, so mutating it would be
    # writing to a temporary. Fall back to the audio block it came from.
    return config.sources or [_SingleSourceProxy(config)]


class _SingleSourceProxy:
    """Makes the single `audio:` block look like a named source."""

    def __init__(self, config) -> None:
        self._audio = config.audio
        self.name = "Main"

    def __getattr__(self, item):
        return getattr(self._audio, item)

    def __setattr__(self, item, value):
        if item in ("_audio", "name"):
            super().__setattr__(item, value)
        else:
            setattr(self._audio, item, value)


def coerce(setting: Setting, value):
    """Validate and convert a value from the wire, or raise ValueError.

    Bounds live with the setting rather than in the UI, so a hand-made request
    cannot put the pipeline somewhere it cannot come back from.
    """
    if setting.kind == "bool":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    if setting.kind == "choice":
        text = str(value)
        if text not in setting.choices:
            raise ValueError(f"{text!r} is not one of {', '.join(setting.choices)}")
        return text

    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{value!r} is not a number") from None
    if setting.minimum is not None and number < setting.minimum:
        raise ValueError(f"must be at least {setting.minimum}")
    if setting.maximum is not None and number > setting.maximum:
        raise ValueError(f"must be at most {setting.maximum}")
    return int(round(number)) if setting.kind == "int" else number


def describe(config) -> list[dict]:
    """Everything a dashboard needs to render the panel, values included."""
    described = []
    for setting in for_config(config):
        described.append(
            {
                "path": setting.path,
                "label": setting.label,
                "group": setting.group,
                "help": setting.help,
                "kind": setting.kind,
                "min": setting.minimum,
                "max": setting.maximum,
                "step": setting.step,
                "choices": list(setting.choices),
                "cost": setting.cost,
                "value": get_value(config, setting.path),
            }
        )
    return described


def file_path(config, path: str) -> str:
    """Where a setting lives in config.yaml, which is not always where it lives
    in memory.

    A single-input setup has no `sources:` block: the synthesised "Main" source
    is really the `audio:` block, so its level has to be written back there or
    the save silently does nothing.
    """
    if path.startswith("sources.") and not config.sources:
        return "audio." + path.split(".", 2)[2]
    return path


def find(config, path: str) -> Setting | None:
    return next((s for s in for_config(config) if s.path == path), None)
