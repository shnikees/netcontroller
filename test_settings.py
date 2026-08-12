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

"""Tests for the settings a dashboard can change mid-net.

Bounds live with the setting rather than in the browser, because the browser
is not the only thing that can send a request. A value that would wedge the
pipeline has to be refused here.
"""

from __future__ import annotations

import pytest

import config_writer
import settings
from config import Config, SourceConfig


@pytest.fixture
def config() -> Config:
    return Config()


# --------------------------------------------------------------------------
# Reading and validating
# --------------------------------------------------------------------------


def test_current_values_come_from_the_config(config: Config) -> None:
    described = {s["path"]: s["value"] for s in settings.describe(config)}
    assert described["vad.silence_ms"] == config.vad.silence_ms
    assert described["whisper.model_size"] == config.whisper.model_size


def test_every_setting_is_described_well_enough_to_render(config: Config) -> None:
    for described in settings.describe(config):
        assert described["label"] and described["help"] and described["group"]
        if described["kind"] in ("int", "float"):
            assert described["min"] is not None and described["max"] is not None
        if described["kind"] == "choice":
            assert described["choices"]


def test_out_of_range_values_are_refused(config: Config) -> None:
    setting = settings.find(config, "vad.aggressiveness")
    with pytest.raises(ValueError):
        settings.coerce(setting, 9)
    with pytest.raises(ValueError):
        settings.coerce(setting, -1)


def test_an_unknown_model_is_refused(config: Config) -> None:
    setting = settings.find(config, "whisper.model_size")
    with pytest.raises(ValueError):
        settings.coerce(setting, "enormous")


def test_nonsense_is_refused(config: Config) -> None:
    with pytest.raises(ValueError):
        settings.coerce(settings.find(config, "vad.silence_ms"), "soon")


def test_integers_stay_integers(config: Config) -> None:
    assert settings.coerce(settings.find(config, "vad.silence_ms"), "850.4") == 850


def test_booleans_accept_what_a_browser_sends(config: Config) -> None:
    setting = settings.find(config, "traffic.detect")
    assert settings.coerce(setting, True) is True
    assert settings.coerce(setting, "false") is False


# --------------------------------------------------------------------------
# Writing back into the config objects
# --------------------------------------------------------------------------


def test_setting_a_value_updates_the_config(config: Config) -> None:
    settings.set_value(config, "vad.silence_ms", 950)
    assert config.vad.silence_ms == 950


def test_each_source_gets_its_own_level_control(config: Config) -> None:
    config.sources = [SourceConfig(name="Repeater"), SourceConfig(name="Simplex")]
    paths = {s["path"] for s in settings.describe(config)}
    assert "sources.Repeater.gain" in paths
    assert "sources.Simplex.gain" in paths

    settings.set_value(config, "sources.Simplex.gain", 4.0)
    assert config.sources[1].gain == 4.0
    assert config.sources[0].gain == 1.0  # the other receiver is untouched


def test_a_single_input_still_gets_a_level_control(config: Config) -> None:
    # No `sources:` block: the control has to reach the plain audio settings.
    assert any(s["path"].endswith(".gain") for s in settings.describe(config))
    settings.set_value(config, "sources.Main.gain", 3.5)
    assert config.audio.gain == 3.5


# --------------------------------------------------------------------------
# Saving, without destroying the file
# --------------------------------------------------------------------------


EXAMPLE = """# The audio input.
audio:
  device: null
  # VAD frame size in ms.
  frame_ms: 30
  gain: 1.0

vad:
  # Raise this if a check-in splits across lines.
  silence_ms: 800
  aggressiveness: 3

traffic:
  detect: true
  acknowledge: true
"""


def test_saving_keeps_the_comments(tmp_path) -> None:
    # The comments are most of what makes the config readable; a YAML
    # round-trip would throw all of them away.
    path = tmp_path / "config.yaml"
    path.write_text(EXAMPLE, encoding="utf-8")

    config_writer.patch(path, {"vad.silence_ms": 1200})
    text = path.read_text(encoding="utf-8")
    assert "silence_ms: 1200" in text
    assert "# Raise this if a check-in splits across lines." in text
    assert "# The audio input." in text


def test_saving_writes_only_the_named_key(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(EXAMPLE, encoding="utf-8")

    config_writer.patch(path, {"traffic.detect": False})
    text = path.read_text(encoding="utf-8")
    assert "detect: false" in text
    assert "acknowledge: true" in text  # its neighbour is untouched


def test_a_key_in_two_sections_is_not_confused(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "escalation:\n  enabled: false\n\nvoice:\n  enabled: false\n", encoding="utf-8"
    )
    config_writer.patch(path, {"voice.enabled": True})
    text = path.read_text(encoding="utf-8")
    assert "escalation:\n  enabled: false" in text
    assert "voice:\n  enabled: true" in text


def test_a_missing_key_is_reported_not_appended(tmp_path) -> None:
    # Appending it would put the key somewhere arbitrary with no explanation
    # beside it, which is worse than saying so.
    path = tmp_path / "config.yaml"
    path.write_text("vad:\n  silence_ms: 800\n", encoding="utf-8")

    result = config_writer.patch(path, {"whisper.beam_size": 8})
    assert result["written"] == []
    assert result["missing"] == ["whisper.beam_size"]


def test_a_backup_is_left_behind(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(EXAMPLE, encoding="utf-8")
    config_writer.patch(path, {"vad.silence_ms": 1200})
    assert (tmp_path / "config.yaml.bak").read_text(encoding="utf-8") == EXAMPLE


def test_a_named_source_is_patched_in_place(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "sources:\n"
        "  - name: Repeater\n    device: a\n    gain: 1.0\n"
        "  - name: Simplex\n    device: b\n    gain: 1.0\n",
        encoding="utf-8",
    )
    config_writer.patch(path, {"sources.Simplex.gain": 4.0})
    text = path.read_text(encoding="utf-8")
    # The right receiver, and only that one.
    assert "name: Simplex\n    device: b\n    gain: 4.0" in text
    assert "name: Repeater\n    device: a\n    gain: 1.0" in text


def test_the_patched_file_still_loads(tmp_path) -> None:
    from config import load_config

    path = tmp_path / "config.yaml"
    path.write_text(EXAMPLE, encoding="utf-8")
    config_writer.patch(path, {"vad.silence_ms": 1150, "traffic.detect": False})

    loaded = load_config(path)
    assert loaded.vad.silence_ms == 1150
    assert loaded.traffic.detect is False


def test_a_single_inputs_level_is_saved_to_the_audio_block(config: Config, tmp_path) -> None:
    """Where a setting lives in the file is not always where it lives in memory.

    With no `sources:` block the "Main" source is really the `audio:` block,
    and writing to sources.Main.gain would silently do nothing.
    """
    assert settings.file_path(config, "sources.Main.gain") == "audio.gain"

    path = tmp_path / "config.yaml"
    path.write_text(EXAMPLE, encoding="utf-8")
    result = config_writer.patch(path, {"audio.gain": 3.5})
    assert result["written"] == ["audio.gain"]
    assert "gain: 3.5" in path.read_text(encoding="utf-8")


def test_a_named_source_keeps_its_own_path(config: Config) -> None:
    config.sources = [SourceConfig(name="Repeater")]
    assert settings.file_path(config, "sources.Repeater.gain") == "sources.Repeater.gain"


def test_a_missing_file_is_reported(tmp_path) -> None:
    result = config_writer.patch(tmp_path / "nope.yaml", {"vad.silence_ms": 900})
    assert result["error"]
