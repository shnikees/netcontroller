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

"""Tests for running a trained speaker model through ONNX Runtime.

There is no real speaker model in the repository -- it would be tens of
megabytes of downloaded weights. What is tested instead is the *adapter*, which
is where the risk actually lives: exported speaker models disagree about
whether they want a waveform or features, which way round the feature axes go,
and whether a length has to be passed alongside. Getting that wrong on a model
somebody downloads later is the failure this guards against.

So the models here are stand-ins built on the spot, one per input convention.
"""

from __future__ import annotations

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import voice_id  # noqa: E402
import voice_onnx  # noqa: E402
from onnx import TensorProto, helper  # noqa: E402

DIMENSIONS = 32


def clip(seconds: float = 2.0, freq: float = 200.0) -> np.ndarray:
    t = np.arange(int(16_000 * seconds)) / 16_000
    return (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)


def _save(graph, path) -> str:
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, str(path))
    return str(path)


def waveform_model(path) -> str:
    """[1, samples] in, [1, D] out -- the ECAPA-style convention."""
    audio = helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, "n"])
    out = helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, DIMENSIONS])
    # Mean *absolute* amplitude, then project. The absolute value matters:
    # averaging a waveform directly gives zero for anything symmetric, which
    # would make this stand-in produce a null vector rather than exercise
    # anything.
    magnitude = helper.make_node("Abs", ["audio"], ["magnitude"])
    reduced = helper.make_node(
        "ReduceMean", ["magnitude"], ["pooled"], axes=[1], keepdims=1
    )
    weights = helper.make_tensor(
        "w", TensorProto.FLOAT, [1, DIMENSIONS],
        np.linspace(0.5, 2.0, DIMENSIONS).astype(np.float32).tobytes(), raw=True,
    )
    project = helper.make_node("MatMul", ["pooled", "w"], ["embedding"])
    graph = helper.make_graph(
        [magnitude, reduced, project], "waveform", [audio], [out],
        initializer=[weights],
    )
    return _save(graph, path)


def feature_model(path, transposed: bool, with_length: bool) -> str:
    """Features in, [1, D] out. `transposed` is the NeMo [batch, mels, frames]."""
    shape = [1, 80, "t"] if transposed else [1, "t", 80]
    features = helper.make_tensor_value_info("features", TensorProto.FLOAT, shape)
    inputs = [features]
    if with_length:
        inputs.append(
            helper.make_tensor_value_info("length", TensorProto.INT64, [1])
        )
    out = helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, DIMENSIONS])

    axis = 2 if transposed else 1
    reduced = helper.make_node(
        "ReduceMean", ["features"], ["pooled"], axes=[axis], keepdims=0
    )
    # pooled is [1, 80] either way.
    weights = helper.make_tensor(
        "w", TensorProto.FLOAT, [80, DIMENSIONS],
        np.linspace(0.1, 1.0, 80 * DIMENSIONS).astype(np.float32).tobytes(), raw=True,
    )
    project = helper.make_node("MatMul", ["pooled", "w"], ["embedding"])
    graph = helper.make_graph(
        [reduced, project], "features", inputs, [out], initializer=[weights]
    )
    return _save(graph, path)


# --------------------------------------------------------------------------
# Each input convention the wild actually uses
# --------------------------------------------------------------------------


def test_a_waveform_model_is_fed_a_waveform(tmp_path) -> None:
    embedder = voice_onnx.OnnxEmbedder(waveform_model(tmp_path / "wave.onnx"))
    assert embedder.kind == "waveform"

    vector = embedder(clip())
    assert vector is not None
    assert vector.shape == (DIMENSIONS,)
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("transposed", [False, True])
def test_a_feature_model_is_fed_features_the_right_way_round(
    tmp_path, transposed: bool
) -> None:
    # The axis order differs between families; reading it off the declared
    # shape is what stops a downloaded model needing a code change.
    path = feature_model(tmp_path / f"feat-{transposed}.onnx", transposed, False)
    embedder = voice_onnx.OnnxEmbedder(path)
    assert embedder.kind == "features"

    vector = embedder(clip())
    assert vector is not None and vector.shape == (DIMENSIONS,)


def test_a_model_that_wants_a_length_gets_one(tmp_path) -> None:
    # TitaNet and friends take (features, length) so padding can be masked.
    path = feature_model(tmp_path / "len.onnx", True, with_length=True)
    embedder = voice_onnx.OnnxEmbedder(path)
    assert embedder(clip()) is not None


def test_the_output_dimension_is_reported(tmp_path) -> None:
    embedder = voice_onnx.OnnxEmbedder(waveform_model(tmp_path / "wave.onnx"))
    assert embedder.dimensions == DIMENSIONS


# --------------------------------------------------------------------------
# Behaving like the built-in embedder
# --------------------------------------------------------------------------


def test_different_audio_gives_different_vectors(tmp_path) -> None:
    embedder = voice_onnx.OnnxEmbedder(
        feature_model(tmp_path / "feat.onnx", True, False)
    )
    one = embedder(clip(freq=150))
    two = embedder(clip(freq=900))
    assert not np.allclose(one, two)


def test_a_clip_too_short_returns_nothing(tmp_path) -> None:
    embedder = voice_onnx.OnnxEmbedder(waveform_model(tmp_path / "wave.onnx"))
    assert embedder(clip(seconds=0.2)) is None


# --------------------------------------------------------------------------
# Failing safely -- a missing model must never stop a net
# --------------------------------------------------------------------------


def test_a_missing_model_falls_back_rather_than_raising(tmp_path) -> None:
    assert voice_onnx.load(tmp_path / "not-here.onnx") is None


def test_an_unset_path_falls_back(tmp_path) -> None:
    assert voice_onnx.load(None) is None


def test_a_corrupt_model_falls_back(tmp_path) -> None:
    broken = tmp_path / "broken.onnx"
    broken.write_bytes(b"this is not a model")
    assert voice_onnx.load(broken) is None


# --------------------------------------------------------------------------
# Swapping the backend under the profile store
# --------------------------------------------------------------------------


def test_the_backend_replaces_the_built_in_embedder(tmp_path) -> None:
    try:
        voice_id.set_backend(
            voice_onnx.OnnxEmbedder(waveform_model(tmp_path / "wave.onnx"))
        )
        assert voice_id.embed(clip()).shape == (DIMENSIONS,)
    finally:
        voice_id.set_backend(None)
    assert voice_id.embed(clip()).shape == (24,)


def test_profiles_rebuild_onto_a_new_backend(tmp_path) -> None:
    """The whole point of keeping the enrolment audio: swapping the embedder
    is a re-embed pass, not weeks of re-enrolment."""
    from voice_id import EnrolmentAudio, VoiceProfiles

    store = EnrolmentAudio(tmp_path / "audio", per_station=3)
    profiles = VoiceProfiles(audio=store, min_enrolments=1)
    for freq in (150, 160, 170):
        profiles.enrol("W6ABC", clip(freq=freq))
    assert profiles.profiles["W6ABC"].centroid.shape == (24,)

    try:
        voice_id.set_backend(
            voice_onnx.OnnxEmbedder(feature_model(tmp_path / "f.onnx", True, False))
        )
        stations, clips = profiles.rebuild()
    finally:
        voice_id.set_backend(None)

    assert stations == 1 and clips == 3
    assert profiles.profiles["W6ABC"].centroid.shape == (DIMENSIONS,)


# --------------------------------------------------------------------------
# Mean normalisation
#
# Found by finally running a real model. A wespeaker ECAPA export loaded
# cleanly, reported the right dimensions and returned unit vectors -- and
# embedded five different clips of net audio to within 0.013 of each other.
# Every voice looked like every other voice, and nothing errored.
# --------------------------------------------------------------------------


def test_features_are_mean_normalised_before_inference(tmp_path) -> None:
    """The model must see features with the time-mean removed."""
    model = feature_model(tmp_path / "m.onnx", transposed=False, with_length=False)
    embedder = voice_onnx.OnnxEmbedder(model)
    embedder.session = _Recorder(embedder.session)

    embedder(np.random.default_rng(0).normal(0, 0.1, 16_000).astype(np.float32))

    fed = embedder.session.last_input
    assert fed is not None
    # Mean over the time axis should be ~0 for every feature.
    means = fed[0].mean(axis=0)
    assert np.allclose(means, 0.0, atol=1e-4), means[:4]


def test_normalisation_can_be_switched_off(tmp_path) -> None:
    """A model normalising internally would be normalised twice otherwise."""
    model = feature_model(tmp_path / "m.onnx", transposed=False, with_length=False)
    embedder = voice_onnx.OnnxEmbedder(model, mean_normalise=False)
    embedder.session = _Recorder(embedder.session)

    embedder(np.random.default_rng(1).normal(0, 0.1, 16_000).astype(np.float32))

    means = embedder.session.last_input[0].mean(axis=0)
    assert not np.allclose(means, 0.0, atol=1e-4), "should not have been centred"


class _Recorder:
    """Passes calls through, keeping the array the model was given."""

    def __init__(self, wrapped) -> None:
        self._wrapped = wrapped
        self.last_input = None

    def run(self, outputs, feed):
        self.last_input = next(iter(feed.values()))
        return self._wrapped.run(outputs, feed)

    def get_inputs(self):
        return self._wrapped.get_inputs()

    def get_outputs(self):
        return self._wrapped.get_outputs()
