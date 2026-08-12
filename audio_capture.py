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

"""Capture mono 16 kHz PCM from any audio input.

Three sources are supported, and the app does not care which you use:

- **SDR loopback** -- a monitor source fed by SDR++/GQRX. Best fidelity, since
  the audio never leaves the machine.
- **Line in** -- a USB sound card or built-in line input taking the speaker or
  headphone output of a physical radio. The practical choice when the radio is
  a handheld or a mobile rig rather than an SDR.
- **Microphone** -- pointed at a speaker. Works, and is the fastest thing to
  set up, but the room is in the recording too.

The differences that matter are handled here: line inputs and microphones
usually run at 44.1 kHz (resampled, see resample.py), are often stereo with the
radio on one channel only (`channel`), and arrive at wildly different levels
depending on whether you tapped a speaker output or a line output (`gain`).

Run `python app.py --list-devices` to see what the host offers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import sounddevice as sd

from resample import Resampler, describe
from ring_buffer import RingBuffer

log = logging.getLogger(__name__)

TARGET_RATE = 16_000
"""Whisper and webrtcvad both want 16 kHz; everything here resamples to it."""

PREFERRED_RATES = (16_000, 48_000, 44_100, 32_000, 22_050, 8_000)
"""Tried in order. The first two need no real resampling."""


def list_devices() -> str:
    """Human-readable device list, annotated with what each one looks like.

    The point is to make the choice obvious to somebody who does not know
    whether their radio shows up as "USB Audio CODEC" or "monitor of Null
    Output" -- which is most people, the first time.
    """
    lines = [str(sd.query_devices()), "", "Likely candidates for this app:"]
    try:
        devices = sd.query_devices()
    except Exception:  # pragma: no cover - no audio subsystem at all
        return lines[0]

    for index, device in enumerate(devices):
        if device["max_input_channels"] < 1:
            continue
        name = device["name"]
        lowered = name.lower()
        if "monitor" in lowered:
            hint = "SDR loopback (an app's output)"
        elif any(word in lowered for word in ("usb", "codec", "line")):
            hint = "line in / USB sound card -- a radio's speaker output"
        elif any(word in lowered for word in ("mic", "built-in", "internal")):
            hint = "microphone -- picks up the room as well"
        else:
            continue
        lines.append(
            f"  [{index}] {name}  ({device['max_input_channels']} ch, "
            f"{int(device['default_samplerate'])} Hz) -- {hint}"
        )
    if len(lines) == 3:
        lines.append("  (none recognised; any input device above will work)")
    return "\n".join(lines)


def find_device(name: str | None) -> int | None:
    """Resolve a device by index or name substring. None means system default."""
    if name is None or name == "":
        return None
    text = str(name).strip()
    if text.lstrip("-").isdigit():
        return int(text)

    devices = sd.query_devices()
    for index, device in enumerate(devices):
        if device["max_input_channels"] > 0 and text.lower() in device["name"].lower():
            return index
    available = "\n  ".join(
        f"[{i}] {d['name']}"
        for i, d in enumerate(devices)
        if d["max_input_channels"] > 0
    )
    raise ValueError(
        f"No input device matching {text!r}. Available input devices:\n  {available}"
    )


@dataclass
class AudioCapture:
    """Yields fixed-size int16 frames of 16 kHz mono audio.

    device: name substring or index of the input, or None for the default.
    frame_ms: frame size handed to the VAD; webrtcvad accepts 10, 20, or 30.
    channel: "mix" averages all channels; "left"/"right" or an integer index
        takes one. Use a single channel when a stereo line-in has the radio on
        one side only -- mixing in a dead channel halves the level.
    gain: linear multiplier applied before anything else. A speaker output into
        a line input is usually hot; a line output into a mic input is usually
        far too quiet. Clipping is counted, not silently ignored.
    """

    device: str | int | None = None
    frame_ms: int = 30
    channel: str | int = "mix"
    gain: float = 1.0
    buffer_seconds: float = 30.0
    """Depth of the pre-allocated ring buffer. This is the slack the pipeline
    has when transcription falls behind; 30 s covers a long over on a slow box."""
    _resampler: Resampler = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.frame_ms not in (10, 20, 30):
            raise ValueError("frame_ms must be 10, 20, or 30 (webrtcvad constraint)")
        if self.gain <= 0:
            raise ValueError("gain must be positive")

        self._device_index = find_device(self.device)
        self._info = sd.query_devices(self._device_index, "input")
        self._channels = int(self._info["max_input_channels"]) or 1
        self._channel_index = self._resolve_channel()
        self._rate = self._pick_samplerate()
        self._resampler = Resampler(self._rate, TARGET_RATE)

        self._frame_samples = int(TARGET_RATE * self.frame_ms / 1000)
        self._block = int(self._rate * self.frame_ms / 1000)
        # Allocated once, up front: the audio callback never grows the heap.
        self._ring = RingBuffer(int(TARGET_RATE * self.buffer_seconds))
        self._stream: sd.InputStream | None = None
        self.overflows = 0
        self.clipped = 0

    # -- properties --------------------------------------------------------

    @property
    def samplerate(self) -> int:
        return self._rate

    @property
    def device_name(self) -> str:
        return str(self._info["name"])

    def describe(self) -> str:
        """One line for the startup log: what we opened and what we do to it."""
        channel = (
            "mixed" if self._channel_index is None else f"channel {self._channel_index}"
        )
        gain = "" if self.gain == 1.0 else f", gain x{self.gain:g}"
        return (
            f"{self.device_name} [{self._channels} ch, {channel}{gain}] "
            f"{describe(self._rate, TARGET_RATE)}"
        )

    # -- setup -------------------------------------------------------------

    def _resolve_channel(self) -> int | None:
        """None means mix everything; otherwise a zero-based channel index."""
        value = self.channel
        if isinstance(value, str):
            key = value.strip().lower()
            if key in ("mix", "both", "mono", ""):
                return None
            if key in ("left", "l", "0"):
                return 0
            if key in ("right", "r", "1"):
                return 1
            if key.isdigit():
                value = int(key)
            else:
                raise ValueError(
                    f"Unknown channel {self.channel!r}; use mix, left, right, "
                    "or a 0-based index"
                )
        index = int(value)
        if index >= self._channels:
            raise ValueError(
                f"Device {self.device_name!r} has {self._channels} channel(s); "
                f"channel {index} does not exist"
            )
        return index

    def _pick_samplerate(self) -> int:
        """First rate the device accepts, preferring ones needing no resampling.

        Microphones and USB sound cards frequently refuse 16 kHz outright and
        only offer 44.1 kHz -- which is why resampling exists rather than the
        old "must be a multiple of 16 kHz" rule.
        """
        candidates = list(PREFERRED_RATES)
        default = int(self._info["default_samplerate"])
        if default not in candidates:
            candidates.append(default)

        for rate in candidates:
            try:
                sd.check_input_settings(
                    device=self._device_index,
                    channels=self._channels,
                    samplerate=rate,
                )
                return rate
            except Exception:
                continue
        # Nothing was accepted; let the device's own default surface the real
        # PortAudio error when the stream opens.
        return default

    # -- streaming ---------------------------------------------------------

    def _to_mono(self, indata: np.ndarray) -> np.ndarray:
        if indata.ndim == 1:
            return indata
        if self._channel_index is None:
            return indata.mean(axis=1)
        return indata[:, self._channel_index]

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            self.overflows += 1

        mono = self._to_mono(np.asarray(indata))
        if self.gain != 1.0:
            boosted = mono.astype(np.float32) * self.gain
            clipped = int(np.count_nonzero(np.abs(boosted) >= 32767))
            if clipped:
                self.clipped += clipped
            mono = np.clip(boosted, -32768, 32767)

        resampled = self._resampler.process(np.asarray(mono, dtype=np.int16))
        if len(resampled) == 0:
            return

        # Straight into pre-allocated storage. Re-chunking into exact VAD frames
        # happens on the reader side, where an allocation costs nothing.
        dropped = self._ring.write(resampled)
        if dropped:
            self.overflows += 1

    def start(self) -> None:
        self._stream = sd.InputStream(
            device=self._device_index,
            channels=self._channels,
            samplerate=self._rate,
            dtype="int16",
            blocksize=self._block,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._ring.close()

    @property
    def backlog(self) -> float:
        """How full the ring buffer is, 0.0-1.0."""
        return self._ring.fill

    @property
    def dropped_samples(self) -> int:
        return self._ring.dropped

    def frames(self):
        """Blocking iterator of 16 kHz int16 frames, ending when stop() is called."""
        while True:
            samples = self._ring.read(self._frame_samples)
            if samples is None:
                return
            yield samples.tobytes()
