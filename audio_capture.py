"""Capture mono 16 kHz PCM from a PulseAudio/PipeWire source.

The expected input is a monitor source fed by SDR++/GQRX, e.g.
`alsa_output.pci-0000_00_1f.3.analog-stereo.monitor` or a null sink created for
the purpose. Run `python app.py --list-devices` to see what the host offers.
"""

from __future__ import annotations

import queue
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

TARGET_RATE = 16_000
"""Whisper and webrtcvad both want 16 kHz; everything upstream resamples to it."""


def list_devices() -> str:
    return str(sd.query_devices())


def find_device(name: str | None) -> int | None:
    """Resolve a device by substring match. None means the system default."""
    if not name:
        return None
    devices = sd.query_devices()
    for index, device in enumerate(devices):
        if device["max_input_channels"] > 0 and name.lower() in device["name"].lower():
            return index
    available = "\n  ".join(
        d["name"] for d in devices if d["max_input_channels"] > 0
    )
    raise ValueError(
        f"No input device matching {name!r}. Available input devices:\n  {available}"
    )


@dataclass
class AudioCapture:
    """Yields fixed-size int16 frames of 16 kHz mono audio.

    device: substring of the input device name, or None for the default.
    frame_ms: frame size handed to the VAD; webrtcvad accepts 10, 20, or 30.
    """

    device: str | None = None
    frame_ms: int = 30
    queue_maxsize: int = 200

    def __post_init__(self) -> None:
        if self.frame_ms not in (10, 20, 30):
            raise ValueError("frame_ms must be 10, 20, or 30 (webrtcvad constraint)")
        self._device_index = find_device(self.device)
        self._rate = self._pick_samplerate()
        self._ratio = self._rate // TARGET_RATE
        # Frames are counted at the *source* rate, then downsampled.
        self._source_frame = int(self._rate * self.frame_ms / 1000)
        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=self.queue_maxsize)
        self._stream: sd.InputStream | None = None
        self.overflows = 0

    @property
    def samplerate(self) -> int:
        return self._rate

    def _pick_samplerate(self) -> int:
        """Prefer capturing at 16 kHz; fall back to an integer multiple of it."""
        try:
            sd.check_input_settings(
                device=self._device_index, channels=1, samplerate=TARGET_RATE
            )
            return TARGET_RATE
        except Exception:
            pass
        default = int(
            sd.query_devices(self._device_index, "input")["default_samplerate"]
        )
        if default % TARGET_RATE != 0:
            raise RuntimeError(
                f"Device sample rate {default} is not a multiple of {TARGET_RATE}; "
                "configure the loopback sink for 48000 Hz (or 16000 Hz)."
            )
        return default

    def _downsample(self, block: np.ndarray) -> np.ndarray:
        """Decimate by an integer ratio, averaging to act as a crude anti-alias.

        Net audio is narrowband voice off an FM/SSB receiver, so the content
        above 8 kHz is noise we are happy to lose.
        """
        if self._ratio == 1:
            return block
        usable = len(block) - (len(block) % self._ratio)
        return (
            block[:usable]
            .reshape(-1, self._ratio)
            .mean(axis=1)
            .astype(np.int16)
        )

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            self.overflows += 1
        mono = indata[:, 0] if indata.ndim > 1 else indata
        frame = self._downsample(np.asarray(mono, dtype=np.int16))
        try:
            self._queue.put_nowait(frame.tobytes())
        except queue.Full:
            # Dropping is the right failure mode: the STT worker fell behind and
            # we would rather lose a frame than grow an unbounded backlog.
            self.overflows += 1

    def start(self) -> None:
        self._stream = sd.InputStream(
            device=self._device_index,
            channels=1,
            samplerate=self._rate,
            dtype="int16",
            blocksize=self._source_frame,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._queue.put(None)

    def frames(self):
        """Blocking iterator of 16 kHz int16 frames, ending when stop() is called."""
        while True:
            frame = self._queue.get()
            if frame is None:
                return
            yield frame
