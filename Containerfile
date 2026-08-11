# Container image for the STT/dashboard app only.
#
# SDR++/GQRX stays on the host: passing a USB SDR into a container is the most
# fragile part of this setup. Point the SDR app at a loopback sink on the host
# and let this container read that sink's monitor source.
#
#   podman build -t net-stt -f Containerfile .
#   podman run --rm -it \
#     -v $XDG_RUNTIME_DIR/pulse:/run/user/1000/pulse \
#     -e PULSE_SERVER=unix:/run/user/1000/pulse/native \
#     -v ./roster.csv:/app/roster.csv:ro \
#     -p 8080:8080 net-stt
#
# Works unchanged on a dev laptop or a Raspberry Pi (build with --platform
# linux/arm64 if cross-building).

FROM python:3.12-slim

# libportaudio2 backs sounddevice; libpulse0 lets it talk to the host's
# PulseAudio/PipeWire socket. build-essential is only needed to compile
# webrtcvad, so it is dropped again in the same layer.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libportaudio2 \
        libpulse0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY *.py ./
COPY static/ ./static/
COPY config.yaml.example roster.example.csv ./

# Bake the model into the image so a field deployment never needs the network.
# Override at build time: --build-arg MODEL_SIZE=tiny for a Pi 4.
ARG MODEL_SIZE=base
ENV NETSTT_WHISPER_MODEL_SIZE=${MODEL_SIZE}
RUN python -c "from faster_whisper import WhisperModel; \
    WhisperModel('${MODEL_SIZE}', device='cpu', compute_type='int8')"

# Run as a non-root user whose UID matches the typical desktop user, so the
# bind-mounted Pulse socket is readable. Override with --user if yours differs.
ARG UID=1000
RUN useradd -m -u ${UID} netstt && chown -R netstt /app
USER netstt

ENV NETSTT_SERVER_HOST=0.0.0.0 \
    NETSTT_SERVER_PORT=8080 \
    NETSTT_ROSTER_PATH=/app/roster.csv \
    NETSTT_EXPORT_DIR=/app/logs

EXPOSE 8080
VOLUME ["/app/logs"]

# No --config: the image runs on defaults plus NETSTT_* env vars. Mount a file
# at /app/config.yaml to override, and it is picked up automatically.
CMD ["python", "app.py"]
