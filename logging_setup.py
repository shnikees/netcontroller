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

"""Console plus rotating file logging.

The console is for the operator during the net. The file is for the morning
after, when someone asks why a station is missing from the log -- so it keeps
more detail than the console does, and rotates rather than filling the disk on
a Pi that never gets rebooted.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

CONSOLE_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
FILE_FORMAT = "%(asctime)s %(levelname)-8s %(name)s [%(threadName)s]: %(message)s"

# faster-whisper narrates every clip at INFO; useful in the file, noise on the
# console where the operator is watching for their own app's messages.
NOISY_LOGGERS = (
    "faster_whisper",
    "uvicorn.error",
    "uvicorn.access",
    "httpx",
    "httpcore",
    "asyncio",
    "urllib3",
    "PIL",
)


def setup_logging(
    *,
    log_dir: str | Path | None = "logs",
    level: str = "INFO",
    file_level: str = "DEBUG",
    max_bytes: int = 5_000_000,
    backups: int = 5,
    verbose: bool = False,
) -> Path | None:
    """Configure root logging. Returns the log file path, or None if disabled.

    A file that cannot be opened (read-only media, wrong owner in a container)
    must not stop the net: it is reported on the console and the app carries on
    with console logging alone.
    """
    console_level = logging.DEBUG if verbose else _level(level)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT))
    root.addHandler(console)

    path: Path | None = None
    if log_dir:
        try:
            directory = Path(log_dir)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "net-stt.log"
            file_handler = logging.handlers.RotatingFileHandler(
                path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8"
            )
            file_handler.setLevel(_level(file_level))
            file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
            root.addHandler(file_handler)
        except OSError as exc:
            path = None
            logging.getLogger("net-stt").warning(
                "File logging disabled (%s); continuing with console only", exc
            )

    if not verbose:
        for name in NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)

    return path


def _level(name: str) -> int:
    return getattr(logging, str(name).upper(), logging.INFO)
