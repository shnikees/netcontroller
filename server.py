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

"""HTTP + WebSocket server for the net control dashboard."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from callsign_match import CallsignMatcher, RosterEntry
from feedback import FeedbackLog, record_correction
from health import HealthMonitor
from transcript_store import TranscriptStore

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


class CorrectionRequest(BaseModel):
    entry_id: int
    callsign: str


class Broadcaster:
    """Fan-out of new transcript entries to every connected dashboard."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def register(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._clients.add(websocket)

    async def unregister(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(websocket)

    async def broadcast(self, message: dict) -> None:
        async with self._lock:
            clients = list(self._clients)
        for client in clients:
            try:
                await client.send_json(message)
            except Exception:
                # A dead socket is normal (tab closed); drop it and move on.
                await self.unregister(client)


def create_app(
    store: TranscriptStore,
    roster: list[RosterEntry],
    broadcaster: Broadcaster,
    export_dir: str = ".",
    matcher: CallsignMatcher | None = None,
    feedback: FeedbackLog | None = None,
    health: HealthMonitor | None = None,
) -> FastAPI:
    app = FastAPI(title="Ham Net STT")
    by_callsign = {e.callsign: e for e in roster}

    @app.middleware("http")
    async def no_cache(request, call_next):  # noqa: ANN001, ANN202
        """Never let a browser hold a stale dashboard.

        The operator reloads the tab when something looks wrong; serving them
        yesterday's JS at that moment is the worst possible answer.
        """
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/history")
    async def history() -> JSONResponse:
        return JSONResponse(
            {
                "entries": store.all(),
                "roster": [
                    {"callsign": e.callsign, "name": e.name} for e in roster
                ],
                "check_ins": store.check_ins(),
            }
        )

    @app.post("/api/correct")
    async def correct(payload: CorrectionRequest) -> JSONResponse:
        """Apply an operator correction, and learn from it.

        Three things happen, in order: the log line is fixed so the display is
        right, the correction is appended to the feedback log, and the matcher
        learns the alias so the *next* transmission from that station matches on
        its own. The alias is what makes this worth clicking twice.
        """
        entry = store.get(payload.entry_id)
        if entry is None:
            return JSONResponse(
                {"error": f"No entry {payload.entry_id}"}, status_code=404
            )
        callsign = payload.callsign.strip().upper()
        if callsign not in by_callsign:
            return JSONResponse(
                {"error": f"{callsign} is not on the roster"}, status_code=400
            )

        was = entry.matched_callsign
        candidate = entry.candidate
        raw_text = entry.raw_text
        station = by_callsign[callsign]
        store.correct(payload.entry_id, callsign, station.name)

        if feedback is not None:
            record_correction(
                feedback,
                entry_id=payload.entry_id,
                candidate=candidate,
                from_callsign=was,
                to_callsign=callsign,
                raw_text=raw_text,
                confidence=entry.confidence,
                clip_duration=entry.clip_duration,
            )

        learned = False
        if matcher is not None:
            learned = matcher.learn_alias(candidate, callsign)
        log.info(
            "Correction: entry %d %s -> %s%s",
            payload.entry_id,
            was or "unmatched",
            callsign,
            f" (learned {candidate} -> {callsign})" if learned else "",
        )

        await broadcaster.broadcast(
            {"type": "correction", "entry": entry.to_dict(), "learned": learned}
        )
        return JSONResponse(
            {"entry": entry.to_dict(), "learned": learned, "alias": candidate}
        )

    @app.get("/api/health")
    async def health_check() -> JSONResponse:
        """Pipeline health, for the dashboard banner and for external checks.

        Returns 503 when the pipeline is in error, so a container healthcheck,
        systemd, or a one-line cron `curl -f` can act on it without parsing the
        body.
        """
        if health is None:
            return JSONResponse({"state": "unknown", "issues": []})
        snapshot = health.snapshot()
        return JSONResponse(
            snapshot.to_dict(), status_code=503 if snapshot.state == "error" else 200
        )

    @app.get("/api/aliases")
    async def aliases() -> JSONResponse:
        """What the matcher has learned so far, for inspection during a net."""
        return JSONResponse({"aliases": matcher.aliases if matcher else {}})

    @app.post("/api/export")
    async def export() -> JSONResponse:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = Path(export_dir) / f"net-log-{stamp}"
        written = [
            str(store.export_csv(base.with_suffix(".csv"))),
            str(store.export_text(base.with_suffix(".txt"))),
        ]
        log.info("Exported session log: %s", ", ".join(written))
        return JSONResponse({"files": written, "entries": len(store.entries)})

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await broadcaster.register(websocket)
        try:
            await websocket.send_json(
                {
                    "type": "history",
                    "entries": store.all(),
                    "health": health.snapshot().to_dict() if health else None,
                }
            )
            while True:
                # The dashboard is read-only; this just keeps the socket open
                # and gives us a clean disconnect signal.
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            await broadcaster.unregister(websocket)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
