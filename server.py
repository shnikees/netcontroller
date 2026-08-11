"""HTTP + WebSocket server for the net control dashboard."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from callsign_match import RosterEntry
from transcript_store import TranscriptStore

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


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
) -> FastAPI:
    app = FastAPI(title="Ham Net STT")

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
            await websocket.send_json({"type": "history", "entries": store.all()})
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
