"""FastAPI entry point.

Run on your LAN:  uvicorn app.main:app --host 0.0.0.0 --port 8000
Phones on the same Wi-Fi open http://<laptop-ip>:8000
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse, PlainTextResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from . import db, game, llm, rooms  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("control")

STATIC = Path(__file__).resolve().parent.parent / "static"


DEBUG_ENDPOINTS = os.getenv("DEBUG_ENDPOINTS", "").lower() in ("1", "true", "yes")


@asynccontextmanager
async def lifespan(app: FastAPI):
    async def sweeper():
        while True:
            await asyncio.sleep(60)
            rooms.sweep_empty_rooms()

    await db.connect()
    task = asyncio.create_task(sweeper())
    log.info("LLM provider: %s | storage: %s | debug endpoints: %s", llm.provider(),
             "postgres" if db.database_url() else f"sqlite ({db.SQLITE_PATH.name})", DEBUG_ENDPOINTS)
    yield
    task.cancel()
    await db.disconnect()


app = FastAPI(title="Contact / CONTROL", lifespan=lifespan)


# --- game socket ----------------------------------------------------------------

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    """First message: {"type":"create","name","group","playerId","apiKey"?}
    or {"type":"join","code","name","playerId"}. playerId is the browser's lasting device id.
    After that, every message is a room action (see Room.a_* in rooms.py)."""
    await ws.accept()
    room = None
    try:
        hello = await ws.receive_json()
        if hello.get("type") == "create":
            room, pid = await rooms.create_room(str(hello.get("group", "")), str(hello.get("name", "")),
                                                hello.get("playerId"), hello.get("apiKey"))
        elif hello.get("type") == "join":
            room, pid = rooms.join_room(str(hello.get("code", "")), str(hello.get("name", "")),
                                        hello.get("playerId"))
        else:
            raise rooms.ActionError("first message must be create or join")
        rooms.attach(room, ws, pid)
        await ws.send_json({"type": "joined", "code": room.code, "playerId": pid})
        await room.broadcast()

        while True:
            msg = await ws.receive_json()
            try:
                reply = await room.handle(pid, msg)
                if reply:
                    await ws.send_json(reply)
            except rooms.ActionError as e:
                await ws.send_json({"type": "error", "message": str(e)})
    except rooms.ActionError as e:
        await ws.send_json({"type": "error", "message": str(e)})
        await ws.close()
    except WebSocketDisconnect:
        pass
    finally:
        if room is not None:
            rooms.detach(room, ws)
            await room.broadcast()


# --- standalone backend checks --------------------------------------------------
# These spend quota, so they are off unless DEBUG_ENDPOINTS=1 (never on the public site).

class PickIn(BaseModel):
    difficulty: str = "medium"


class GuessIn(BaseModel):
    clue: str
    prefix: str
    group: str | None = None


if DEBUG_ENDPOINTS:
    @app.post("/api/debug/pick-word")
    async def debug_pick(body: PickIn):
        word, flavor, source = await game.pick_secret_word(body.difficulty, [])
        return {"word": word, "flavor": flavor, "source": source}

    @app.post("/api/debug/guess")
    async def debug_guess(body: GuessIn):
        dictionary = await db.load_dictionary(body.group) if body.group else []
        return await game.wordmaster_guess(body.prefix.upper(), body.clue, dictionary) or {"guess": None}


@app.get("/api/dictionary/{group}")
async def dictionary(group: str):
    """Every local reference saved by a group - the growing slang corpus."""
    return await db.load_dictionary(group)


@app.get("/healthz")
async def healthz():
    """Render pings this; it also wakes the service after a sleep."""
    return {"ok": True, "rooms": len(rooms.ROOMS)}


@app.get("/api/usage")
async def usage():
    """LLM calls made today - divide by games played for the calls-per-game number."""
    return llm.usage()


# --- frontend -----------------------------------------------------------------

@app.get("/")
async def index():
    page = STATIC / "contact.html"
    if page.exists():
        return FileResponse(page)
    return PlainTextResponse("Backend is up. Put contact.html in static/ to serve the game.\n"
                             "API explorer: /docs")


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
