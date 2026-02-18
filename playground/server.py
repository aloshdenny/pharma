#!/usr/bin/env python3
"""
Playground server — serves the UI and generates LiveKit tokens.

Run:
    poetry run python playground/server.py
    # or
    python playground/server.py
"""

import os
import sys
import uuid
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from livekit.api import AccessToken, VideoGrants
from livekit.protocol.room import RoomConfiguration
from livekit.protocol.agent_dispatch import RoomAgentDispatch

app = FastAPI(title="Pharmacy Agent Playground")

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/api/token")
async def get_token():
    """Generate a fresh LiveKit token with agent dispatch.
    Each call produces a unique room + identity so refresh = new session."""
    if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
        return JSONResponse(
            {"error": "Server missing LIVEKIT env vars"},
            status_code=500,
        )

    room_name = f"pharma-{uuid.uuid4().hex[:8]}"
    identity = f"user-{uuid.uuid4().hex[:6]}"

    token = (
        AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_grants(VideoGrants(room_join=True, room=room_name))
        .with_room_config(
            RoomConfiguration(
                agents=[RoomAgentDispatch(agent_name="pharmacy-agent")],
            )
        )
        .with_ttl(datetime.timedelta(hours=1))
        .to_jwt()
    )

    return {
        "token": token,
        "url": LIVEKIT_URL,
        "room": room_name,
        "identity": identity,
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(html_path.read_text())


# Serve static assets (JS, CSS, etc.)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PLAYGROUND_PORT", "8080"))
    print(f"[Playground] Starting on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
