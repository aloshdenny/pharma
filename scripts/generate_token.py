#!/usr/bin/env python3
"""
Generate a LiveKit token for the playground with agent dispatch.

Usage:
    poetry run python scripts/generate_token.py          # print token to console
    poetry run python scripts/generate_token.py --serve   # launch playground UI

Use the printed token + LIVEKIT_URL in the playground (Manual mode),
or just run --serve and open http://localhost:8080 — a fresh token is
generated on every page load / refresh automatically.
"""
import os
import sys
import uuid
import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from livekit.api import AccessToken, VideoGrants
from livekit.protocol.room import RoomConfiguration
from livekit.protocol.agent_dispatch import RoomAgentDispatch

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
    print("Error: Set LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET in .env")
    sys.exit(1)


def make_token(room_name: str | None = None, identity: str | None = None) -> str:
    """Create a signed JWT with agent dispatch for pharmacy-agent."""
    room_name = room_name or f"pharma-{uuid.uuid4().hex[:8]}"
    identity = identity or f"user-{uuid.uuid4().hex[:6]}"
    return (
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


if __name__ == "__main__":
    if "--serve" in sys.argv:
        # Launch the playground web UI
        from playground.server import app
        import uvicorn
        port = int(os.getenv("PLAYGROUND_PORT", "8080"))
        print(f"[Playground] Open http://localhost:{port}")
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        room = os.getenv("LIVEKIT_ROOM", f"pharma-{uuid.uuid4().hex[:8]}")
        token = make_token(room_name=room)
        print("=" * 60)
        print("LIVEKIT PLAYGROUND - Manual Connect")
        print("=" * 60)
        print(f"\n1. Server URL:\n{LIVEKIT_URL}\n")
        print(f"2. Token (copy all):\n{token}\n")
        print("=" * 60)
        print("\nPaste in the playground (Manual tab) and click Connect,")
        print("or run with --serve to launch the built-in playground UI.")
        print("=" * 60)
