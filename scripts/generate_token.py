#!/usr/bin/env python3
"""
Generate a LiveKit token for the playground with agent dispatch.
Run: poetry run python scripts/generate_token.py

Use this token + LIVEKIT_URL in the playground (Manual mode).
"""
import os
import sys

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
ROOM_NAME = os.getenv("LIVEKIT_ROOM", "pharmacy-test-room")

if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
    print("Error: Set LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET in .env")
    sys.exit(1)

# Token with agent dispatch - dispatches pharmacy-agent when participant connects
token = (
    AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    .with_identity("playground-user")
    .with_grants(VideoGrants(room_join=True, room=ROOM_NAME))
    .with_room_config(
        RoomConfiguration(
            agents=[RoomAgentDispatch(agent_name="pharmacy-agent")],
        )
    )
    .with_ttl(__import__("datetime").timedelta(hours=1))
    .to_jwt()
)

print("=" * 60)
print("LIVEKIT PLAYGROUND - Manual Connect")
print("=" * 60)
print(f"\n1. Server URL:\n{LIVEKIT_URL}\n")
print(f"2. Token (copy all):\n{token}\n")
print("=" * 60)
print("\nPaste these in the playground (Manual tab) and click Connect.")
print("=" * 60)
