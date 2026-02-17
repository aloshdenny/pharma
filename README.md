# Pharma Voice Agent

A LiveKit voice agent for pharmacy assistance. It helps with drug rejections, insurance questions, and PBM policies using a RAG-powered database of past rejection call records.

## Features

- **Voice AI**: Real-time speech-to-text (Deepgram), LLM (Groq), and text-to-speech (ElevenLabs)
- **RAG Search**: Semantic search over Pinecone for pharmacy and PBM rejection records
- **LiveKit Playground**: Test the agent in a web-based playground with audio and text chat

## Prerequisites

- **Python** 3.10–3.14
- **Poetry** (recommended) or pip
- API keys for:
  - [LiveKit Cloud](https://cloud.livekit.io/) (free tier works)
  - [Deepgram](https://deepgram.com/) (STT)
  - [ElevenLabs](https://elevenlabs.io/) (TTS)
  - [Groq](https://groq.com/) (LLM)
  - [Pinecone](https://www.pinecone.io/) (vector DB for RAG)

---

## Project Setup

### 1. Clone and install dependencies

```bash
cd pharma
poetry install
```

Or with pip:

```bash
pip install -r requirements.txt
```

### 2. Environment variables

Create a `.env` file in the project root with:

```env
# LiveKit (from https://cloud.livekit.io/)
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret
LIVEKIT_LOG_LEVEL=DEBUG

# Optional: custom room name (default: pharmacy-test-room)
# LIVEKIT_ROOM=pharmacy-test-room

# Deepgram (Speech-to-Text)
DEEPGRAM_API_KEY=your_deepgram_api_key

# ElevenLabs (Text-to-Speech)
ELEVEN_API_KEY=your_elevenlabs_api_key

# Groq (LLM)
GROQ_API_KEY=your_groq_api_key

# Pinecone (RAG search)
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_HOST=https://your-index.svc.region.pinecone.io
PINECONE_NAMESPACE=your_namespace
```

### 3. Pre-download Silero VAD model (optional, for faster startup)

```bash
poetry run python src/agent.py download-files
```

---

## Running the Agent

### 1. Start the agent in dev mode

```bash
poetry run python src/agent.py dev
```

You should see:

```
[PHARMA] >>> Env validation OK (STT, TTS, LLM, RAG keys present)
...
registered worker {"agent_name": "pharmacy-agent", ...}
```

Keep this terminal running.

### 2. Generate a playground token

In a **new terminal**:

```bash
poetry run python scripts/generate_token.py
```

This prints:
- **Server URL** (your `LIVEKIT_URL`)
- **Token** (JWT with agent dispatch for `pharmacy-agent`)

Copy both values.

### 3. Connect via LiveKit Playground

1. Open the [LiveKit Agents Playground](https://agents-playground.livekit.io/)
2. Select **Manual** mode (not Cloud)
3. Paste the **Server URL** and **Token** from step 2
4. Click **Connect**
5. Allow microphone access when prompted
6. Wait 10–20 seconds on first connect (cold start on free tier)
7. Speak or type to interact with the pharmacy assistant

---

## Project Structure

```
pharma/
├── src/
│   ├── agent.py          # Main voice agent (LiveKit AgentServer)
│   ├── rag.py             # Pinecone RAG search
│   ├── stt.py             # Standalone STT utilities
│   ├── tts.py             # Standalone TTS utilities
│   └── client.py          # LiveKit room client
├── scripts/
│   └── generate_token.py  # Token generation for playground
├── pinecone/
│   ├── pinecone_upsert.py # Indexing documents
│   └── pinecone_query.py # Query testing
├── data/                  # Sample/mock data
├── .env                   # Your credentials (create from template above)
├── pyproject.toml
└── requirements.txt
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Missing required env vars` | Ensure all keys in `.env` are set and non-empty |
| No audio / no response | Check agent logs for STT/TTS/LLM errors; verify API keys |
| Agent never joins room | Ensure token includes `RoomAgentDispatch(agent_name="pharmacy-agent")` |
| RAG search fails | Verify Pinecone index has inference endpoint; check `PINECONE_NAMESPACE` |
| 10–20 sec delay on connect | Normal on free tier (agent cold start) |

---

## API Reference

- [LiveKit Agents](https://docs.livekit.io/agents/)
- [LiveKit Playground](https://docs.livekit.io/agents/start/playground/)
- [Agent Dispatch](https://docs.livekit.io/agents/server/agent-dispatch/)
