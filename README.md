# Pharma Voice Agent

A LiveKit voice agent for pharmacy insurance approval. Checks whether a patient is cleared to use a drug class based on their insurance tier, with a RAG-powered database of past rejection call records.

## Features

- **Voice AI**: Real-time speech-to-text (Deepgram Nova-3), LLM (Groq / Llama 3.3 70B), and text-to-speech (ElevenLabs)
- **RAG Search**: Semantic search over Pinecone for pharmacy and PBM rejection records
- **Built-in Playground**: Self-hosted web UI — auto-connects, streams transcript word-by-word as the agent speaks, no external playground needed
- **Streaming transcript**: Agent text appears in the chat panel as tokens are generated, not after TTS finishes

## Prerequisites

- **Python** 3.10–3.14
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
pip install -r requirements.txt
```

### 2. Environment variables

Create a `.env` file in the project root:

```env
touch .env
```

### 3. Environment variables

Create a `.env` file in the project root:

```env
# LiveKit (from https://cloud.livekit.io/)
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret

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

### 4. Pre-download Silero VAD model (optional, speeds up first start)

```bash
python src/agent.py download-files
```

---

## Deploying to Vercel

The frontend (UI) and token server are stateless and fit perfectly on Vercel.

1.  **Push your code** to GitHub.
2.  **Import the project** in Vercel.
3.  **Configure Build Settings**:
    -   **Framework Preset**: Other
    -   **Build Command**: (Leave empty)
    -   **Deepgram/ElevenLabs keys**: NOT needed on Vercel (only on the Agent).
    -   **Install Command**: `pip install -r requirements-vercel.txt`
        > **CRITICAL**: You MUST set this in the Vercel Project Settings. If you leave it as default, Vercel will try to install `requirements.txt` which contains heavy libraries (Torch) that will cause the build to fail or timeout.
    -   **Output Directory**: `.` (default)
4.  **Environment Variables**:
    Add the following variables in Vercel Project Settings:
    -   `LIVEKIT_URL`
    -   `LIVEKIT_API_KEY`
    -   `LIVEKIT_API_SECRET`
5.  **Deploy**.

---

## Running Locally

Two processes need to run simultaneously — the **agent** and the **playground server**. Open two terminals in the `pharma/` directory.

### Terminal 1 — Agent

```bash
python src/agent.py dev
```

You should see:

```
[PHARMA] >>> Env validation OK (STT, TTS, LLM, RAG keys present)
...
registered worker {"agent_name": "pharmacy-agent", ...}
```

Keep this running. The agent will handle any room that dispatches `pharmacy-agent`.

### Terminal 2 — Playground UI

```bash
python playground/server.py
```

Or equivalently:

```bash
python scripts/generate_token.py --serve
```

Then open **http://localhost:8080** in your browser.

> **Important:** Always use `http://localhost:8080`, not `127.0.0.1` or a LAN IP. Browsers only allow microphone access on `localhost` or HTTPS.

The playground will:
1. Request microphone permission
2. Automatically generate a fresh LiveKit token (unique room + identity per session)
3. Connect to the room and dispatch the agent
4. Stream the transcript word-by-word as the agent responds

**Refreshing the page** starts a completely fresh call with a new token and room.

### Optional — Print a token for the LiveKit Cloud Playground

If you prefer to use the [LiveKit Agents Playground](https://agents-playground.livekit.io/) instead:

```bash
python scripts/generate_token.py
```

Copy the printed URL and token into the playground's **Manual** tab.

---

## Project Structure

```
pharma/
├── src/
│   ├── agent.py              # Main voice agent (LiveKit AgentServer, pharmacy-agent)
│   ├── rag.py                # Pinecone RAG search + Groq tool-call loop
│   ├── stt.py                # Standalone STT utilities
│   └── tts.py                # Standalone TTS utilities
├── playground/
│   ├── server.py             # FastAPI server — serves UI + /api/token endpoint
│   ├── static/
│   │   └── index.html        # Single-page playground UI (LiveKit JS SDK)
│   └── __init__.py
├── scripts/
│   └── generate_token.py     # Print a token, or --serve to launch playground
├── pinecone/
│   ├── pinecone_upsert.py    # Index documents into Pinecone
│   └── pinecone_query.py     # Test Pinecone queries
├── data/                     # Sample / mock data
├── .env                      # Your credentials (gitignored)
├── pyproject.toml
└── requirements.txt
```

---

## How the Token / Room Flow Works

```
Browser refresh
  └─► GET /api/token          (playground/server.py)
        └─► mints JWT with unique room name + RoomAgentDispatch(pharmacy-agent)
              └─► LiveKit room created on connect
                    └─► Agent worker picks up the dispatch
                          └─► pharmacy-agent joins the room
```

Each page load / refresh creates an isolated session. There is no shared state between calls.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Missing required env vars` | Ensure all keys in `.env` are set and non-empty |
| `getUserMedia` / mic error | Open on `http://localhost:8080` (not an IP address) |
| No audio from agent | Check agent logs for TTS/LLM errors; ensure ElevenLabs key has `speech` scope |
| No transcript / silence | Groq model returned empty content — confirm model is not a reasoning-only model |
| Agent never joins room | Check agent is running and token includes `RoomAgentDispatch(agent_name="pharmacy-agent")` |
| RAG search fails | Verify Pinecone index has inference endpoint enabled; check `PINECONE_NAMESPACE` |
| 10–20 sec delay on first connect | Normal cold-start on LiveKit free tier |
| Mute button stuck | Browser may have blocked re-acquiring mic — reload the page |

---

## API Reference

- [LiveKit Agents](https://docs.livekit.io/agents/)
- [Agent Dispatch](https://docs.livekit.io/agents/server/agent-dispatch/)
- [LiveKit JS SDK](https://docs.livekit.io/client-sdk-js/)
- [Deepgram Nova-3](https://developers.deepgram.com/docs/models)
- [ElevenLabs TTS Streaming](https://elevenlabs.io/docs/eleven-api/guides/cookbooks/text-to-speech/streaming)
- [Groq API](https://console.groq.com/docs/openai)