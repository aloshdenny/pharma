# Deploying the Pharma Voice Agent

This project has two distinct components that must be deployed separately due to their different runtime requirements:

1.  **Frontend & Token Server** (`playground/server.py` + `playground/static`): Serves the UI and mints LiveKit tokens.
    -   **Deploy to:** Vercel, Netlify, or similar serverless platforms.
2.  **Voice Agent** (`src/agent.py`): The AI logic that processes audio and speaks back.
    -   **Deploy to:** A platform that supports long-running processes (Fly.io, Railway, DigitalOcean, AWS EC2). **Cannot** be deployed to Vercel.

---

## Part 1: Deploying Frontend to Vercel

The frontend (UI) and token server are stateless and fit perfectly on Vercel.

1.  **Push your code** to GitHub.
2.  **Import the project** in Vercel.
3.  **Configure Build Settings**:
    -   **Framework Preset**: Other
    -   **Build Command**: (Leave empty)
    -   **Deepgram/ElevenLabs keys**: NOT needed on Vercel (only on the Agent).
    -   **Install Command**: `pip install -r requirements-vercel.txt` *(Crucial: This avoids installing heavy ML libraries like torch on the frontend server)*.
    -   **Output Directory**: `.` (default)
4.  **Environment Variables**:
    Add the following variables in Vercel Project Settings:
    -   `LIVEKIT_URL`
    -   `LIVEKIT_API_KEY`
    -   `LIVEKIT_API_SECRET`
5.  **Deploy**.

Your Vercel URL will now serve the playground UI. However, for the agent to actually join the room and speak, you must complete Part 2.

---

## Part 2: Deploying the Agent (Backend)

The agent needs to maintain a persistent WebSocket connection to LiveKit Cloud. Vercel functions time out after a few seconds, so they cannot run the agent.

### Option A: Run Locally (Easiest for testing)

Simply run the agent on your computer:
```bash
python src/agent.py dev
```
As long as this terminal is open, the agent will pick up calls from your Vercel-deployed frontend.

### Option B: Deploy to Render (Recommended for stability and ease of use)

Render can host the agent as a persistent background worker.

1.  **Push your code** to GitHub.
2.  **Sign up/Log in** to [Render](https://render.com/).
3.  **Click "New +" -> "Blueprint"**.
4.  **Connect your repository**.
5.  Render will detect the `render.yaml` file.
6.  **Set Environment Variables**:
    You will need to manually enter the values for:
    -   `LIVEKIT_URL`
    -   `LIVEKIT_API_KEY`
    -   `LIVEKIT_API_SECRET`
    -   `DEEPGRAM_API_KEY`
    -   `ELEVEN_API_KEY`
    -   `GROQ_API_KEY`
    -   `PINECONE_API_KEY`
    -   `PINECONE_HOST`
    -   `PINECONE_NAMESPACE`
7.  **Click "Apply"**.
    Render will build the Docker container and start your agent.

**Note:** A standard Background Worker on Render costs ~$7/month. This ensures the agent stays online 24/7 to answer calls.

*(Alternatively, you can try deploying as a "Web Service" on the free tier, but it will spin down after 15 minutes of inactivity, causing the first call to fail or time out)*.
