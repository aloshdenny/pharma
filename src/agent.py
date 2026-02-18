import logging
import sys
from dotenv import load_dotenv

load_dotenv()

# Ensure logs are visible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)

from decouple import config
from livekit.agents import (
    JobContext,
    AgentServer,
    cli,
    llm,
    room_io,
)
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import openai, deepgram, elevenlabs, silero

# Optional: noise cancellation improves audio quality (pip install livekit-plugins-noise-cancellation)
try:
    from livekit.plugins import noise_cancellation
    _HAS_NOISE_CANCELLATION = True
except ImportError:
    _HAS_NOISE_CANCELLATION = False


from rag import pinecone_search


logger = logging.getLogger("voice-agent")

# Required env vars for STT, TTS, LLM, RAG - fail fast if missing
_REQUIRED_KEYS = [
    ("DEEPGRAM_API_KEY", "STT (speech-to-text)"),
    ("ELEVEN_API_KEY", "TTS (text-to-speech)"),
    ("GROQ_API_KEY", "LLM"),
    ("PINECONE_API_KEY", "RAG search"),
    ("PINECONE_HOST", "RAG search"),
    ("PINECONE_NAMESPACE", "RAG search"),
]


def _validate_env():
    """Validate required API keys at startup. Exits with clear error if any missing."""
    missing = []
    for key, desc in _REQUIRED_KEYS:
        val = config(key, default="")
        if not val or not str(val).strip():
            missing.append(f"  {key} ({desc})")
    if missing:
        msg = "Missing required env vars. Set them in .env:\n" + "\n".join(missing)
        print(f"[PHARMA] >>> ERROR: {msg}", flush=True)
        raise SystemExit(1)


server = AgentServer()


def prewarm(proc):
    """Prewarm VAD for faster startup."""
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="pharmacy-agent")
async def pharmacy_agent(ctx: JobContext):
    print(f"[PHARMA] >>> Entrypoint called for room {ctx.room.name}", flush=True)
    logger.info(f"Starting agent for room {ctx.room.name}")

    try:
        initial_ctx = llm.ChatContext()
        
        # New Pharmacy Insurance Approval System Prompt
        system_prompt = (
            "You are a pharmacy insurance approval agent. Your goal is to check if a patient is cleared "
            "to use a specific class of drugs based on their insurance tier. "
            "When a patient name or ID is provided, look up their insurance tier and verify if the requested "
            "drug class is allowed.\n\n"
            "Here is the database you have access to:\n"
            "Insurance Tiers:\n"
            "- Basic Tier (TIER_BASIC): Allowed drug classes: Antipyretics, Analgesics, Antibiotics.\n"
            "- Premium Tier (TIER_PREMIUM): Allowed drug classes: Antipyretics, Analgesics, Antibiotics, "
            "Antihypertensives, Antidiabetics, Cardiac Drugs, Oncology Drugs, Biologics.\n\n"
            "Patients:\n"
            "- P001: Rahul Menon (TIER_BASIC)\n"
            "- P002: Anita Sharma (TIER_PREMIUM)\n"
            "- P003: Vikram Iyer (TIER_BASIC)\n"
            "- P004: Neha Kapoor (TIER_PREMIUM)\n"
            "- P005: Suresh Nair (TIER_BASIC)\n\n"
            "Procedures:\n"
            "1. Identify the patient using their Name or ID.\n"
            "2. Identify the drug class being requested.\n"
            "3. Check if the drug class is in the list of allowed classes for the patient's tier.\n"
            "4. Inform the user whether the drug is approved or if it's rejected due to tier constraints.\n\n"
            "Be professional, clear, and efficient."
            "Don't use short forms like e.g or i.e etc. Use fullforms, for example, milligram instead of mg, milliletre instead of ml, and so on." \
            "Keep sentences short and to the point, as if you are speaking to a patient on the phone. Do not provide long explanations or use bullet points. Only say what is necessary for the current turn, and ask one question at a time if you need more information."
        )
        
        initial_ctx.add_message(
            content=system_prompt,
            role="system",
        )

        class PharmacyTools:
            @llm.function_tool(
                description="Search for relevant pharmacy and PBM rejection call records to find context about drug rejections, insurance plans, and PBM policies."
            )
            def search_records(self, query: str, top_k: int = 3):
                logger.info(f"Searching Pinecone for: {query}")
                try:
                    results = pinecone_search(query, top_k)
                except Exception as e:
                    logger.exception("RAG pinecone_search failed")
                    print(f"[PHARMA] >>> RAG ERROR: {e}", flush=True)
                    return f"Search failed: {e}. Please try rephrasing or ask without database lookup."
                if not results:
                    return "No relevant records found."
                return "\n\n".join(results)

        pharmacy_tools = PharmacyTools()
        vad = ctx.proc.userdata.get("vad") or silero.VAD.load()

        session = AgentSession(
            stt=deepgram.STT(
                model="nova-3",
                language="en-US",
                api_key=config("DEEPGRAM_API_KEY"),
            ),
            llm=openai.LLM(
                base_url="https://api.groq.com/openai/v1",
                api_key=config("GROQ_API_KEY"),
                model="llama-3.3-70b-versatile",
            ),
            tts=elevenlabs.TTS(
                api_key=config("ELEVEN_API_KEY"),
                voice_id="EzoxNTKsg4JNN7wxAgut",
            ),
            vad=vad,
            turn_detection="vad",
            allow_interruptions=True,
            tools=[pharmacy_tools.search_records],
        )

        class PharmacyAgent(Agent):
            async def on_enter(self) -> None:
                """Greet immediately via direct TTS - publishes audio track for playground."""
                print("[PHARMA] >>> on_enter called, saying greeting", flush=True)
                self.session.say("Hello! This is a pharmacy insurance approval agent. Please provide the patient's name or ID and the drug class you're inquiring about.")

        agent = PharmacyAgent(
            instructions="You are a professional pharmacy insurance approval agent. Verify drug coverage based on patient tiers.",
            chat_ctx=initial_ctx,
        )

        room_opts = room_io.RoomOptions()
        if _HAS_NOISE_CANCELLATION:
            room_opts = room_io.RoomOptions(
                audio_input=room_io.AudioInputOptions(
                    noise_cancellation=noise_cancellation.BVC(),
                ),
            )

        # session.start() connects to the room automatically - do NOT call ctx.connect()
        # (see livekit agents basic_agent.py and docs: AgentSession connects when started)
        print("[PHARMA] >>> Starting session (connects to room automatically)...", flush=True)
        await session.start(
            agent=agent,
            room=ctx.room,
            room_options=room_opts,
        )
        print(f"[PHARMA] >>> Agent session started for room {ctx.room.name}", flush=True)
        logger.info(f"Agent session started for room {ctx.room.name}")
    except Exception as e:
        print(f"[PHARMA] >>> ERROR: {e}", flush=True)
        logger.exception("Agent failed")
        raise


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "download-files":
        print("Downloading Silero VAD model...", flush=True)
        silero.VAD.load()
        print("Done.", flush=True)
        sys.exit(0)
    _validate_env()
    print("[PHARMA] >>> Env validation OK (STT, TTS, LLM, RAG keys present)", flush=True)
    cli.run_app(server)
