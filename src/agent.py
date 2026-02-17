import logging
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()
from decouple import config
from livekit.agents import (
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import openai, deepgram, elevenlabs, silero

# Import RAG function
try:
    from src.rag import pinecone_search
except ImportError:
    from rag import pinecone_search


# Configure logging
logger = logging.getLogger("voice-agent")
logger.setLevel(logging.INFO)

def prewarm(proc: JobProcess):
    proc.userdata["first_participant_joined"] = False

async def entrypoint(ctx: JobContext):
    logger.info(f"Connecting to room {ctx.room.name}")
    await ctx.connect()
    
    # Context regarding the user
    initial_ctx = llm.ChatContext()
    initial_ctx.add_message(
        content="You are a helpful pharmacy assistant. You can answer questions about drug rejections, insurance, and more. You have access to a database of past rejection calls.",
        role="system",
    )

    # 1. STT - Deepgram
    stt = deepgram.STT(
        model="nova-2-general", 
        language="en-US",
    )

    # 2. LLM - Groq (via OpenAI plugin)
    llm_plugin = openai.LLM(
        base_url="https://api.groq.com/openai/v1",
        api_key=config("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
    )

    # 3. TTS - ElevenLabs
    tts = elevenlabs.TTS(
        api_key=config("ELEVEN_API_KEY"),
        voice_id="TX3LPaxmHKxFdv7VOQHJ",
    )
    
    # 4. VAD - Silero
    vad = silero.VAD.load()

    # 4. RAG Tool
    class PharmacyTools:
        @llm.function_tool(description="Search for relevant pharmacy and PBM rejection call records to find context about drug rejections, insurance plans, and PBM policies.")
        def search_records(self, query: str, top_k: int = 3):
            logger.info(f"Searching Pinecone for: {query}")
            results = pinecone_search(query, top_k)
            if not results:
                return "No relevant records found."
            return "\n\n".join(results)

    pharmacy_tools = PharmacyTools()

    # 5. Agent & Session
    class PharmacyAgent(Agent):
        def stt_node(self, audio, model_settings):
            async def _audio_wrapper(audio_stream):
                frame_count = 0
                async for frame in audio_stream:
                    frame_count += 1
                    if frame_count % 50 == 0:
                        print(f"DEBUG: Received {frame_count} audio frames", end="\r")
                    yield frame

            async def _logging_stt_node(audio, model_settings):
                async for event in super(PharmacyAgent, self).stt_node(_audio_wrapper(audio), model_settings):
                    if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                        print(f"🎤 STT: {event.alternatives[0].text}")
                    elif event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
                        print(f"   stt: {event.alternatives[0].text}", end="\r")
                    yield event
            return _logging_stt_node(audio, model_settings)

        def llm_node(self, chat_ctx, tools, model_settings):
            async def _logging_llm_node(chat_ctx, tools, model_settings):
                print(f"🧠 LLM: Processing query...")
                full_response = ""
                async for chunk in super(PharmacyAgent, self).llm_node(chat_ctx, tools, model_settings):
                    if isinstance(chunk, llm.ChatChunk) and chunk.choices:
                        content = chunk.choices[0].delta.content
                        if content:
                            full_response += content
                            print(content, end="", flush=True)
                    yield chunk
                print("\n") # Newline after LLM stream
            return _logging_llm_node(chat_ctx, tools, model_settings)

        def tts_node(self, text, model_settings):
            async def _logging_tts_node(text, model_settings):
                print(f"🗣️  TTS: Synthesizing...")
                async for frame in super(PharmacyAgent, self).tts_node(text, model_settings):
                    yield frame
            return _logging_tts_node(text, model_settings)

    agent = PharmacyAgent(
        instructions="You are a helpful pharmacy assistant.",
        stt=stt,
        llm=llm_plugin,
        tts=tts,
        vad=vad,
        chat_ctx=initial_ctx,
        tools=[pharmacy_tools.search_records],
        turn_detection="vad", # Use VAD for turn detection
    )
    
    session = AgentSession(
        allow_interruptions=True,
    )
    
    # Start the session
    # AgentSession.start attaches to the room
    await session.start(agent, room=ctx.room)
    
    # Greeting (Agent doesn't have say(), using session capabilities if possible, or just wait for user)
    # If we want to greet, we might need to push a message to LLM or TTS directly.
    # For now, we wait for user.

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
