import os
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
from decouple import config

# Load the API key from .env
api_key = config("ELEVEN_API_KEY")

# Initialize the ElevenLabs client
elevenlabs = ElevenLabs(api_key=api_key)

# Example text to convert
text = "Hello! This is a simple text to speech test using ElevenLabs."

# Replace with a voice ID you have (e.g., a default voice like "Bella" or any voice ID from your dashboard)
voice_id = "TX3LPaxmHKxFdv7VOQHJ"

# Convert text to audio bytes
audio_bytes = elevenlabs.text_to_speech.convert(
    text=text,
    voice_id=voice_id,
    model_id="eleven_multilingual_v2",  # default recommended model
    output_format="mp3_44100_128"
)

play(audio_bytes)