import asyncio
import os
import signal
import pyaudio
from livekit import rtc
from livekit.api import AccessToken, VideoGrants
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

SAMPLE_RATE = 16000
NUM_CHANNELS = 1
SAMPLES_PER_CHANNEL = 480  # 30ms at 16kHz

async def capture_microphone(source: rtc.AudioSource):
    """
    Captures audio from the default microphone using PyAudio and forwards it to the LiveKit AudioSource.
    """
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=NUM_CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=SAMPLES_PER_CHANNEL,
    )

    print(f"Sampling microphone at {SAMPLE_RATE}Hz, {NUM_CHANNELS} channel(s)...")

    loop = asyncio.get_running_loop()
    from functools import partial

    try:
        while True:
            # Read raw bytes from PyAudio stream in a separate thread to avoid blocking the event loop
            read_func = partial(stream.read, SAMPLES_PER_CHANNEL, exception_on_overflow=False)
            data = await loop.run_in_executor(None, read_func)

            rms = calculate_rms(data)
            if rms > 0.01:
                print(f"🎤 Mic Level: {rms:.3f}", end="\r")
            
            # Create a LiveKit AudioFrame
            frame = rtc.AudioFrame(
                data=data,
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                samples_per_channel=SAMPLES_PER_CHANNEL,
            )
            
            # Push frame to the source
            await source.capture_frame(frame)
            # Yield control to event loop (implicit in await, but good practice)
            # await asyncio.sleep(0) 

    except asyncio.CancelledError:
        print("Microphone capture stopped.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

async def main():
    if not LIVEKIT_URL or not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        print("Error: Missing LIVEKIT_URL, LIVEKIT_API_KEY, or LIVEKIT_API_SECRET in environment variables.")
        return

    # 1. Generate Token
    grant = VideoGrants(room_join=True, room="pharmacy-test-room")
    token = (
        AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity("client-python")
        .with_grants(grant)
        .to_jwt()
    )

    # 2. Connect to Room
    room = rtc.Room()

    @room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            print(f"Track subscribed: {track.kind} from {participant.identity}")
            print("Receiving audio... Starting playback.")
            asyncio.create_task(play_audio(track))

    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        print(f"Participant connected: {participant.identity}")

    @room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        print(f"Participant disconnected: {participant.identity}")

    print(f"Connecting to {LIVEKIT_URL}...")
    try:
        await room.connect(LIVEKIT_URL, token)
        print(f"Connected to room: {room.name}")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    # 3. Publish Microphone
    try:
        print("Publishing microphone...")
        # Create AudioSource
        source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
        # Create Track
        track = rtc.LocalAudioTrack.create_audio_track("mic", source)
        # Publish Track
        await room.local_participant.publish_track(track)
        print("Microphone published. You can speak now!")

        # Start capturing audio in background task
        capture_task = asyncio.create_task(capture_microphone(source))

    except Exception as e:
        print(f"Failed to publish microphone: {e}")
        await room.disconnect()
        return

    # Keep connection alive until interrupted
    try:
        stop_event = asyncio.Event()
        
        # Handle graceful shutdown
        loop = asyncio.get_running_loop()
        def signal_handler():
            stop_event.set()
        
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)
            
        await stop_event.wait()
    finally:
        print("Disconnecting...")
        capture_task.cancel()
        try:
            await capture_task
        except asyncio.CancelledError:
            pass
        await room.disconnect()

async def play_audio(track: rtc.Track):
    """
    Receives audio from a LiveKit track and plays it using PyAudio.
    """
    print(f"Starting playback for track: {track.sid}")
    audio_stream = rtc.AudioStream(track)
    p = pyaudio.PyAudio()
    output_stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=audio_stream._sample_rate, # Usually 48000 or 24000
        output=True,
    )

    try:
        async for event in audio_stream:
            # event.frame.data is a memoryview of int16
            # We need to convert it to bytes for PyAudio
            data = bytes(event.frame.data)
            await asyncio.get_running_loop().run_in_executor(None, output_stream.write, data)
    except Exception as e:
        print(f"Playback error: {e}")
    finally:
        print(f"Stopping playback for track: {track.sid}")
        output_stream.stop_stream()
        output_stream.close()
        p.terminate()

def calculate_rms(data):
    import math
    import struct
    count = len(data) / 2
    format = "%dh" % count
    shorts = struct.unpack(format, data)
    sum_squares = 0.0
    for sample in shorts:
        n = sample * (1.0 / 32768.0)
        sum_squares += n * n
    return math.sqrt(sum_squares / count)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Should be handled by signal handler, but just in case
        pass
