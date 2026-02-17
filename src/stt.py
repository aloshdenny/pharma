import os
import json
import queue
import threading
import sounddevice as sd
from websocket import create_connection
from decouple import config
# -------------------------------------------------------------------
# CONFIG — replace or export your Deepgram API key first:
#   export DEEPGRAM_API_KEY="YOUR_DEEPGRAM_API_KEY"
# -------------------------------------------------------------------
API_KEY = config("DEEPGRAM_API_KEY")

# WebSocket endpoint with Nova-3 and interim results turned on
WS_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?model=nova-3"
    "&language=en-US"
    "&interim_results=true"
    "&encoding=linear16"
    "&sample_rate=16000"
)

audio_queue = queue.Queue()
stop_event = threading.Event()

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
BLOCK_SIZE = 1024

def mic_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def receive_loop(ws):
    last_interim = ""

    while not stop_event.is_set():
        try:
            msg = ws.recv()
            if not msg:
                continue
            data = json.loads(msg)

            # Print metadata or other non-transcript messages for debugging
            if "channel" not in data:
                if "metadata" in data:
                    print(f"DEBUG: Connection Metadata: {data['metadata']}")
                continue

            alt = data["channel"]["alternatives"][0]
            transcript = alt.get("transcript", "")

            if not transcript:
                continue

            if data.get("is_final", False) or data.get("speech_final", False):
                # Final transcript
                print("\rFINAL   →", transcript.strip() + " " * 20)
                last_interim = ""
            else:
                # Interim transcript (overwrite same line)
                if transcript != last_interim:
                    print("\rLIVE    →", transcript, end="", flush=True)
                    last_interim = transcript

        except Exception as e:
            if not stop_event.is_set():
                print(f"\n[Receiver Error]: {e}")
            break

def main():
    print("🎙️  Listening... Press Ctrl+C to stop.\n")

    ws = create_connection(
        WS_URL,
        header=[f"Authorization: Token {API_KEY}"],
    )

    receiver = threading.Thread(target=receive_loop, args=(ws,), daemon=True)
    receiver.start()
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=mic_callback,
        ):
            print("🟢 Audio stream started. Sending data to Deepgram...")
            while not stop_event.is_set():
                try:
                    audio = audio_queue.get(timeout=1.0)
                    if ws.connected:
                        ws.send(audio.tobytes(), opcode=0x2)
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Error sending audio: {e}")
                    break

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stop_event.set()
        ws.close()

if __name__ == "__main__":
    main()