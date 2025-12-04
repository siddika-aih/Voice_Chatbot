# test_rest.py - Simpler REST API test
import requests
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import tempfile

def test_rest_api():
    print("🎤 Recording...")
    audio = sd.rec(int(4 * 16000), samplerate=16000, channels=1, dtype='float32')
    sd.wait()
    print("✔ Done")
    
    # Save to file
    audio_int16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp_file.name, 16000, audio_int16)
    
    # Send to API
    print("📤 Sending to API...")
    with open(temp_file.name, 'rb') as f:
        response = requests.post(
            "http://localhost:8000/api/voice-query",
            files={"audio": f}
        )
    
    result = response.json()
    print(f"\n👤 You: {result['transcript']}")
    print(f"🤖 Bot: {result['response']}\n")

if __name__ == "__main__":
    test_rest_api()
