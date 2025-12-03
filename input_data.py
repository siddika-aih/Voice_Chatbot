import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import tempfile
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

api_key=os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

SAMPLE_RATE = 16000

def record_audio(duration=4):
    print("Speak now")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    print("✔ Recording done")
    return audio

def speech_to_text():
    flag=True
    while flag:
        choice=input("recording-Y/N:")
        if choice.upper()=="N":
            flag=False
        else:
            audio = record_audio(4)

            # Save to a temp .wav file because Gemini REQUIRES files
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            wav.write(temp.name, SAMPLE_RATE, np.int16(audio * 32767))

            print("Uploading file:", temp.name)

            uploaded = client.files.upload(file=temp.name)

            # Ask Gemini to transcribe it
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    "Transcribe this audio:",
                    uploaded
                ]
            )

            return response.text