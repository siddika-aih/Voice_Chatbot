from generate_rag_response import generate_rag_response
import pyttsx3

tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 180)  # Speed
tts_engine.setProperty('volume', 0.9)  # Volume

def text_to_speech(text):
    """4. NEW: Text → Speech (TTS)"""
    print("🔊 Speaking response...")
    tts_engine.say(text)
    tts_engine.runAndWait()
    print("✔ Speech complete")