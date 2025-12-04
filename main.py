# from input_data import speech_to_text
# from data_store import data_extract_chunk,data_store_vectorstore
# from generate_rag_response import generate_rag_response
# from output_data import text_to_speech
# def main():
#     print("Hello from voicebot!")
#     file_path=""
#     content, embeddings = data_extract_chunk(file_path)
#     data_store_vectorstore(content, embeddings)

#     speech_to_text()
#     result=generate_rag_response(speech_to_text)
#     text_to_speech(result)


# if __name__ == "__main__":
#     main()


# streaming_voice_bot.py - Main orchestrator with async streaming
import asyncio
import sounddevice as sd
import numpy as np
from google import genai
from dotenv import load_dotenv
import os
from collections import deque
import queue
import threading

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
SAMPLE_RATE = 16000

class StreamingVoiceBot:
    def __init__(self):
        self.audio_buffer = deque(maxlen=100)
        self.is_recording = False
        self.session_context = []  # Track conversation history
        
    async def stream_audio_input(self, duration=10):
        """Continuous audio streaming with VAD"""
        audio_queue = queue.Queue()
        
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio status: {status}")
            audio_queue.put(indata.copy())
        
        # Start non-blocking recording
        stream = sd.InputStream(
            callback=audio_callback,
            channels=1,
            samplerate=SAMPLE_RATE,
            dtype='float32'
        )
        
        with stream:
            print("🎤 Listening... (speak naturally)")
            chunks = []
            silence_threshold = 0.01
            silence_duration = 0
            max_silence = 1.5  # seconds of silence to end recording
            
            while silence_duration < max_silence:
                try:
                    chunk = audio_queue.get(timeout=0.1)
                    chunks.append(chunk)
                    
                    # Simple VAD - check if audio energy below threshold
                    if np.abs(chunk).mean() < silence_threshold:
                        silence_duration += 0.1
                    else:
                        silence_duration = 0
                        
                except queue.Empty:
                    continue
                    
            print("✔ Speech detected, processing...")
            return np.concatenate(chunks) if chunks else None
    
    # async def transcribe_streaming(self, audio_data):
    #     """Fast transcription with Gemini"""
    #     import tempfile
    #     import scipy.io.wavfile as wav
        
    #     # Save temporarily
    #     temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    #     wav.write(temp.name, SAMPLE_RATE, np.int16(audio_data * 32767))
        
    #     uploaded = client.files.upload(file=temp.name)
        
    #     response = client.models.generate_content(
    #         model="gemini-2.5-flash",
    #         contents=["Transcribe this audio concisely:", uploaded]
    #     )
        
    #     os.unlink(temp.name)
    #     return response.text.strip()
    async def transcribe_streaming(self, audio_data):
        """Fast transcription with Gemini"""
        import tempfile
        import scipy.io.wavfile as wav
        
        # FIX: Normalize audio properly for float32
        if audio_data.dtype == np.float32:
            # Ensure values are in range [-1, 1]
            audio_data = np.clip(audio_data, -1.0, 1.0)
            # Convert to int16 safely
            audio_int16 = (audio_data * 32767).astype(np.int16)
        else:
            audio_int16 = audio_data.astype(np.int16)
        
        # Save temporarily with delete=True to auto-cleanup
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            temp_path = temp.name
        
        wav.write(temp_path, SAMPLE_RATE, audio_int16)
        
        try:
            uploaded = client.files.upload(file=temp_path)
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=["Transcribe this audio concisely:", uploaded]
            )
            
            return response.text.strip()
        
        finally:
            # FIX: Proper cleanup with error handling
            try:
                import time
                time.sleep(0.1)  # Small delay for file release
                os.unlink(temp_path)
            except PermissionError:
                pass  # Ignore if file still locked (Windows issue)

    # async def generate_response_streaming(self, query):
    #     """Streaming RAG + LLM response"""
    #     from data_store import hybrid_retrieve
    #     from generate_rag_response import stream_rag_response
        
    #     # Parallel RAG retrieval
    #     context_task = asyncio.create_task(hybrid_retrieve(query, top_k=5))
        
    #     # Immediate acknowledgment
    #     print("🤔 Processing your query...")
        
    #     # Wait for context
    #     context = await context_task
        
    #     # Stream response generation
    #     full_response = ""
    #     async for chunk in stream_rag_response(query, context, self.session_context):
    #         full_response += chunk
    #         print(chunk, end="", flush=True)
        
    #     print()  # New line after streaming
    #     return full_response
    async def generate_response_streaming(self, query):
        """Streaming RAG + LLM response"""
        from data_store import hybrid_retrieve
        
        # Parallel RAG retrieval
        print("🤔 Processing your query...")
        context = await hybrid_retrieve(query, top_k=5)
        
        # Build conversation history
        history_text = ""
        if self.session_context:
            recent_history = self.session_context[-3:]
            history_text = "\n".join([
                f"User: {turn['user']}\nAssistant: {turn['assistant']}"
                for turn in recent_history
            ])
        
        # Enhanced prompt
        prompt = f"""You are a helpful DCB Bank voice assistant. Answer concisely in 2-3 sentences for voice.

    CONVERSATION HISTORY:
    {history_text}

    CONTEXT:
    {context}

    USER: {query}

    ANSWER (concise):"""
        
        # FIX: Non-streaming response for stability
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt]
        )
        
        full_response = response.text.strip()
        print(full_response)
        
        return full_response

    
    async def speak_response(self, text):
        """TTS with pyttsx3 (non-blocking)"""
        import pyttsx3
        
        def speak_async(text):
            engine = pyttsx3.init()
            engine.setProperty('rate', 180)
            engine.setProperty('volume', 0.9)
            engine.say(text)
            engine.runAndWait()
        
        # Run TTS in thread to avoid blocking
        thread = threading.Thread(target=speak_async, args=(text,))
        thread.start()
        
    async def run_conversation_loop(self):
        """Main conversation loop"""
        print("🚀 DCB Bank Voice Assistant Ready!")
        
        while True:
            try:
                # 1. Listen for audio
                audio = await self.stream_audio_input()
                if audio is None:
                    continue
                
                # 2. Transcribe
                query = await self.transcribe_streaming(audio)
                print(f"\n👤 You: {query}")
                
                # Exit conditions
                if any(word in query.lower() for word in ["exit", "quit", "goodbye", "bye"]):
                    await self.speak_response("Thank you for contacting DCB Bank. Have a great day!")
                    break
                
                # 3. Generate response (RAG + LLM)
                response = await self.generate_response_streaming(query)
                print(f"🤖 Bot: {response}\n")
                
                # 4. Speak response
                await self.speak_response(response)
                
                # 5. Update conversation context
                self.session_context.append({"user": query, "assistant": response})
                
            except KeyboardInterrupt:
                print("\n👋 Session ended")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                await self.speak_response("I apologize, I encountered an error. Please try again.")

# Run the bot
if __name__ == "__main__":
    bot = StreamingVoiceBot()
    asyncio.run(bot.run_conversation_loop())
