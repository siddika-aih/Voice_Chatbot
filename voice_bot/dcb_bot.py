"""Main DCB Voice Bot with Gemini Live API"""
import asyncio
from typing import Optional
from google import genai
from google.genai import types

from voice_bot.audio_handler import AudioHandler
from retrieval.hybrid_search import hybrid_search
from utils.config import config

class DCBVoiceBot:
    """DCB Bank RAG-powered voice assistant"""
    
    def __init__(self):
        self.audio_handler = AudioHandler()
        self.session: Optional[genai.LiveSession] = None
        self.audio_queue = asyncio.Queue()
        self.session_context = []
        self.current_key_index = 0
        self.is_listening = True
        
    def _get_client(self):
        """Get Gemini client with key rotation"""
        return genai.Client(
            http_options={"api_version": "v1alpha"},
            api_key=config.GEMINI_KEYS[
                self.current_key_index % len(config.GEMINI_KEYS)
            ]
        )
    
    async def _rag_middleware(self, user_text: str) -> str:
        """Fetch RAG context and format prompt"""
        print(f"🔍 Retrieving context for: {user_text[:50]}...")
        
        # Get relevant context
        context = await hybrid_search.hybrid_retrieve(
            user_text,
            top_k=config.TOP_K_RESULTS
        )
        
        # Build conversation history
        history = ""
        if self.session_context:
            recent = self.session_context[-2:]
            history = "\n".join([
                f"User: {t['user']}\nAssistant: {t['bot']}"
                for t in recent
            ])
        
        # Format prompt
        prompt = f"""You are a helpful DCB Bank voice assistant. Answer questions clearly and concisely in 2-3 sentences maximum.

Knowledge Base Context:
{context}

Recent Conversation:
{history}

User Question: {user_text}

Your Response:"""
        
        return prompt
    
    async def send_audio_to_gemini(self):
        """Capture microphone and stream to Gemini"""
        await self.audio_handler.start_input_stream()
        
        while self.is_listening:
            try:
                audio_data = await self.audio_handler.read_audio_chunk()
                await self.session.send(
                    input={"data": audio_data, "mime_type": "audio/pcm"}
                )
            except Exception as e:
                print(f"❌ Audio send error: {e}")
                break
    
    async def receive_from_gemini(self):
        """Receive responses from Gemini with RAG injection"""
        current_user_query = ""
        current_bot_response = ""
        
        while self.is_listening:
            try:
                turn = self.session.receive()
                
                async for response in turn:
                    # Handle text (transcription or bot response)
                    if text := response.text:
                        # Check if it's user speech (Gemini transcribes user)
                        if not current_bot_response:
                            current_user_query = text
                            print(f"\n👤 User: {text}")
                            
                            # Get RAG context and send enriched prompt
                            enriched_prompt = await self._rag_middleware(text)
                            await self.session.send(
                                input=enriched_prompt,
                                end_of_turn=True
                            )
                        else:
                            # Bot is responding
                            current_bot_response += text
                            print(text, end="", flush=True)
                    
                    # Handle audio response from bot
                    if data := response.data:
                        self.audio_queue.put_nowait(data)
                
                # Turn complete - save to context
                if current_user_query and current_bot_response:
                    self.session_context.append({
                        "user": current_user_query,
                        "bot": current_bot_response
                    })
                    print(f"\n🤖 Bot: {current_bot_response}\n")
                    current_user_query = ""
                    current_bot_response = ""
                
                # Clear audio queue on interruption
                while not self.audio_queue.empty():
                    self.audio_queue.get_nowait()
                    
            except Exception as e:
                print(f"❌ Receive error: {e}")
                # Try key rotation
                self.current_key_index += 1
                if self.current_key_index >= len(config.GEMINI_KEYS):
                    print("❌ All API keys exhausted")
                    break
    
    async def play_audio_responses(self):
        """Play audio responses from Gemini"""
        await self.audio_handler.start_output_stream()
        
        while self.is_listening:
            try:
                audio_bytes = await self.audio_queue.get()
                await self.audio_handler.write_audio_chunk(audio_bytes)
            except Exception as e:
                print(f"❌ Audio play error: {e}")
    
    async def run(self):
        """Main bot loop"""
        print("\n" + "="*50)
        print("🏦 DCB Bank Voice Assistant")
        print("="*50)
        print("🎤 Speak naturally - I'll respond with voice")
        print("💬 Press Ctrl+C to exit\n")
        
        # Configure Gemini Live API
        live_config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=config.VOICE_NAME
                    )
                )
            ),
            system_instruction=types.Content(
                parts=[types.Part(
                    text="""You are a professional DCB Bank customer service assistant.
                    - Be helpful, concise, and friendly
                    - Answer in 2-3 sentences for voice clarity
                    - Use the provided knowledge base context to answer accurately
                    - If uncertain, acknowledge it politely"""
                )]
            )
        )
        
        try:
            client = self._get_client()
            
            async with (
                client.aio.live.connect(
                    model=config.GEMINI_MODEL,
                    config=live_config
                ) as session,
                asyncio.TaskGroup() as tg
            ):
                self.session = session
                
                # Start all tasks
                tg.create_task(self.send_audio_to_gemini())
                tg.create_task(self.receive_from_gemini())
                tg.create_task(self.play_audio_responses())
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
        finally:
            self.is_listening = False
            self.audio_handler.cleanup()
