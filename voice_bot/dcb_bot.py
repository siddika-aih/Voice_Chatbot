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
        prompt = f"""You are DCB Bank's voice assistant. IMPORTANT RULES:

1. ONLY answer questions about DCB Bank products, services, accounts, loans, cards
2. If the question is NOT about DCB Bank, politely say: "I can only help with DCB Bank related queries. Please ask about our banking services."
3. Keep responses under 3 sentences for voice clarity
4. Be professional and helpful

KNOWLEDGE BASE (Use this information ONLY):
{context}

RECENT CONVERSATION:
{history}

User Question: {user_text}

Your Response:"""
        
        return prompt
    
    # async def send_audio_to_gemini(self):
    #     """Capture microphone and stream to Gemini"""
    #     await self.audio_handler.start_input_stream()
        
    #     while self.is_listening:
    #         try:
    #             audio_data = await self.audio_handler.read_audio_chunk()
    #             await self.session.send(
    #                 input={"data": audio_data, "mime_type": "audio/pcm"}
    #             )
    #             # ✅ Send keepalive every 5 seconds (prevents mid-sentence stops)
    #             current_time = asyncio.get_event_loop().time()
    #             if current_time - last_keepalive > 5.0:
    #                 await self.session.send(
    #                     input={"data": b"", "mime_type": "audio/pcm"}
    #                 )
    #                 last_keepalive = current_time
    #         except Exception as e:
    #             if self.is_listening:  # Only log if not shutting down
    #                 print(f"❌ Audio send error: {e}")
    #             break
    
    # async def receive_from_gemini(self):
    #     """Receive responses from Gemini with RAG injection"""
    #     current_user_query = ""
    #     current_bot_response = ""
        
    #     while self.is_listening:
    #         try:
    #             turn = self.session.receive()
                
    #             async for response in turn:
    #                 # Handle text (transcription or bot response)
    #                 if text := response.text:
    #                     # Check if it's user speech (Gemini transcribes user)
    #                     if not current_bot_response:
    #                         current_user_query = text
    #                         print(f"\n👤 User: {text}")
                            
    #                         # Get RAG context and send enriched prompt
    #                         enriched_prompt = await self._rag_middleware(text)
    #                         await self.session.send(
    #                             input=enriched_prompt,
    #                             end_of_turn=True
    #                         )
    #                     else:
    #                         # Bot is responding
    #                         current_bot_response += text
    #                         print(text, end="", flush=True)
                    
    #                 # Handle audio response from bot
    #                 if data := response.data:
    #                     self.audio_queue.put_nowait(data)
                
    #             # Turn complete - save to context
    #             if current_user_query and current_bot_response:
    #                 self.session_context.append({
    #                     "user": current_user_query,
    #                     "bot": current_bot_response
    #                 })
    #                 print(f"\n🤖 Bot: {current_bot_response}\n")
    #                 current_user_query = ""
    #                 current_bot_response = ""
                
    #             # Clear audio queue on interruption
    #             while not self.audio_queue.empty():
    #                 self.audio_queue.get_nowait()
                    
    #         except Exception as e:
    #             print(f"❌ Receive error: {e}")
    #             # Try key rotation
    #             self.current_key_index += 1
    #             if self.current_key_index >= len(config.GEMINI_KEYS):
    #                 print("❌ All API keys exhausted")
    #                 break
    
    async def send_audio_to_gemini(self):
        """Capture microphone and stream to Gemini"""
        await self.audio_handler.start_input_stream()
        
        # ✅ Keep-alive mechanism to prevent socket timeout
        last_keepalive = asyncio.get_event_loop().time()
        
        while self.is_listening:
            try:
                # Read audio chunk
                audio_data = await self.audio_handler.read_audio_chunk()
                
                # Send to Gemini
                await self.session.send(
                    input={"data": audio_data, "mime_type": "audio/pcm"}
                )
                
                # ✅ Send keepalive every 5 seconds (prevents mid-sentence stops)
                current_time = asyncio.get_event_loop().time()
                if current_time - last_keepalive > 5.0:
                    await self.session.send(
                        input={"data": b"", "mime_type": "audio/pcm"}
                    )
                    last_keepalive = current_time
                    
            except Exception as e:
                if self.is_listening:  # Only log if not shutting down
                    print(f"❌ Audio send error: {e}")
                break
    
    async def receive_from_gemini(self):
        """Receive responses from Gemini with proper handling"""
        
        while self.is_listening:
            try:
                # ✅ Proper response iteration
                async for response in self.session.receive():
                    
                    # ✅ Handle interruption (user spoke while bot was speaking)
                    if hasattr(response, 'server_content') and response.server_content:
                        if hasattr(response.server_content, 'interrupted') and response.server_content.interrupted:
                            print("\n🛑 [Interrupted] Clearing audio queue...")
                            
                            # Clear all pending audio immediately
                            while not self.audio_queue.empty():
                                try:
                                    self.audio_queue.get_nowait()
                                except:
                                    break
                            
                            self.is_bot_speaking = False
                            continue
                    
                    # ✅ Handle user transcription (input_transcription)
                    if hasattr(response, 'input_transcription'):
                        if response.input_transcription and response.input_transcription.text:
                            user_text = response.input_transcription.text
                            self.current_user_input = user_text
                            print(f"\n👤 User: {user_text}")
                            
                            # Get RAG context and send enriched prompt
                            enriched_prompt = await self._rag_middleware(user_text)
                            await self.session.send(
                                input=enriched_prompt,
                                end_of_turn=True
                            )
                            self.is_bot_speaking = True
                    
                    # ✅ Handle bot text response (output_transcription)
                    if hasattr(response, 'output_transcription'):
                        if response.output_transcription and response.output_transcription.text:
                            bot_text = response.output_transcription.text
                            print(bot_text, end="", flush=True)
                            
                            # Save to context
                            if self.current_user_input:
                                self.session_context.append({
                                    "user": self.current_user_input,
                                    "bot": bot_text
                                })
                                self.current_user_input = ""
                    
                    # ✅ Handle audio data (server_content with inline_data)
                    if hasattr(response, 'server_content') and response.server_content:
                        if hasattr(response.server_content, 'model_turn') and response.server_content.model_turn:
                            for part in response.server_content.model_turn.parts:
                                if hasattr(part, 'inline_data') and part.inline_data:
                                    if part.inline_data.mime_type.startswith('audio/'):
                                        # ✅ Put audio in queue for smooth playback
                                        try:
                                            await asyncio.wait_for(
                                                self.audio_queue.put(part.inline_data.data),
                                                timeout=0.1
                                            )
                                        except asyncio.TimeoutError:
                                            # Queue full, skip frame to prevent lag
                                            pass
                        
                        # ✅ Check turn complete
                        if hasattr(response.server_content, 'turn_complete') and response.server_content.turn_complete:
                            print("\n")  # New line after bot finishes
                            self.is_bot_speaking = False
                    
            except Exception as e:
                if self.is_listening:
                    print(f"\n❌ Receive error: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Try key rotation
                    self.current_key_index += 1
                    if self.current_key_index >= len(config.GEMINI_KEYS):
                        print("❌ All API keys exhausted")
                        break
    
    async def play_audio_responses(self):
        """Play audio responses smoothly without stuttering"""
        await self.audio_handler.start_output_stream()
        
        while self.is_listening:
            try:
                # ✅ Get audio chunk with timeout to prevent blocking
                audio_bytes = await asyncio.wait_for(
                    self.audio_queue.get(),
                    timeout=0.5
                )
                
                # ✅ Play immediately for low latency
                await self.audio_handler.write_audio_chunk(audio_bytes)
                
            except asyncio.TimeoutError:
                # No audio in queue, continue waiting
                continue
            except Exception as e:
                if self.is_listening:
                    print(f"❌ Audio play error: {e}")
    
    # async def run(self):
    #     """Main bot loop"""
    #     print("\n" + "="*60)
    #     print("🏦 DCB Bank Voice Assistant - RAG Enabled")
    #     print("="*60)
    #     print("🎤 Speak naturally about DCB Bank services")
    #     print("💬 Press Ctrl+C to exit")
    #     print("⚠️  I only answer DCB Bank related questions\n")
        
    #     # ✅ Configure Gemini Live API with proper settings
    #     live_config = types.LiveConnectConfig(
    #         response_modalities=["AUDIO"],
            
    #         # ✅ Enable transcriptions for both input and output
    #         input_audio_transcription=types.AudioTranscriptionConfig(),
    #         output_audio_transcription=types.AudioTranscriptionConfig(),
            
    #         # ✅ Speech configuration
    #         speech_config=types.SpeechConfig(
    #             voice_config=types.VoiceConfig(
    #                 prebuilt_voice_config=types.PrebuiltVoiceConfig(
    #                     voice_name=config.VOICE_NAME
    #                 )
    #             )
    #         ),
            
    #         # ✅ Realtime input config with VAD
    #         realtime_input_config=types.RealtimeInputConfig(
    #             automatic_activity_detection=types.AutomaticActivityDetection(
    #                 disabled=False  # Enable VAD for natural conversation
    #             ),
    #             activity_handling=types.ActivityHandling.NO_INTERRUPTION
    #         ),
            
    #         # ✅ System instruction for strict behavior
    #         system_instruction=types.Content(
    #             parts=[types.Part(
    #                 text="""You are DCB Bank's professional voice assistant.
    #                 - ONLY answer DCB Bank related queries
    #                 - If question is not about DCB Bank, politely decline
    #                 - Be concise (2-3 sentences max)
    #                 - Be helpful and professional
    #                 - Use provided context ONLY"""
    #             )]
    #         )
    #     )
        
    #     try:
    #         client = self._get_client()
            
    #         async with client.aio.live.connect(
    #             model=config.GEMINI_MODEL,
    #             config=live_config
    #         ) as session:
                
    #             self.session = session
    #             print("✅ Connected to Gemini Live API")
    #             print("🎙️  Listening...\n")
                
    #             # ✅ Start all tasks concurrently
    #             async with asyncio.TaskGroup() as tg:
    #                 tg.create_task(self.send_audio_to_gemini())
    #                 tg.create_task(self.receive_from_gemini())
    #                 tg.create_task(self.play_audio_responses())
                
    #     except KeyboardInterrupt:
    #         print("\n\n👋 Goodbye!")
    #     except Exception as e:
    #         # ✅ Handle ExceptionGroup from TaskGroup
    #         print(f"\n❌ Fatal error: {e}")
    #         import traceback
    #         traceback.print_exc()
    #     finally:
    #         self.is_listening = False
    #         self.audio_handler.cleanup()
    #         print("\n🛑 Cleanup complete")
    
    
    
    # async def play_audio_responses(self):
    #     """Play audio responses from Gemini"""
    #     await self.audio_handler.start_output_stream()
        
    #     while self.is_listening:
    #         try:
    #             audio_bytes = await self.audio_queue.get()
    #             await self.audio_handler.write_audio_chunk(audio_bytes)
    #         except Exception as e:
    #             print(f"❌ Audio play error: {e}")
    
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
