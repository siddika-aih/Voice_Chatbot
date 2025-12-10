"""Agentic Voice Bot with tool execution"""
import asyncio
from typing import Optional
from google import genai
from google.genai import types

from voice_bot.audio_handler import AudioHandler
from agents.tools_executor import tool_executor
from agents.agent_prompts import AGENTIC_SYSTEM_PROMPT, TOOL_CONFIRMATION_PROMPT
from utils.config import config

class AgenticVoiceBot:
    """DCB Bank Agentic Voice Assistant with tool execution"""
    
    def __init__(self):
        self.audio_handler = AudioHandler()
        self.session: Optional[genai.LiveSession] = None
        self.audio_queue = asyncio.Queue(maxsize=50)
        self.session_context = []
        self.current_key_index = 0
        self.is_listening = True
        self.is_bot_speaking = False
        self.current_user_input = ""
        self.pending_tool_calls = []
        
    def _get_client(self):
        """Get Gemini client"""
        return genai.Client(
            http_options={"api_version": "v1alpha"},
            api_key=config.GEMINI_KEYS[
                self.current_key_index % len(config.GEMINI_KEYS)
            ]
        )
    
    # async def send_audio_to_gemini(self):
    #     """Capture and stream microphone audio"""
    #     await self.audio_handler.start_input_stream()
    #     last_keepalive = asyncio.get_event_loop().time()
        
    #     while self.is_listening:
    #         try:
    #             audio_data = await self.audio_handler.read_audio_chunk()
    #             await self.session.send(
    #                 input={"data": audio_data, "mime_type": "audio/pcm"}
    #             )
                
    #             # Keepalive
    #             current_time = asyncio.get_event_loop().time()
    #             if current_time - last_keepalive > 5.0:
    #                 await self.session.send(
    #                     input={"data": b"", "mime_type": "audio/pcm"}
    #                 )
    #                 last_keepalive = current_time
                    
    #         except Exception as e:
    #             if self.is_listening:
    #                 print(f"❌ Audio send error: {e}")
    #             break
    
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
        """Receive and handle responses with tool execution"""
        
        while self.is_listening:
            try:
                async for response in self.session.receive():
                    
                    # Handle interruption
                    if hasattr(response, 'server_content') and response.server_content:
                        if hasattr(response.server_content, 'interrupted') and response.server_content.interrupted:
                            print("\n🛑 [Interrupted]")
                            while not self.audio_queue.empty():
                                try:
                                    self.audio_queue.get_nowait()
                                except:
                                    break
                            self.is_bot_speaking = False
                            continue
                    
                    # Handle user transcription
                    if hasattr(response, 'input_transcription'):
                        if response.input_transcription and response.input_transcription.text:
                            user_text = response.input_transcription.text
                            self.current_user_input = user_text
                            print(f"\n👤 User: {user_text}")
                            self.is_bot_speaking = True
                    
                    # ✅ Handle FUNCTION CALLS (tool execution)
                    if hasattr(response, 'server_content') and response.server_content:
                        if hasattr(response.server_content, 'model_turn') and response.server_content.model_turn:
                            for part in response.server_content.model_turn.parts:
                                
                                # ✅ FUNCTION CALL DETECTED
                                if hasattr(part, 'function_call') and part.function_call:
                                    function_call = part.function_call
                                    print(f"\n🤖 Agent wants to use tool: {function_call.name}")
                                    
                                    # Execute tool
                                    tool_result = await tool_executor.execute_tool(function_call)
                                    
                                    # Send result back to Gemini
                                    function_response = types.FunctionResponse(
                                        name=function_call.name,
                                        response=tool_result
                                    )
                                    
                                    await self.session.send(
                                        input=types.Content(
                                            parts=[types.Part(function_response=function_response)]
                                        ),
                                        end_of_turn=True
                                    )
                                    
                                    print(f"✅ Tool result sent back to model")
                    
                    # Handle text output
                    if hasattr(response, 'output_transcription'):
                        if response.output_transcription and response.output_transcription.text:
                            bot_text = response.output_transcription.text
                            print(bot_text, end="", flush=True)
                            
                            if self.current_user_input:
                                self.session_context.append({
                                    "user": self.current_user_input,
                                    "bot": bot_text
                                })
                                self.current_user_input = ""
                    
                    # Handle audio output
                    if hasattr(response, 'server_content') and response.server_content:
                        if hasattr(response.server_content, 'model_turn') and response.server_content.model_turn:
                            for part in response.server_content.model_turn.parts:
                                if hasattr(part, 'inline_data') and part.inline_data:
                                    if part.inline_data.mime_type.startswith('audio/'):
                                        try:
                                            await asyncio.wait_for(
                                                self.audio_queue.put(part.inline_data.data),
                                                timeout=0.1
                                            )
                                        except asyncio.TimeoutError:
                                            pass
                        
                        # Turn complete
                        if hasattr(response.server_content, 'turn_complete') and response.server_content.turn_complete:
                            print("\n")
                            self.is_bot_speaking = False
                    
            except Exception as e:
                if self.is_listening:
                    print(f"\n❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Try key rotation
                    self.current_key_index += 1
                    if self.current_key_index >= len(config.GEMINI_KEYS):
                        print("❌ All API keys exhausted")
                        break
    
    async def play_audio_responses(self):
        """Play audio smoothly"""
        await self.audio_handler.start_output_stream()
        
        while self.is_listening:
            try:
                audio_bytes = await asyncio.wait_for(
                    self.audio_queue.get(),
                    timeout=0.5
                )
                await self.audio_handler.write_audio_chunk(audio_bytes)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if self.is_listening:
                    print(f"❌ Audio error: {e}")
    
    async def run(self):
                """Main agentic bot loop"""
                print("\n" + "="*70)
                print("🤖 DCB Bank AGENTIC Voice Assistant")
                print("="*70)
                print("✨ I can:")
                print("   📧 Send emails")
                print("   🔍 Search Google for current info")
                print("   🐍 Execute Python code")
                print("   🏦 Query DCB Bank knowledge")
                print("\n💬 Speak naturally - I'll execute tasks for you!")
                print("🛑 Press Ctrl+C to exit\n")
                
                # ✅ Configure with TOOLS (Fixed syntax)
                live_config = types.LiveConnectConfig(
                    response_modalities=["AUDIO"],
                    input_audio_transcription=types.AudioTranscriptionConfig(),
                    output_audio_transcription=types.AudioTranscriptionConfig(),
                    
                    # ✅ CORRECTED TOOLS CONFIGURATION
                    tools=[
                        # Custom function tools
                        types.Tool(
                            function_declarations=tool_executor.get_tool_declarations()
                        ),
                        # Native Google Search
                        types.Tool(
                            google_search=types.GoogleSearch()
                        ),
                        # Native Code Execution
                        types.Tool(
                            code_execution={}  # ✅ Fixed: Empty dict instead of CodeExecution()
                        )
                    ],
                    
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=config.VOICE_NAME
                            )
                        )
                    ),
                    
                    realtime_input_config=types.RealtimeInputConfig(
                        automatic_activity_detection=types.AutomaticActivityDetection(
                            disabled=False
                        ),
                        activity_handling=types.ActivityHandling.NO_INTERRUPTION
                    ),
                    
                    # AGENTIC SYSTEM INSTRUCTION
                    system_instruction=types.Content(
                        parts=[types.Part(text=AGENTIC_SYSTEM_PROMPT)]
                    )
                )
                
                try:
                    client = self._get_client()
                    
                    async with client.aio.live.connect(
                        model=config.GEMINI_MODEL,
                        config=live_config
                    ) as session:
                        
                        self.session = session
                        print("✅ Agent connected with tools enabled")
                        print("🎙️  Listening for commands...\n")
                        
                        # Start all tasks
                        tasks = [
                            asyncio.create_task(self.send_audio_to_gemini()),
                            asyncio.create_task(self.receive_from_gemini()),
                            asyncio.create_task(self.play_audio_responses())
                        ]
                        
                        done, pending = await asyncio.wait(
                            tasks,
                            return_when=asyncio.FIRST_EXCEPTION
                        )
                        
                        for task in pending:
                            task.cancel()
                        
                        for task in done:
                            if task.exception():
                                raise task.exception()
                        
                except KeyboardInterrupt:
                    print("\n\n👋 Agent shutting down...")
                except Exception as e:
                    print(f"\n❌ Fatal error: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    self.is_listening = False
                    self.audio_handler.cleanup()
                    print("\n🛑 Cleanup complete")

