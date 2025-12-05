# # test_websocket.py - Python WebSocket tester
# import asyncio
# import websockets
# import base64
# import numpy as np
# import sounddevice as sd
# import json
# import threading
# from scipy.io.wavfile import write
# import tempfile
# import os

# async def test_voice_websocket():
#     uri = "ws://localhost:8000/ws/voice/test-session"
#     async with websockets.connect(uri) as websocket:
#         print("✅ WebSocket connected!")
        
#         # Start recording thread
#         audio_buffer = []
#         def record_audio():
#             def callback(indata, frames, time, status):
#                 audio_buffer.extend(indata.flatten())
            
#             with sd.InputStream(callback=callback, channels=1, samplerate=16000):
#                 print("🎤 Bolna shuru karo... (Ctrl+C to stop)")
#                 sd.sleep(4000)  # 4 seconds
        
#         # Record and send
#         record_thread = threading.Thread(target=record_audio)
#         record_thread.start()
#         record_thread.join()
        
#         if audio_buffer:
#             # Convert to bytes and send
#             audio_np = np.array(audio_buffer, dtype=np.float32)
#             audio_bytes = (audio_np * 32767).astype(np.int16).tobytes()
            
#             await websocket.send(json.dumps({
#                 "type": "audio",
#                 "audio": base64.b64encode(audio_bytes).decode()
#             }))
#             print("📤 Audio sent!")
            
#             # Receive responses
#             async for message in websocket:
#                 data = json.loads(message)
#                 if data["type"] == "response":
#                     print(f"🤖 Bot: {data['text']}")

# if __name__ == "__main__":
#     asyncio.run(test_voice_websocket())


# # testr.py - Improved WebSocket client
# import asyncio
# import websockets
# import base64
# import numpy as np
# import sounddevice as sd
# import json

# async def test_voice_websocket():
#     uri = "ws://localhost:8000/ws/voice/test-session"
    
#     try:
#         async with websockets.connect(uri, ping_timeout=60, close_timeout=10) as websocket:
#             print("✅ WebSocket connected!")
            
#             # Record audio
#             print("🎤 Recording for 4 seconds...")
#             audio = sd.rec(int(4 * 16000), samplerate=16000, channels=1, dtype='float32')
#             sd.wait()
#             print("✔ Recording complete")
            
#             # Normalize and convert
#             audio_normalized = np.clip(audio.flatten(), -1.0, 1.0)
#             audio_int16 = (audio_normalized * 32767).astype(np.int16)
#             audio_bytes = audio_int16.tobytes()
            
#             # Send to server
#             print("📤 Sending audio...")
#             await websocket.send(json.dumps({
#                 "type": "audio",
#                 "audio": base64.b64encode(audio_bytes).decode()
#             }))
#             print("✅ Audio sent!")
            
#             # Wait for responses
#             transcript_received = False
#             response_received = False
            
#             try:
#                 while not (transcript_received and response_received):
#                     message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
#                     data = json.loads(message)
                    
#                     if data["type"] == "transcript":
#                         print(f"\n👤 You said: {data['text']}")
#                         transcript_received = True
                        
#                     elif data["type"] == "response":
#                         print(f"\n🤖 Bot replied: {data['text']}\n")
#                         response_received = True
                        
#                     elif data["type"] == "error":
#                         print(f"\n❌ Error: {data['text']}")
#                         break
                
#                 print("✅ Test completed successfully!")
                        
#             except asyncio.TimeoutError:
#                 print("\n⏱️ Timeout waiting for response")
#                 print("Check server logs for errors")
                
#     except websockets.exceptions.ConnectionClosedError as e:
#         print(f"❌ Connection closed unexpectedly: {e}")
#     except ConnectionRefusedError:
#         print("❌ Cannot connect to server. Is it running on port 8000?")
#     except Exception as e:
#         print(f"❌ Error: {type(e).__name__}: {e}")

# if __name__ == "__main__":
#     print("🚀 Starting voice bot test...\n")
#     asyncio.run(test_voice_websocket())


# # testr.py - Fast WebSocket client with latency tracking
# import asyncio
# import websockets
# import base64
# import numpy as np
# import sounddevice as sd
# import json
# import time


# async def test_voice_websocket():
#     uri = "ws://localhost:8000/ws/voice/test-session"
    
#     try:
#         async with websockets.connect(uri, ping_timeout=60) as websocket:
#             print("✅ WebSocket connected!\n")
            
#             while True:
#                 # Ask to continue
#                 choice = input("🎤 Press ENTER to speak (or 'q' to quit): ")
#                 if choice.lower() == 'q':
#                     print("👋 Goodbye!")
#                     break
                
#                 # Record audio
#                 print("🎤 Recording for 5 seconds...")
#                 start_time = time.time()
                
#                 audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype='float32')
#                 sd.wait()
                
#                 record_time = time.time() - start_time
#                 print(f"✔ Recording done ({record_time:.2f}s)")
                
#                 # Process and send
#                 audio_normalized = np.clip(audio.flatten(), -1.0, 1.0)
#                 audio_int16 = (audio_normalized * 32767).astype(np.int16)
#                 audio_bytes = audio_int16.tobytes()
                
#                 send_start = time.time()
#                 await websocket.send(json.dumps({
#                     "type": "audio",
#                     "audio": base64.b64encode(audio_bytes).decode()
#                 }))
#                 print(f"📤 Audio sent ({time.time() - send_start:.2f}s)\n")
                
#                 # Receive responses
#                 transcript_received = False
#                 response_received = False
                
#                 try:
#                     while not (transcript_received and response_received):
#                         message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
#                         data = json.loads(message)
                        
#                         if data["type"] == "transcript":
#                             latency = data.get("latency", 0)
#                             print(f"👤 You [{latency}s]: {data['text']}")
#                             transcript_received = True
                            
#                         elif data["type"] == "response":
#                             latency = data.get("latency", 0)
#                             print(f"🤖 Bot [{latency}s]: {data['text']}\n")
#                             print("🔊 Speaking response...\n")
#                             response_received = True
                            
#                         elif data["type"] == "error":
#                             print(f"❌ Error: {data['text']}")
#                             break
                    
#                     total_time = time.time() - start_time
#                     print(f"⏱️ Total time: {total_time:.2f}s")
#                     print("="*80 + "\n")
                            
#                 except asyncio.TimeoutError:
#                     print("⏱️ Response timeout")
                
#     except KeyboardInterrupt:
#         print("\n\n👋 Stopped by user")
#     except Exception as e:
#         print(f"❌ Error: {e}")


# if __name__ == "__main__":
#     print("🚀 DCB Bank Voice Assistant - WebSocket Mode")
#     print("="*80 + "\n")
#     asyncio.run(test_voice_websocket())


# testr.py - Client that can interrupt bot
import asyncio
import websockets
import base64
import numpy as np
import sounddevice as sd
import json
import time
import sys

async def test_voice_websocket():
    uri = "ws://localhost:8000/ws/voice/test-session"
    
    try:
        async with websockets.connect(uri, ping_timeout=60) as websocket:
            print("✅ Connected!\n")
            print("💡 TIP: Press ENTER while bot is speaking to interrupt!\n")
            
            while True:
                choice = input("🎤 Press ENTER to speak (or 'q' to quit): ")
                if choice.lower() == 'q':
                    print("👋 Goodbye!")
                    break
                
                # Send interrupt signal first (stops previous TTS)
                await websocket.send(json.dumps({"type": "interrupt"}))
                
                # Record audio
                print("🎤 Recording for 5 seconds...")
                start_time = time.time()
                
                audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype='float32')
                sd.wait()
                
                print(f"✔ Recording done ({time.time()-start_time:.1f}s)")
                
                # Send audio
                audio_normalized = np.clip(audio.flatten(), -1.0, 1.0)
                audio_int16 = (audio_normalized * 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()
                
                await websocket.send(json.dumps({
                    "type": "audio",
                    "audio": base64.b64encode(audio_bytes).decode()
                }))
                print("📤 Sent!\n")
                
                # Receive responses
                response_complete = False
                
                try:
                    while not response_complete:
                        message = await asyncio.wait_for(websocket.recv(), timeout=60.0)
                        data = json.loads(message)
                        
                        if data["type"] == "transcript":
                            latency = data.get("latency", 0)
                            print(f"👤 You [{latency}s]: {data['text']}\n")
                            
                        elif data["type"] == "response":
                            latency = data.get("latency", 0)
                            print(f"🤖 Bot [{latency}s]: {data['text']}")
                            print("\n🔊 Bot is speaking... (Press ENTER to interrupt)")
                            print("="*80 + "\n")
                            response_complete = True
                            
                        elif data["type"] == "error":
                            print(f"❌ Error: {data['text']}\n")
                            response_complete = True
                            
                except asyncio.TimeoutError:
                    print("⏱️ Timeout\n")
                
    except KeyboardInterrupt:
        print("\n👋 Stopped")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("🚀 DCB Voice Bot - With Interruption Support")
    print("="*80)
    print("💡 Press ENTER anytime to interrupt bot and ask new question!")
    print("="*80 + "\n")
    asyncio.run(test_voice_websocket())
