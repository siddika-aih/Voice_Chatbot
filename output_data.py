# from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# import asyncio
# import numpy as np
# from main import StreamingVoiceBot
# from session_manager import session_manager
# import base64

# app = FastAPI(title="DCB Voice Bot API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # DON'T initialize here - remove this line:
# # tts_engine = pyttsx3.init()  ❌

# # NEW: Initialize inside function
# def speak_now(text):
#     """Speak response using pyttsx3 - lazy initialization"""
#     try:
#         import pyttsx3
#         engine = pyttsx3.init()  # Initialize HERE ✅
#         engine.setProperty('rate', 180)
#         engine.setProperty('volume', 1.0)
        
#         print("🔊 Speaking response...")
#         engine.say(text)
#         engine.runAndWait()
#         print("✔ Speech complete")
        
#         engine.stop()  # Cleanup
#         del engine
        
#     except Exception as e:
#         print(f"TTS Error: {e}")

# @app.post("/api/voice-query")
# async def voice_query_api(audio: UploadFile = File(...)):
#     """REST API with voice response"""
#     try:
#         # Read audio
#         audio_bytes = await audio.read()
#         audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        
#         # Process
#         bot = StreamingVoiceBot()
#         query = await bot.transcribe_streaming(audio_array)
#         response = await bot.generate_response_streaming(query)
        
#         # SPEAK IT!
#         speak_now(response)
        
#         return {
#             "transcript": query,
#             "response": response,
#             "status": "success"
#         }
        
#     except Exception as e:
#         return {
#             "error": str(e),
#             "status": "error"
#         }

# @app.websocket("/ws/voice/{session_id}")
# async def voice_websocket(websocket: WebSocket, session_id: str):
#     """WebSocket with voice output"""
#     await websocket.accept()
#     print(f"✅ WebSocket connected: {session_id}")
    
#     session = session_manager.get_session(session_id)
#     if not session:
#         session_id = await session_manager.create_session()
#         session = session_manager.get_session(session_id)
    
#     bot = StreamingVoiceBot()
#     bot.session_context = session["context"]
    
#     try:
#         while True:
#             data = await websocket.receive_json()
            
#             if data["type"] == "audio":
#                 try:
#                     audio_bytes = base64.b64decode(data["audio"])
#                     audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
                    
#                     query = await bot.transcribe_streaming(audio_array)
#                     await websocket.send_json({"type": "transcript", "text": query})
                    
#                     response = await bot.generate_response_streaming(query)
#                     await websocket.send_json({"type": "response", "text": response})
                    
#                     # SPEAK IT!
#                     speak_now(response)
                    
#                     session["context"].append({"user": query, "assistant": response})
#                     session["query_count"] += 1
                    
#                 except Exception as e:
#                     print(f"❌ Error: {e}")
                
#     except WebSocketDisconnect:
#         print(f"🔌 Disconnected: {session_id}")


# @app.post("/api/ingest")
# async def trigger_ingestion():
#     """Trigger data ingestion"""
#     from data_store import ingest_dcb_website
#     chunks = await ingest_dcb_website()
#     return {"status": "success", "chunks_ingested": len(chunks)}

# @app.get("/health")
# async def health_check():
#     """Health check"""
#     return {
#         "status": "healthy",
#         "active_sessions": len(session_manager.sessions),
#         "max_concurrent": session_manager.max_concurrent
#     }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="localhost", port=8000, workers=4)
# output_data.py - Fixed with proper TTS
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import numpy as np
from main import StreamingVoiceBot, stop_speaking
from session_manager import session_manager
from tts_streaming import tts
import base64
import time

app = FastAPI(title="DCB Voice Bot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Start TTS on server start"""
    tts.start()
    print("✅ Server ready with TTS")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    tts.shutdown()


@app.websocket("/ws/voice/{session_id}")
async def voice_websocket(websocket: WebSocket, session_id: str):
    """WebSocket with interruption + TTS"""
    await websocket.accept()
    print(f"✅ Connected: {session_id}")
    
    # Get/create session
    session = session_manager.get_session(session_id)
    if not session:
        session_id = await session_manager.create_session()
        session = session_manager.get_session(session_id)
    
    bot = StreamingVoiceBot()
    bot.session_context = session["context"]
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "audio":
                start_time = time.time()
                
                # STOP previous TTS immediately
                print("🛑 Stopping previous TTS...")
                tts.stop()
                stop_speaking.set()
                await asyncio.sleep(0.1)
                stop_speaking.clear()
                
                try:
                    # Decode audio
                    audio_bytes = base64.b64decode(data["audio"])
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
                    
                    # Transcribe
                    query = await bot.transcribe_streaming(audio_array)
                    transcription_time = time.time() - start_time
                    
                    print(f"👤 [{transcription_time:.2f}s] {query}")
                    
                    # Send transcript
                    await websocket.send_json({
                        "type": "transcript",
                        "text": query,
                        "latency": round(transcription_time, 2)
                    })
                    
                    # Generate response
                    response_start = time.time()
                    
                    # Use non-streaming for reliability (streaming was causing issues)
                    full_response = await bot.generate_response_streaming(query, callback=None)
                    
                    response_time = time.time() - response_start
                    print(f"🤖 [{response_time:.2f}s] {full_response[:80]}...")
                    
                    # Send complete response
                    await websocket.send_json({
                        "type": "response",
                        "text": full_response,
                        "latency": round(response_time, 2)
                    })
                    
                    # SPEAK THE RESPONSE
                    print("🔊 Starting TTS...")
                    tts.speak_full(full_response)
                    
                    # Update session
                    session["context"].append({
                        "user": query,
                        "assistant": full_response
                    })
                    session["query_count"] += 1
                    
                    total_time = time.time() - start_time
                    print(f"✅ Total: {total_time:.2f}s\n")
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    await websocket.send_json({
                        "type": "error",
                        "text": str(e)
                    })
            
            elif data["type"] == "interrupt":
                print("🛑 Manual interrupt")
                tts.stop()
                stop_speaking.set()
    
    except WebSocketDisconnect:
        tts.stop()
        print(f"🔌 Disconnected: {session_id}")
    except Exception as e:
        print(f"❌ Error: {e}")


@app.post("/api/voice-query")
async def voice_query_api(audio: UploadFile = File(...)):
    """REST API"""
    try:
        audio_bytes = await audio.read()
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        
        bot = StreamingVoiceBot()
        query = await bot.transcribe_streaming(audio_array)
        response = await bot.generate_response_streaming(query, callback=None)
        
        # Speak
        tts.speak_full(response)
        
        return {
            "transcript": query,
            "response": response,
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_sessions": len(session_manager.sessions)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)



# from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# import asyncio
# import numpy as np
# from main import StreamingVoiceBot
# from session_manager import session_manager
# import base64
# import time
# from concurrent.futures import ThreadPoolExecutor

# app = FastAPI(title="DCB Voice Bot API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Thread pool for TTS (non-blocking)
# tts_executor = ThreadPoolExecutor(max_workers=4)


# def speak_sync(text):
#     """Sync TTS function"""
#     try:
#         import pyttsx3
#         engine = pyttsx3.init()
#         engine.setProperty('rate', 200)  # Faster speech
#         engine.setProperty('volume', 1.0)
#         engine.say(text)
#         engine.runAndWait()
#         engine.stop()
#         del engine
#     except Exception as e:
#         print(f"TTS Error: {e}")


# async def speak_async(text):
#     """Non-blocking TTS"""
#     loop = asyncio.get_event_loop()
#     await loop.run_in_executor(tts_executor, speak_sync, text)


# @app.websocket("/ws/voice/{session_id}")
# async def voice_websocket(websocket: WebSocket, session_id: str):
#     """Ultra-fast WebSocket with parallel processing"""
#     await websocket.accept()
#     print(f"✅ Connected: {session_id}")
    
#     # Get/create session
#     session = session_manager.get_session(session_id)
#     if not session:
#         session_id = await session_manager.create_session()
#         session = session_manager.get_session(session_id)
    
#     bot = StreamingVoiceBot()
#     bot.session_context = session["context"]
    
#     try:
#         while True:
#             data = await websocket.receive_json()
            
#             if data["type"] == "audio":
#                 start_time = time.time()
                
#                 try:
#                     # Decode audio
#                     audio_bytes = base64.b64decode(data["audio"])
#                     audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
                    
#                     # PARALLEL PROCESSING
#                     # 1. Start transcription
#                     transcription_task = asyncio.create_task(
#                         bot.transcribe_streaming(audio_array)
#                     )
                    
#                     # Wait for transcription
#                     query = await transcription_task
#                     transcription_time = time.time() - start_time
                    
#                     print(f"👤 [{transcription_time:.2f}s] Query: {query}")
                    
#                     # Send transcript immediately
#                     await websocket.send_json({
#                         "type": "transcript", 
#                         "text": query,
#                         "latency": round(transcription_time, 2)
#                     })
                    
#                     # 2. Generate response (fast RAG)
#                     response_start = time.time()
#                     response = await bot.generate_response_streaming(query)
#                     response_time = time.time() - response_start
                    
#                     print(f"🤖 [{response_time:.2f}s] Response: {response[:100]}...")
                    
#                     # Send response immediately
#                     await websocket.send_json({
#                         "type": "response", 
#                         "text": response,
#                         "latency": round(response_time, 2)
#                     })
                    
#                     # 3. TTS in parallel (don't wait)
#                     asyncio.create_task(speak_async(response))
                    
#                     # Update session
#                     session["context"].append({"user": query, "assistant": response})
#                     session["query_count"] += 1
                    
#                     total_time = time.time() - start_time
#                     print(f"✅ Total: {total_time:.2f}s\n")
                    
#                 except Exception as e:
#                     print(f"❌ Error: {e}")
#                     import traceback
#                     traceback.print_exc()
                    
#                     await websocket.send_json({
#                         "type": "error",
#                         "text": "Processing error"
#                     })
    
#     except WebSocketDisconnect:
#         print(f"🔌 Disconnected: {session_id}")
#     except Exception as e:
#         print(f"❌ WebSocket error: {e}")


# @app.post("/api/voice-query")
# async def voice_query_api(audio: UploadFile = File(...)):
#     """REST API fallback"""
#     try:
#         audio_bytes = await audio.read()
#         audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        
#         bot = StreamingVoiceBot()
#         query = await bot.transcribe_streaming(audio_array)
#         response = await bot.generate_response_streaming(query)
        
#         # Async TTS (non-blocking)
#         asyncio.create_task(speak_async(response))
        
#         return {
#             "transcript": query,
#             "response": response,
#             "status": "success"
#         }
#     except Exception as e:
#         return {"error": str(e), "status": "error"}


# @app.post("/api/ingest")
# async def trigger_ingestion():
#     """Trigger data ingestion"""
#     from data_store import UniversalIngestionEngine
    
#     engine = UniversalIngestionEngine(max_depth=2, max_pages=50)
    
#     # Example sources
#     sources = ["https://www.dcbbank.com"]
#     stats = await engine.ingest(sources)
    
#     return {"status": "success", "stats": stats}


# @app.get("/health")
# async def health_check():
#     """Health check"""
#     return {
#         "status": "healthy",
#         "active_sessions": len(session_manager.sessions),
#         "max_concurrent": session_manager.max_concurrent
#     }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
