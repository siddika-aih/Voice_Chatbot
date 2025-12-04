# # from generate_rag_response import generate_rag_response
# # import pyttsx3

# # tts_engine = pyttsx3.init()
# # tts_engine.setProperty('rate', 180)  # Speed
# # tts_engine.setProperty('volume', 0.9)  # Volume

# # def text_to_speech(text):
# #     """4. NEW: Text → Speech (TTS)"""
# #     print("🔊 Speaking response...")
# #     tts_engine.say(text)
# #     tts_engine.runAndWait()
# #     print("✔ Speech complete")
# # fastapi_server.py - REST + WebSocket API for telephony integration
# from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# import asyncio

# import numpy as np
# # from streaming_voice_bot import StreamingVoiceBot
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
# @app.websocket("/ws/voice/{session_id}")
# async def voice_websocket(websocket: WebSocket, session_id: str):
#     """WebSocket endpoint for real-time voice streaming"""
#     await websocket.accept()
#     print(f"✅ WebSocket connected: {session_id}")
    
#     # Get or create session
#     from session_manager import session_manager
#     session = session_manager.get_session(session_id)
#     if not session:
#         session_id = await session_manager.create_session()
#         session = session_manager.get_session(session_id)
    
#     from main import StreamingVoiceBot
#     bot = StreamingVoiceBot()
#     bot.session_context = session["context"]
    
#     try:
#         while True:
#             # Receive audio chunk
#             data = await websocket.receive_json()
            
#             if data["type"] == "audio":
#                 try:
#                     # Decode audio
#                     audio_bytes = base64.b64decode(data["audio"])
#                     audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
                    
#                     # Transcribe
#                     query = await bot.transcribe_streaming(audio_array)
#                     print(f"👤 Query: {query}")
                    
#                     # Send transcript immediately
#                     await websocket.send_json({"type": "transcript", "text": query})
                    
#                     # Generate response
#                     response = await bot.generate_response_streaming(query)
#                     print(f"🤖 Response: {response}")
                    
#                     # Send response
#                     await websocket.send_json({"type": "response", "text": response})
                    
#                     # Update session
#                     session["context"].append({"user": query, "assistant": response})
#                     session["query_count"] += 1
                    
#                     print("✅ Response sent successfully")
                    
#                 except Exception as e:
#                     print(f"❌ Processing error: {e}")
#                     import traceback
#                     traceback.print_exc()
                    
#                     try:
#                         await websocket.send_json({
#                             "type": "error", 
#                             "text": "Sorry, I encountered an error."
#                         })
#                     except:
#                         pass
                
#     except WebSocketDisconnect:
#         print(f"🔌 Session {session_id} disconnected")
#     except Exception as e:
#         print(f"❌ WebSocket error: {e}")
#         import traceback
#         traceback.print_exc()

# from fastapi import File, UploadFile

# @app.post("/api/voice-query")
# async def voice_query_api(audio: UploadFile = File(...)):
#     """Simple REST endpoint for voice queries"""
#     try:
#         # Read audio
#         audio_bytes = await audio.read()
#         audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        
#         # Process
#         from main import StreamingVoiceBot
#         bot = StreamingVoiceBot()
        
#         query = await bot.transcribe_streaming(audio_array)
#         response = await bot.generate_response_streaming(query)
        
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


# @app.post("/api/ingest")
# async def trigger_ingestion():
#     """Trigger data ingestion from DCB Bank website"""
#     from data_store import ingest_dcb_website
    
#     chunks = await ingest_dcb_website()
#     return {"status": "success", "chunks_ingested": len(chunks)}

# @app.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     active_sessions = len(session_manager.sessions)
#     return {
#         "status": "healthy",
#         "active_sessions": active_sessions,
#         "max_concurrent": session_manager.max_concurrent
#     }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)



from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import numpy as np
from main import StreamingVoiceBot
from session_manager import session_manager
import base64
import uvicorn

app = FastAPI(title="DCB Voice Bot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# DON'T initialize here - remove this line:
# tts_engine = pyttsx3.init()  ❌

# NEW: Initialize inside function
def speak_now(text):
    """Speak response using pyttsx3 - lazy initialization"""
    try:
        import pyttsx3
        engine = pyttsx3.init()  # Initialize HERE ✅
        engine.setProperty('rate', 180)
        engine.setProperty('volume', 1.0)
        
        print("🔊 Speaking response...")
        engine.say(text)
        engine.runAndWait()
        print("✔ Speech complete")
        
        engine.stop()  # Cleanup
        del engine
        
    except Exception as e:
        print(f"TTS Error: {e}")

@app.post("/api/voice-query")
async def voice_query_api(audio: UploadFile = File(...)):
    """REST API with voice response"""
    try:
        # Read audio
        audio_bytes = await audio.read()
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        
        # Process
        bot = StreamingVoiceBot()
        query = await bot.transcribe_streaming(audio_array)
        response = await bot.generate_response_streaming(query)
        
        # SPEAK IT!
        speak_now(response)
        
        return {
            "transcript": query,
            "response": response,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }

@app.websocket("/ws/voice/{session_id}")
async def voice_websocket(websocket: WebSocket, session_id: str):
    """WebSocket with voice output"""
    await websocket.accept()
    print(f"✅ WebSocket connected: {session_id}")
    
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
                try:
                    audio_bytes = base64.b64decode(data["audio"])
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
                    
                    query = await bot.transcribe_streaming(audio_array)
                    await websocket.send_json({"type": "transcript", "text": query})
                    
                    response = await bot.generate_response_streaming(query)
                    await websocket.send_json({"type": "response", "text": response})
                    
                    # SPEAK IT!
                    speak_now(response)
                    
                    session["context"].append({"user": query, "assistant": response})
                    session["query_count"] += 1
                    
                except Exception as e:
                    print(f"Error: {e}")
                
    except WebSocketDisconnect:
        print(f"🔌 Disconnected: {session_id}")

@app.post("/api/ingest")
async def trigger_ingestion():
    """Trigger data ingestion"""
    from data_store import ingest_dcb_website
    chunks = await ingest_dcb_website()
    return {"status": "success", "chunks_ingested": len(chunks)}

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "active_sessions": len(session_manager.sessions),
        "max_concurrent": session_manager.max_concurrent
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080, workers=4)
