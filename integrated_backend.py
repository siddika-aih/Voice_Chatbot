import asyncio
import websockets
import json
import subprocess
import sys
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# --- Configuration ---
FRONTEND_BUILD_DIR = "frontend-react/dist"
HOST = "127.0.0.1"
PORT = 8000 # This will be the port for the integrated backend

# --- FastAPI App Setup ---
app = FastAPI(title="DCB Voice Bot Integrated Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global State for Bot Process Management ---
connected_clients = set()
bot_process = None
ingestion_process = None # To manage the ingestion process
bot_output_queue = asyncio.Queue() # Queue for bot's stdout/stderr
ingestion_output_queue = asyncio.Queue() # Queue for ingestion's stdout/stderr

# --- WebSocket Helper Functions ---

async def register(websocket: WebSocket):
    """Register a new client connection."""
    connected_clients.add(websocket)
    print(f"New client connected. Total clients: {len(connected_clients)}")
    await send_bot_state()

async def unregister(websocket: WebSocket):
    """Unregister a client connection."""
    connected_clients.remove(websocket)
    print(f"Client disconnected. Total clients: {len(connected_clients)}")

async def broadcast(message: dict):
    """Send a message to all connected clients."""
    if connected_clients:
        # Ensure message is JSON serializable
        message_str = json.dumps(message)
        await asyncio.gather(*[client.send_text(message_str) for client in connected_clients])

async def send_log(source: str, message: str, log_type: str = 'info'):
    """Broadcast a log message."""
    log_entry = {
        "type": "log",
        "message": f"[{source}] {message.strip()}",
        "log_type": log_type
    }
    await broadcast(log_entry)

async def send_bot_state():
    """Broadcast the current running state of the bot."""
    state = {
        "type": "bot_state",
        "running": bot_process is not None and bot_process.returncode is None
    }
    await broadcast(state)

# --- Bot Process Management ---

async def read_stream(stream, source: str, output_queue: asyncio.Queue):
    """Read lines from a stream and put them into a queue."""
    while True:
        line = await stream.readline()
        if line:
            await output_queue.put({"source": source, "message": line.decode('utf-8', errors='ignore')})
        else:
            break

async def process_output_queue(output_queue: asyncio.Queue):
    """Continuously read from an output queue and broadcast logs."""
    while True:
        output = await output_queue.get()
        await send_log(output["source"], output["message"])
        output_queue.task_done()

async def start_bot():
    """Start the voice bot as a subprocess."""
    global bot_process
    if bot_process and bot_process.returncode is None:
        await send_log("System", "Bot is already running.", "error")
        return

    await send_log("System", "Starting DCB VoiceBot process...", "success")
    
    try:
        # Using sys.executable to ensure we use the same Python interpreter
        # run_bot.py is in the root directory
        bot_process = await asyncio.create_subprocess_exec(
            sys.executable, 'run_bot.py',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd() # Ensure it runs from the project root
        )

        asyncio.create_task(read_stream(bot_process.stdout, 'Bot-Stdout', bot_output_queue))
        asyncio.create_task(read_stream(bot_process.stderr, 'Bot-Stderr', bot_output_queue))

        await send_bot_state()
        await send_log("System", f"Bot process started with PID: {bot_process.pid}", "success")

        asyncio.create_task(monitor_bot_process())

    except Exception as e:
        await send_log("System", f"Failed to start bot process: {e}", "error")
        bot_process = None
        await send_bot_state()

async def monitor_bot_process():
    """Monitors the bot process and updates state when it exits."""
    global bot_process
    if bot_process:
        await bot_process.wait()
        await send_log("System", f"Bot process (PID: {bot_process.pid}) has stopped.", "info")
        bot_process = None
        await send_bot_state()

async def stop_bot():
    """Stop the voice bot process."""
    global bot_process
    if bot_process and bot_process.returncode is None:
        await send_log("System", "Stopping bot process...", "info")
        bot_process.terminate()
        try:
            await asyncio.wait_for(bot_process.wait(), timeout=5.0)
            await send_log("System", "Bot process terminated.", "success")
        except asyncio.TimeoutError:
            await send_log("System", "Process did not terminate gracefully, killing.", "error")
            bot_process.kill()
        
        bot_process = None
        await send_bot_state()
    else:
        await send_log("System", "Bot is not currently running.", "error")

async def start_ingestion(url: str):
    """Start the data ingestion process."""
    global ingestion_process
    if ingestion_process and ingestion_process.returncode is None:
        await send_log("System", "Data ingestion is already running.", "error")
        return

    await send_log("System", f"Starting data ingestion for URL: {url}...", "info")
    try:
        # Assuming ingest_data.py can take a URL as a command-line argument
        ingestion_process = await asyncio.create_subprocess_exec(
            sys.executable, 'ingest_data.py', url,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd()
        )
        asyncio.create_task(read_stream(ingestion_process.stdout, 'Ingestion-Stdout', ingestion_output_queue))
        asyncio.create_task(read_stream(ingestion_process.stderr, 'Ingestion-Stderr', ingestion_output_queue))

        await send_log("System", f"Ingestion process started with PID: {ingestion_process.pid}", "success")
        asyncio.create_task(monitor_ingestion_process())

    except Exception as e:
        await send_log("System", f"Failed to start ingestion process: {e}", "error")
        ingestion_process = None

async def monitor_ingestion_process():
    """Monitors the ingestion process and updates state when it exits."""
    global ingestion_process
    if ingestion_process:
        await ingestion_process.wait()
        if ingestion_process.returncode == 0:
            await send_log("System", f"Ingestion process (PID: {ingestion_process.pid}) completed successfully.", "success")
        else:
            await send_log("System", f"Ingestion process (PID: {ingestion_process.pid}) failed with exit code {ingestion_process.returncode}.", "error")
        ingestion_process = None

# --- Main WebSocket Handler ---

@app.websocket("/ws/control")
async def websocket_control_endpoint(websocket: WebSocket):
    """WebSocket endpoint for frontend to send commands and receive logs/status."""
    await websocket.accept()
    await register(websocket)
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                command = data.get('command')

                if command == 'start':
                    asyncio.create_task(start_bot())
                elif command == 'stop':
                    asyncio.create_task(stop_bot())
                elif command == 'ingest_url':
                    url = data.get('url')
                    if url:
                        asyncio.create_task(start_ingestion(url))
                    else:
                        await send_log("System", "Ingest URL command received without a URL.", "error")
                else:
                    await send_log("System", f"Unknown command: {command}", "error")

            except json.JSONDecodeError:
                await send_log("System", "Received invalid JSON from frontend.", "error")
            except Exception as e:
                await send_log("System", f"Error processing frontend command: {e}", "error")
    finally:
        await unregister(websocket)

# --- Serve Frontend Static Files ---
# This must be the LAST route registered, otherwise it will catch all other routes
app.mount("/", StaticFiles(directory=FRONTEND_BUILD_DIR, html=True), name="static")

# --- Main Entry Point ---
if __name__ == "__main__":
    import uvicorn
    # Check if frontend build directory exists
    if not os.path.isdir(FRONTEND_BUILD_DIR):
        print(f"Error: Frontend build directory '{FRONTEND_BUILD_DIR}' not found.")
        print("Please run 'npm run build' in the 'frontend-react' directory first.")
        sys.exit(1)
    
    print(f"Serving frontend from: {FRONTEND_BUILD_DIR}")
    print(f"Starting integrated backend on http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)
