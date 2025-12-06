"""Audio I/O handling"""
import asyncio
import pyaudio
from typing import Optional
import threading
from utils.config import config

class AudioHandler:
    """Manages audio input/output streams"""
    
    def __init__(self):
        self.pya = pyaudio.PyAudio()
        self.input_stream: Optional[pyaudio.Stream] = None
        self.output_stream: Optional[pyaudio.Stream] = None
        self.output_lock = threading.Lock()  # ✅ Thread-safe output
        
    async def start_input_stream(self):
        """Initialize microphone input"""
        mic_info = self.pya.get_default_input_device_info()
        self.input_stream = await asyncio.to_thread(
            self.pya.open,
            format=pyaudio.paInt16,
            channels=1,
            rate=config.SAMPLE_RATE_INPUT,
            input=True,
            input_device_index=mic_info['index'],
            frames_per_buffer=config.CHUNK_SIZE,
            stream_callback=None
        )
        print("🎤 Microphone ready")
    
    async def start_output_stream(self):
        """Initialize speaker output"""
        self.output_stream = await asyncio.to_thread(
            self.pya.open,
            format=pyaudio.paInt16,
            channels=1,
            rate=config.SAMPLE_RATE_OUTPUT,
            output=True,
            frames_per_buffer=config.CHUNK_SIZE * 2,
            stream_callback=None
        )
        print("🔊 Speaker ready")
    
    async def read_audio_chunk(self) -> bytes:
        """Read audio chunk from microphone"""
        return await asyncio.to_thread(
            self.input_stream.read,
            config.CHUNK_SIZE,
            exception_on_overflow=False
        )
    
    async def write_audio_chunk(self, data: bytes):
        # """Write audio chunk to speaker"""
        # await asyncio.to_thread(self.output_stream.write, data)
        """Write audio chunk to speaker (thread-safe)"""
        if not data:
            return
            
        try:
            with self.output_lock:
                await asyncio.to_thread(self.output_stream.write, data)
        except Exception as e:
            print(f"⚠️ Speaker write error: {e}")
    
    # def cleanup(self):
    #     """Close audio streams"""
    #     if self.input_stream:
    #         self.input_stream.close()
    #     if self.output_stream:
    #         self.output_stream.close()
    #     self.pya.terminate()
    
    def cleanup(self):
        """Close audio streams"""
        try:
            if self.input_stream and self.input_stream.is_active():
                self.input_stream.stop_stream()
                self.input_stream.close()
            if self.output_stream and self.output_stream.is_active():
                self.output_stream.stop_stream()
                self.output_stream.close()
            self.pya.terminate()
            print("🔇 Audio streams closed")
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")
