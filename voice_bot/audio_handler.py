"""Audio I/O handling"""
import asyncio
import pyaudio
from typing import Optional

from utils.config import config

class AudioHandler:
    """Manages audio input/output streams"""
    
    def __init__(self):
        self.pya = pyaudio.PyAudio()
        self.input_stream: Optional[pyaudio.Stream] = None
        self.output_stream: Optional[pyaudio.Stream] = None
        
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
            frames_per_buffer=config.CHUNK_SIZE
        )
        print("🎤 Microphone ready")
    
    async def start_output_stream(self):
        """Initialize speaker output"""
        self.output_stream = await asyncio.to_thread(
            self.pya.open,
            format=pyaudio.paInt16,
            channels=1,
            rate=config.SAMPLE_RATE_OUTPUT,
            output=True
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
        """Write audio chunk to speaker"""
        await asyncio.to_thread(self.output_stream.write, data)
    
    def cleanup(self):
        """Close audio streams"""
        if self.input_stream:
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.close()
        self.pya.terminate()
