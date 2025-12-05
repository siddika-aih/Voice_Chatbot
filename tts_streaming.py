# tts_streaming.py - Fixed TTS that actually speaks!
import pyttsx3
import threading
import queue
import time

class StreamingTTS:
    """Real-time TTS with interruption - FIXED"""
    
    def __init__(self):
        self.text_queue = queue.Queue()
        self.is_active = False
        self.worker_thread = None
        self.stop_flag = threading.Event()
        self.lock = threading.Lock()
        
    def _worker(self):
        """Background worker that speaks queued text"""
        # Create engine inside thread (Windows requirement)
        engine = pyttsx3.init()
        engine.setProperty('rate', 180)
        engine.setProperty('volume', 1.0)
        
        print("🔊 TTS Worker ready")
        
        while self.is_active:
            try:
                text = self.text_queue.get(timeout=0.1)
                
                if text and not self.stop_flag.is_set():
                    print(f"🔊 Speaking: {text[:50]}...")
                    engine.say(text)
                    engine.runAndWait()
                    print("✔ Spoke")
                
                self.text_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS Error: {e}")
        
        # Cleanup
        try:
            engine.stop()
            del engine
        except:
            pass
    
    def start(self):
        """Start TTS worker"""
        with self.lock:
            if not self.is_active:
                self.is_active = True
                self.stop_flag.clear()
                self.worker_thread = threading.Thread(target=self._worker, daemon=True)
                self.worker_thread.start()
                time.sleep(0.1)  # Let worker initialize
    
    def speak(self, text):
        """Add text to speak"""
        if not self.is_active:
            self.start()
        
        if text and text.strip():
            self.text_queue.put(text.strip())
    
    def speak_full(self, text):
        """Speak full text (non-streaming)"""
        if not self.is_active:
            self.start()
        
        if text and text.strip():
            self.text_queue.put(text.strip())
    
    def stop(self):
        """Stop speaking immediately"""
        self.stop_flag.set()
        
        # Clear queue
        while not self.text_queue.empty():
            try:
                self.text_queue.get_nowait()
                self.text_queue.task_done()
            except:
                break
        
        time.sleep(0.1)
        self.stop_flag.clear()
    
    def restart(self):
        """Restart TTS worker"""
        self.shutdown()
        time.sleep(0.2)
        self.start()
    
    def shutdown(self):
        """Shutdown completely"""
        self.is_active = False
        self.stop_flag.set()
        
        # Clear queue
        while not self.text_queue.empty():
            try:
                self.text_queue.get_nowait()
            except:
                break
        
        if self.worker_thread:
            self.worker_thread.join(timeout=1)
            self.worker_thread = None


# Global TTS
tts = StreamingTTS()
