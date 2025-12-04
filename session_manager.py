# session_manager.py - Handle multiple concurrent users
import asyncio
from collections import defaultdict
import time
from typing import Dict
import uuid

class SessionManager:
    def __init__(self, max_concurrent=400):
        self.sessions: Dict[str, dict] = {}
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    async def create_session(self) -> str:
        """Create new session with rate limiting"""
        async with self.semaphore:
            session_id = str(uuid.uuid4())
            self.sessions[session_id] = {
                "created": time.time(),
                "last_active": time.time(),
                "context": [],
                "query_count": 0
            }
            return session_id
    
    def get_session(self, session_id: str) -> dict:
        """Retrieve session data"""
        if session_id in self.sessions:
            self.sessions[session_id]["last_active"] = time.time()
            return self.sessions[session_id]
        return None
    
    def cleanup_stale_sessions(self, timeout=3600):
        """Remove sessions inactive for >1 hour"""
        current_time = time.time()
        stale = [
            sid for sid, data in self.sessions.items()
            if current_time - data["last_active"] > timeout
        ]
        for sid in stale:
            del self.sessions[sid]
        return len(stale)

# Global session manager
session_manager = SessionManager(max_concurrent=400)
