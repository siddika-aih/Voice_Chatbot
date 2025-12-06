"""Configuration management"""
import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Global configuration"""
    
    # Gemini API Keys with rotation
    GEMINI_KEYS: List[str] = [
        os.getenv("GEMINI_API_KEY"),
        os.getenv("GEMINI_API_KEY_2"),
        os.getenv("GEMINI_API_KEY_3"),
    ]
    GEMINI_KEYS = [k for k in GEMINI_KEYS if k]  # Remove None values
    
    if not GEMINI_KEYS:
        raise ValueError("At least one GEMINI_API_KEY must be set in .env")
    
    # Pinecone
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME: str = "dcb-voice-rag"
    PINECONE_DIMENSION: int = 1024
    
    # Model settings
    EMBEDDING_MODEL: str = "Alibaba-NLP/gte-large-en-v1.5"
    GEMINI_MODEL: str = "models/gemini-2.5-flash-native-audio-preview-09-2025"
    
    # Audio settings
    SAMPLE_RATE_INPUT: int = 16000  # For recording
    SAMPLE_RATE_OUTPUT: int = 24000  # For playback
    CHUNK_SIZE: int = 1024
    
    # RAG settings
    MAX_CONTEXT_LENGTH: int = 800
    TOP_K_RESULTS: int = 3
    HYBRID_ALPHA: float = 0.5  # Balance between vector and BM25
    
    # Voice settings
    VOICE_NAME: str = "Kore"  # Options: Kore, Zephyr, Aoede, Charon
    
    # Ingestion settings
    MAX_CRAWL_DEPTH: int = 2
    MAX_PAGES: int = 30
    CHUNK_SIZE_WORDS: int = 500
    CHUNK_OVERLAP: int = 50

config = Config()
