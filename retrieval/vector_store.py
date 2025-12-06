"""Vector database operations with Pinecone"""
import asyncio
from typing import List, Dict
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import numpy as np

from utils.config import config

class VectorStore:
    """Manages Pinecone vector database"""
    
    def __init__(self):
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.embedding_model = SentenceTransformer(
            config.EMBEDDING_MODEL,
            trust_remote_code=True
        )
        self.index = self._init_index()
        
    def _init_index(self):
        """Initialize or connect to Pinecone index"""
        if config.PINECONE_INDEX_NAME not in self.pc.list_indexes().names():
            print(f"🆕 Creating Pinecone index: {config.PINECONE_INDEX_NAME}")
            self.pc.create_index(
                name=config.PINECONE_INDEX_NAME,
                dimension=config.PINECONE_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        return self.pc.Index(config.PINECONE_INDEX_NAME)
    
    async def store_chunks(self, chunks: List[Dict]):
        """Store chunks with embeddings in Pinecone"""
        # Generate embeddings in batch
        texts = [chunk['text'] for chunk in chunks]
        embeddings = await asyncio.to_thread(
            self.embedding_model.encode,
            texts,
            show_progress_bar=True
        )
        
        # Prepare vectors for upsert
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vectors.append({
                "id": chunk["id"],
                "values": embedding.tolist(),
                "metadata": {
                    "text": chunk["text"][:1000],  # Pinecone metadata limit
                    "page_number": str(chunk.get("page_number", "N/A")),
                    "source": chunk["source"],
                    "type": chunk["type"]
                }
            })
        
        # Batch upsert (max 100 per batch for stability)
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            await asyncio.to_thread(self.index.upsert, vectors=batch)
            print(f"📤 Upserted batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
        
        print(f"✅ Stored {len(vectors)} vectors")
    
    async def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Vector similarity search"""
        # Generate query embedding
        query_embedding = await asyncio.to_thread(
            self.embedding_model.encode,
            [query]
        )
        
        # Query Pinecone
        results = await asyncio.to_thread(
            self.index.query,
            vector=query_embedding[0].tolist(),
            top_k=top_k,
            include_metadata=True
        )
        
        return [{
            "id": match['id'],
            "score": match['score'],
            "text": match['metadata']['text'],
            "source": match['metadata']['source'],
            "page": match['metadata']['page_number']
        } for match in results['matches']]
