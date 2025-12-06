"""Hybrid search combining vector + BM25"""
import asyncio
from typing import List, Dict
from rank_bm25 import BM25Okapi
import numpy as np

from retrieval.vector_store import VectorStore
from utils.config import config

class HybridSearch:
    """Combines vector and keyword search"""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.bm25_index = None
        self.bm25_corpus = []
        self.bm25_metadata = []
    
    def init_bm25(self, chunks: List[Dict]):
        """Initialize BM25 from corpus"""
        self.bm25_corpus = [chunk['text'].lower().split() for chunk in chunks]
        self.bm25_metadata = chunks
        self.bm25_index = BM25Okapi(self.bm25_corpus)
        print(f"✅ BM25 initialized with {len(chunks)} documents")
    
    def bm25_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """BM25 keyword search"""
        if self.bm25_index is None:
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        return [{
            "id": self.bm25_metadata[i]['id'],
            "score": float(scores[i]),
            "text": self.bm25_metadata[i]['text'],
            "source": self.bm25_metadata[i].get('source', 'N/A'),
            "page": self.bm25_metadata[i].get('page_number', 'N/A')
        } for i in top_indices if scores[i] > 0]
    
    async def hybrid_retrieve(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5
    ) -> str:
        """
        Hybrid search with Reciprocal Rank Fusion
        
        Args:
            query: Search query
            top_k: Number of results
            alpha: Weight for vector search (1-alpha for BM25)
            
        Returns:
            Formatted context string
        """
        # Parallel retrieval
        vector_task = asyncio.create_task(
            self.vector_store.search(query, top_k=10)
        )
        bm25_results = await asyncio.to_thread(
            self.bm25_search, query, top_k=10
        )
        vector_results = await vector_task
        
        # Reciprocal Rank Fusion (RRF)
        k = 60  # RRF constant
        fused_scores = {}
        
        for rank, result in enumerate(vector_results):
            doc_id = result['id']
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + alpha / (k + rank + 1)
        
        for rank, result in enumerate(bm25_results):
            doc_id = result['id']
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + (1 - alpha) / (k + rank + 1)
        
        # Get top documents
        all_results = {r['id']: r for r in vector_results + bm25_results}
        sorted_ids = sorted(
            fused_scores.keys(),
            key=lambda x: fused_scores[x],
            reverse=True
        )[:top_k]
        
        # Format context
        context_parts = []
        for i, doc_id in enumerate(sorted_ids):
            if doc_id in all_results:
                result = all_results[doc_id]
                context_parts.append(
                    f"[Source {i+1}] {result['text'][:400]}"
                )
        
        return "\n\n".join(context_parts)[:config.MAX_CONTEXT_LENGTH]

# Global instance
hybrid_search = HybridSearch()
