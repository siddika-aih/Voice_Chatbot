import requests
from bs4 import BeautifulSoup
import uuid
import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import numpy as np
from rich import print

load_dotenv()
# Pinecone setup
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))  # Set your PINECONE_API_KEY env var

# Global embedding model (load once)
embedding_model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)

INDEX_NAME = "rag-web-index"
DIMENSION = 1024  # gte-large-en-v1.5 dimension


def init_pinecone_index():
    """Initialize Pinecone index if not exists"""
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating Pinecone index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pc.Index(INDEX_NAME)
    return index


def data_extract_chunk_from_url(url: str) -> tuple[List[Dict], np.ndarray]:
    """Extract text from a website URL, chunk it and generate embeddings"""
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    # Remove script/style and other non-content elements
    for script in soup(["script", "style", "nav", "footer", "header", "form", "aside"]):
        script.decompose()

    page_text = soup.get_text(separator=' ', strip=True)
    words = page_text.split()
    chunk_size = 1000
    overlap = 100
    content = []

    for j in range(0, len(words), chunk_size - overlap):
        chunk_text = " ".join(words[j:j + chunk_size])
        if len(chunk_text.strip()) > 50:
            content.append({
                "id": str(uuid.uuid4()),
                "text": chunk_text.strip(),
                "url": url,
                "file_name": url.split("/")[-1] or "web_content"
            })

    page_content = [chunk['text'] for chunk in content]
    content_embeddings = embedding_model.encode(page_content)

    print(f"✅ Extracted {len(content)} chunks from website {url}")
    return content, content_embeddings


def store_chunks_to_pinecone(index, chunks: List[Dict], embeddings: np.ndarray):
    # Prepare data to upsert
    to_upsert = []
    for chunk, embed in zip(chunks, embeddings):
        metadata = {
            "text": chunk["text"],
            "url": chunk["url"],
            "file_name": chunk["file_name"]
        }
        to_upsert.append((chunk["id"], embed.tolist(), metadata))

    # Upsert to Pinecone index
    index.upsert(vectors=to_upsert)
    print(f"Stored {len(to_upsert)} vectors to Pinecone index {INDEX_NAME}")


def data_retrieve(query: str, top_k: int = 5, index=None) -> List[Dict]:
    """Retrieve relevant chunks from Pinecone"""
    if index is None:
        index = pc.Index(INDEX_NAME)

    query_embedding = embedding_model.encode([query])[0]

    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )

    retrieved = []
    for match in results['matches']:
        retrieved.append({
            "id": match['id'],
            "score": match['score'],
            "url": match['metadata']['url'],
            "file_name": match['metadata']['file_name'],
            "text": match['metadata']['text']
        })

    print(f"🔍 Retrieved {len(retrieved)} most relevant chunks")
    return retrieved


def main():
    website_url = "https://example.com"  # Replace with your target URL

    print("🌐 Extracting website content and generating embeddings...")
    content, embeddings = data_extract_chunk_from_url(website_url)

    print("\n💾 Initializing Pinecone and storing vectors...")
    index = init_pinecone_index()
    store_chunks_to_pinecone(index, content, embeddings)

    print("\n🔍 Testing retrieval for query 'machine learning algorithms'...")
    results = data_retrieve("machine learning algorithms", top_k=3)

    for i, result in enumerate(results, 1):
        print(f"\n🌐 [{i}] Score: {result['score']:.3f} (URL: {result['url']})")
        print(f"   {result['text'][:200]}...")


if __name__ == "__main__":
    main()
