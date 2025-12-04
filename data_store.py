
# import fitz
# from rich import print
# from sentence_transformers import SentenceTransformer


# def data_extract_chunk(file_path):
#     doc=fitz.open(file_path)
#     # print(doc.page_count)

#     fitz.TOOLS.mupdf_display_errors(False)
#     content=[]

#     for page_number in range(len(doc)):
#         page=doc.load_page(page_number)
#         page_text=page.get_text("text")
#         stripped = page_text.split()
#         chunk_size=1000
#         chunk_overload=100

#         for j in range(0,len(stripped), chunk_size):
#             chunk__text = " ".join(list(stripped[j:j+chunk_size]))

#             content.append(
#                 {
#                     "page_number":page_number + 1,
#                     "text":chunk__text
#                 }
#             )


#     embedding=SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5",trust_remote_code=True)
#     em=embedding.encode("hello")

#     page_content=[chunk['text'] for chunk in content]
#     page_number=[chunk['page_number'] for chunk in content]
#     content_embedding=[embedding.encode(chunk['text']) for chunk in content]
    
#     return content,content_embedding


# # data_extract_chunk(file_path="D:\\FSS\\ai-test\\MACHINE LEARNING(R17A0534) (1).pdf")


# def data_store_vectorstore(content,content_embedding):
    

# def data_retrieve(query):

import fitz
import os
import uuid
from typing import List, Dict, Any
from rich import print
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import numpy as np

load_dotenv()
# Pinecone setup
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))  # Set your PINECONE_API_KEY env var

# Global embedding model (load once)
embedding_model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)

INDEX_NAME = "rag-pdf-index"
DIMENSION = 1024  # gte-large-en-v1.5 dimension

def init_pinecone_index():
    """Initialize Pinecone index if not exists"""
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"🆕 Creating Pinecone index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pc.Index(INDEX_NAME)
    return index

def data_extract_chunk(file_path: str) -> tuple[List[Dict], List[np.ndarray]]:
    """Extract & chunk PDF + generate embeddings"""
    doc = fitz.open(file_path)
    fitz.TOOLS.mupdf_display_errors(False)
    content = []

    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        page_text = page.get_text("text")
        words = page_text.split()
        chunk_size = 1000
        overlap = 100

        for j in range(0, len(words), chunk_size - overlap):
            chunk_text = " ".join(words[j:j + chunk_size])
            if len(chunk_text.strip()) > 50:  # Skip tiny chunks
                content.append({
                    "id": str(uuid.uuid4()),
                    "page_number": page_number + 1,
                    "text": chunk_text.strip(),
                    "file_name": os.path.basename(file_path)
                })

    doc.close()
    
    # Generate embeddings (batch for speed)
    page_content = [chunk['text'] for chunk in content]
    content_embeddings = embedding_model.encode(page_content)
    
    print(f"✅ Extracted {len(content)} chunks from {len(set(c['page_number'] for c in content))} pages")
    return content, content_embeddings

def data_store_vectorstore(content: List[Dict], content_embedding: List[np.ndarray], index=None):
    """Store chunks + embeddings in Pinecone"""
    if index is None:
        index = init_pinecone_index()
    
    # Prepare vectors for upsert
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(content, content_embedding)):
        vectors.append({
            "id": chunk["id"],
            "values": embedding.tolist(),
            "metadata": {
                "page_number": chunk["page_number"],
                "file_name": chunk["file_name"],
                "text": chunk["text"][:1000]  # Truncate for Pinecone limits
            }
        })
    
    # Batch upsert (1000 max per batch)
    batch_size = 1000
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"📤 Upserted batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
    
    print(f"✅ Stored {len(vectors)} vectors in Pinecone!")

def data_retrieve(query: str, top_k: int = 5, index=None) -> List[Dict]:
    """Retrieve relevant chunks from Pinecone"""
    if index is None:
        index = pc.Index(INDEX_NAME)
    
    # Generate query embedding
    query_embedding = embedding_model.encode([query])[0]
    
    # Query Pinecone
    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )
    
    # Format results
    retrieved = []
    for match in results['matches']:
        retrieved.append({
            "id": match['id'],
            "score": match['score'],
            "page_number": match['metadata']['page_number'],
            "file_name": match['metadata']['file_name'],
            "text": match['metadata']['text']
        })
    
    print(f"🔍 Retrieved {len(retrieved)} most relevant chunks")
    return retrieved

# 🧪 COMPLETE USAGE EXAMPLE
def main():
    file_path = r"C:\Users\admin\Downloads\Medical_book.pdf"
    
    # 1. EXTRACT + EMBED
    print("📄 Extracting PDF...")
    content, embeddings = data_extract_chunk(file_path)
    
    # 2. STORE in Pinecone
    print("\n💾 Storing in Pinecone...")
    data_store_vectorstore(content, embeddings)
    
    # 3. RETRIEVE
    print("\n🔍 Testing retrieval...")
    results = data_retrieve("machine learning algorithms", top_k=3)
    
    for i, result in enumerate(results, 1):
        print(f"\n📄 [{i}] Score: {result['score']:.3f} (Page {result['page_number']})")
        print(f"   {result['text'][:200]}...")

if __name__ == "__main__":
    main()

