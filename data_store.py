
# # import fitz
# # from rich import print
# # from sentence_transformers import SentenceTransformer


# # def data_extract_chunk(file_path):
# #     doc=fitz.open(file_path)
# #     # print(doc.page_count)

# #     fitz.TOOLS.mupdf_display_errors(False)
# #     content=[]

# #     for page_number in range(len(doc)):
# #         page=doc.load_page(page_number)
# #         page_text=page.get_text("text")
# #         stripped = page_text.split()
# #         chunk_size=1000
# #         chunk_overload=100

# #         for j in range(0,len(stripped), chunk_size):
# #             chunk__text = " ".join(list(stripped[j:j+chunk_size]))

# #             content.append(
# #                 {
# #                     "page_number":page_number + 1,
# #                     "text":chunk__text
# #                 }
# #             )


# #     embedding=SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5",trust_remote_code=True)
# #     em=embedding.encode("hello")

# #     page_content=[chunk['text'] for chunk in content]
# #     page_number=[chunk['page_number'] for chunk in content]
# #     content_embedding=[embedding.encode(chunk['text']) for chunk in content]
    
# #     return content,content_embedding


# # # data_extract_chunk(file_path="D:\\FSS\\ai-test\\MACHINE LEARNING(R17A0534) (1).pdf")


# # def data_store_vectorstore(content,content_embedding):
    

# # def data_retrieve(query):

# import fitz
# import os
# import uuid
# from typing import List, Dict, Any
# from rich import print
# from sentence_transformers import SentenceTransformer
# from pinecone import Pinecone, ServerlessSpec
# from dotenv import load_dotenv
# import numpy as np


# # Global embedding model (load once)
# embedding_model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)

# INDEX_NAME = "rag-pdf-index"
# DIMENSION = 1024  # gte-large-en-v1.5 dimension

# def init_pinecone_index():
#     """Initialize Pinecone index if not exists"""
#     if INDEX_NAME not in pc.list_indexes().names():
#         print(f"🆕 Creating Pinecone index: {INDEX_NAME}")
#         pc.create_index(
#             name=INDEX_NAME,
#             dimension=DIMENSION,
#             metric="cosine",
#             spec=ServerlessSpec(cloud="aws", region="us-east-1")
#         )
#     index = pc.Index(INDEX_NAME)
#     return index

# def data_extract_chunk(file_path: str) -> tuple[List[Dict], List[np.ndarray]]:
#     """Extract & chunk PDF + generate embeddings"""
#     doc = fitz.open(file_path)
#     fitz.TOOLS.mupdf_display_errors(False)
#     content = []

#     for page_number in range(len(doc)):
#         page = doc.load_page(page_number)
#         page_text = page.get_text("text")
#         words = page_text.split()
#         chunk_size = 1000
#         overlap = 100

#         for j in range(0, len(words), chunk_size - overlap):
#             chunk_text = " ".join(words[j:j + chunk_size])
#             if len(chunk_text.strip()) > 50:  # Skip tiny chunks
#                 content.append({
#                     "id": str(uuid.uuid4()),
#                     "page_number": page_number + 1,
#                     "text": chunk_text.strip(),
#                     "file_name": os.path.basename(file_path)
#                 })

#     doc.close()
    
#     # Generate embeddings (batch for speed)
#     page_content = [chunk['text'] for chunk in content]
#     content_embeddings = embedding_model.encode(page_content)
    
#     print(f"✅ Extracted {len(content)} chunks from {len(set(c['page_number'] for c in content))} pages")
#     return content, content_embeddings

# def data_store_vectorstore(content: List[Dict], content_embedding: List[np.ndarray], index=None):
#     """Store chunks + embeddings in Pinecone"""
#     if index is None:
#         index = init_pinecone_index()
    
#     # Prepare vectors for upsert
#     vectors = []
#     for i, (chunk, embedding) in enumerate(zip(content, content_embedding)):
#         vectors.append({
#             "id": chunk["id"],
#             "values": embedding.tolist(),
#             "metadata": {
#                 "page_number": chunk["page_number"],
#                 "file_name": chunk["file_name"],
#                 "text": chunk["text"][:1000]  # Truncate for Pinecone limits
#             }
#         })
    
#     # Batch upsert (1000 max per batch)
#     batch_size = 1000
#     for i in range(0, len(vectors), batch_size):
#         batch = vectors[i:i + batch_size]
#         index.upsert(vectors=batch)
#         print(f"📤 Upserted batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
    
#     print(f"✅ Stored {len(vectors)} vectors in Pinecone!")

# def data_retrieve(query: str, top_k: int = 5, index=None) -> List[Dict]:
#     """Retrieve relevant chunks from Pinecone"""
#     if index is None:
#         index = pc.Index(INDEX_NAME)
    
#     # Generate query embedding
#     query_embedding = embedding_model.encode([query])[0]
    
#     # Query Pinecone
#     results = index.query(
#         vector=query_embedding.tolist(),
#         top_k=top_k,
#         include_metadata=True
#     )
    
#     # Format results
#     retrieved = []
#     for match in results['matches']:
#         retrieved.append({
#             "id": match['id'],
#             "score": match['score'],
#             "page_number": match['metadata']['page_number'],
#             "file_name": match['metadata']['file_name'],
#             "text": match['metadata']['text']
#         })
    
#     print(f"🔍 Retrieved {len(retrieved)} most relevant chunks")
#     return retrieved

# # 🧪 COMPLETE USAGE EXAMPLE
# def main():
#     file_path = r"C:\Users\admin\Downloads\Medical_book.pdf"
    
#     # 1. EXTRACT + EMBED
#     print("📄 Extracting PDF...")
#     content, embeddings = data_extract_chunk(file_path)
    
#     # 2. STORE in Pinecone
#     print("\n💾 Storing in Pinecone...")
#     data_store_vectorstore(content, embeddings)
    
#     # 3. RETRIEVE
#     print("\n🔍 Testing retrieval...")
#     results = data_retrieve("machine learning algorithms", top_k=3)
    
#     for i, result in enumerate(results, 1):
#         print(f"\n📄 [{i}] Score: {result['score']:.3f} (Page {result['page_number']})")
#         print(f"   {result['text'][:200]}...")

# if __name__ == "__main__":
#     main()

# enhanced_data_store.py - Hybrid search with BM25 + Vector
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from rank_bm25 import BM25Okapi
import asyncio
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
embedding_model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)
INDEX_NAME = "rag-pdf-index"

# Global BM25 index (load from stored chunks)
bm25_index = None
bm25_corpus = []
bm25_metadata = []

def init_bm25(chunks: List[Dict]):
    """Initialize BM25 from chunk corpus"""
    global bm25_index, bm25_corpus, bm25_metadata
    
    bm25_corpus = [chunk['text'].lower().split() for chunk in chunks]
    bm25_metadata = chunks
    bm25_index = BM25Okapi(bm25_corpus)
    print(f"✅ BM25 initialized with {len(chunks)} documents")

async def vector_retrieve(query: str, top_k: int = 10) -> List[Dict]:
    """Async vector search in Pinecone"""
    index = pc.Index(INDEX_NAME)
    query_embedding = embedding_model.encode([query])[0]
    
    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )
    
    return [{
        "id": match['id'],
        "score": match['score'],
        "text": match['metadata']['text'],
        "page": match['metadata'].get('page_number', 'N/A'),
        "source": "vector"
    } for match in results['matches']]

def bm25_retrieve(query: str, top_k: int = 10) -> List[Dict]:
    """BM25 keyword search"""
    if bm25_index is None:
        return []
    
    tokenized_query = query.lower().split()
    scores = bm25_index.get_scores(tokenized_query)
    
    # Get top_k indices
    top_indices = np.argsort(scores)[-top_k:][::-1]
    
    return [{
        "id": bm25_metadata[i]['id'],
        "score": float(scores[i]),
        "text": bm25_metadata[i]['text'],
        "page": bm25_metadata[i].get('page_number', 'N/A'),
        "source": "bm25"
    } for i in top_indices if scores[i] > 0]

async def hybrid_retrieve(query: str, top_k: int = 5, alpha: float = 0.5) -> str:
    """Hybrid search combining vector + BM25 with RRF"""
    # Parallel retrieval
    vector_task = asyncio.create_task(vector_retrieve(query, top_k=10))
    bm25_results = bm25_retrieve(query, top_k=10)
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
    sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)[:top_k]
    
    # Format context
    context = "\n\n".join([
        f"[Document {i+1}] (Page {all_results[doc_id]['page']})\n{all_results[doc_id]['text'][:500]}"
        for i, doc_id in enumerate(sorted_ids) if doc_id in all_results
    ])
    
    return context


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

def init_pinecone_index():
    """Initialize Pinecone index if not exists"""
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"🆕 Creating Pinecone index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pc.Index(INDEX_NAME)
    return index


# Data ingestion with web scraping for DCB Bank
async def ingest_dcb_website():
    """Scrape and ingest DCB Bank website content"""
    import requests
    from bs4 import BeautifulSoup
    import uuid
    
    base_url = "https://www.dcb.bank.in/"
    pages_to_scrape = [
        "",
        "personal-banking",
        "business-banking", 
        "nri-banking",
        "about-us",
        "customer-care"
    ]
    
    all_chunks = []
    
    for page in pages_to_scrape:
        try:
            url = base_url + page
            # FIX: Add redirect control
            response = requests.get(url, timeout=10, allow_redirects=False)
            
            if response.status_code >= 400:
                print(f"⚠️ Skipping {page} (status {response.status_code})")
                continue
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text from paragraphs
            paragraphs = soup.find_all(['p', 'div', 'li'])
            text = " ".join([p.get_text() for p in paragraphs])
            
            # Chunk text
            words = text.split()
            chunk_size = 500
            overlap = 50
            
            for i in range(0, len(words), chunk_size - overlap):
                chunk_text = " ".join(words[i:i+chunk_size])
                if len(chunk_text.strip()) > 100:
                    all_chunks.append({
                        "id": str(uuid.uuid4()),
                        "text": chunk_text.strip(),
                        "page_number": f"web-{page}",
                        "file_name": f"dcb-{page}"
                    })
            
            print(f"✅ Scraped {page}")
            
        except Exception as e:
            print(f"❌ Error scraping {page}: {e}")
    
    # Generate embeddings and store
    embeddings = embedding_model.encode([c['text'] for c in all_chunks])
    
    # FIX: Direct function call (no import needed)
    data_store_vectorstore(all_chunks, embeddings)
    
    # Initialize BM25
    init_bm25(all_chunks)
    
    print(f"✅ Ingested {len(all_chunks)} chunks from DCB website")
    return all_chunks
