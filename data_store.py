
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

# # enhanced_data_store.py - Hybrid search with BM25 + Vector
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from pinecone import Pinecone, ServerlessSpec
# from rank_bm25 import BM25Okapi
# import asyncio
# from typing import List, Dict
# import os
# from dotenv import load_dotenv

# load_dotenv()

# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# embedding_model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)
# INDEX_NAME = "rag-pdf-index"

# # Global BM25 index (load from stored chunks)
# bm25_index = None
# bm25_corpus = []
# bm25_metadata = []

# def init_bm25(chunks: List[Dict]):
#     """Initialize BM25 from chunk corpus"""
#     global bm25_index, bm25_corpus, bm25_metadata
    
#     bm25_corpus = [chunk['text'].lower().split() for chunk in chunks]
#     bm25_metadata = chunks
#     bm25_index = BM25Okapi(bm25_corpus)
#     print(f"✅ BM25 initialized with {len(chunks)} documents")

# async def vector_retrieve(query: str, top_k: int = 10) -> List[Dict]:
#     """Async vector search in Pinecone"""
#     index = pc.Index(INDEX_NAME)
#     query_embedding = embedding_model.encode([query])[0]
    
#     results = index.query(
#         vector=query_embedding.tolist(),
#         top_k=top_k,
#         include_metadata=True
#     )
    
#     return [{
#         "id": match['id'],
#         "score": match['score'],
#         "text": match['metadata']['text'],
#         "page": match['metadata'].get('page_number', 'N/A'),
#         "source": "vector"
#     } for match in results['matches']]

# def bm25_retrieve(query: str, top_k: int = 10) -> List[Dict]:
#     """BM25 keyword search"""
#     if bm25_index is None:
#         return []
    
#     tokenized_query = query.lower().split()
#     scores = bm25_index.get_scores(tokenized_query)
    
#     # Get top_k indices
#     top_indices = np.argsort(scores)[-top_k:][::-1]
    
#     return [{
#         "id": bm25_metadata[i]['id'],
#         "score": float(scores[i]),
#         "text": bm25_metadata[i]['text'],
#         "page": bm25_metadata[i].get('page_number', 'N/A'),
#         "source": "bm25"
#     } for i in top_indices if scores[i] > 0]

# async def hybrid_retrieve(query: str, top_k: int = 5, alpha: float = 0.5) -> str:
#     """Hybrid search combining vector + BM25 with RRF"""
#     # Parallel retrieval
#     vector_task = asyncio.create_task(vector_retrieve(query, top_k=10))
#     bm25_results = bm25_retrieve(query, top_k=10)
#     vector_results = await vector_task
    
#     # Reciprocal Rank Fusion (RRF)
#     k = 60  # RRF constant
#     fused_scores = {}
    
#     for rank, result in enumerate(vector_results):
#         doc_id = result['id']
#         fused_scores[doc_id] = fused_scores.get(doc_id, 0) + alpha / (k + rank + 1)
    
#     for rank, result in enumerate(bm25_results):
#         doc_id = result['id']
#         fused_scores[doc_id] = fused_scores.get(doc_id, 0) + (1 - alpha) / (k + rank + 1)
    
#     # Get top documents
#     all_results = {r['id']: r for r in vector_results + bm25_results}
#     sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)[:top_k]
    
#     # Format context
#     context = "\n\n".join([
#         f"[Document {i+1}] (Page {all_results[doc_id]['page']})\n{all_results[doc_id]['text'][:500]}"
#         for i, doc_id in enumerate(sorted_ids) if doc_id in all_results
#     ])
    
#     return context


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

# def init_pinecone_index():
#     """Initialize Pinecone index if not exists"""
#     if INDEX_NAME not in pc.list_indexes().names():
#         print(f"🆕 Creating Pinecone index: {INDEX_NAME}")
#         pc.create_index(
#             name=INDEX_NAME,
#             dimension=1024,
#             metric="cosine",
#             spec=ServerlessSpec(cloud="aws", region="us-east-1")
#         )
#     index = pc.Index(INDEX_NAME)
#     return index


# # Data ingestion with web scraping for DCB Bank
# async def ingest_dcb_website():
#     """Scrape and ingest DCB Bank website content"""
#     import requests
#     from bs4 import BeautifulSoup
#     import uuid
    
#     base_url = "https://www.dcb.bank.in/"
#     pages_to_scrape = [
#         "",
#         "personal-banking",
#         "business-banking", 
#         "nri-banking",
#         "about-us",
#         "customer-care"
#     ]
    
#     all_chunks = []
    
#     for page in pages_to_scrape:
#         try:
#             url = base_url + page
#             # FIX: Add redirect control
#             response = requests.get(url, timeout=10, allow_redirects=False)
            
#             if response.status_code >= 400:
#                 print(f"⚠️ Skipping {page} (status {response.status_code})")
#                 continue
            
#             soup = BeautifulSoup(response.content, 'html.parser')
            
#             # Extract text from paragraphs
#             paragraphs = soup.find_all(['p', 'div', 'li'])
#             text = " ".join([p.get_text() for p in paragraphs])
            
#             # Chunk text
#             words = text.split()
#             chunk_size = 500
#             overlap = 50
            
#             for i in range(0, len(words), chunk_size - overlap):
#                 chunk_text = " ".join(words[i:i+chunk_size])
#                 if len(chunk_text.strip()) > 100:
#                     all_chunks.append({
#                         "id": str(uuid.uuid4()),
#                         "text": chunk_text.strip(),
#                         "page_number": f"web-{page}",
#                         "file_name": f"dcb-{page}"
#                     })
            
#             print(f"✅ Scraped {page}")
            
#         except Exception as e:
#             print(f"❌ Error scraping {page}: {e}")
    
#     # Generate embeddings and store
#     embeddings = embedding_model.encode([c['text'] for c in all_chunks])
    
#     # FIX: Direct function call (no import needed)
#     data_store_vectorstore(all_chunks, embeddings)
    
#     # Initialize BM25
#     init_bm25(all_chunks)
    
#     print(f"✅ Ingested {len(all_chunks)} chunks from DCB website")
#     return all_chunks

# universal_ingestion.py - Dynamic multi-source ingestion
import os
import uuid
import asyncio
import aiohttp
import requests
from typing import List, Dict, Union
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from rank_bm25 import BM25Okapi
import numpy as np
from dotenv import load_dotenv

# Document parsers
import fitz  # PyMuPDF for PDFs
# from docx import Document  # python-docx for Word
import mimetypes

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
embedding_model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)
INDEX_NAME = "rag-universal-index"
DIMENSION = 1024

# Global BM25 state
bm25_index = None
bm25_corpus = []
bm25_metadata = []


class UniversalIngestionEngine:
    """Handles websites, PDFs, DOCX, TXT with deep crawling"""
    
    def __init__(self, max_depth=3, max_pages=100):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.visited_urls = set()
        self.index = self._init_pinecone()
        
    def _init_pinecone(self):
        """Initialize Pinecone index"""
        if INDEX_NAME not in pc.list_indexes().names():
            print(f"🆕 Creating Pinecone index: {INDEX_NAME}")
            pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        return pc.Index(INDEX_NAME)
    
    async def ingest(self, sources: Union[str, List[str]]) -> Dict:
        """
        Universal ingestion from multiple sources
        
        Args:
            sources: Single source or list of sources (URLs, file paths, or mixed)
        
        Returns:
            Dict with ingestion statistics
        """
        if isinstance(sources, str):
            sources = [sources]
        
        all_chunks = []
        stats = {
            "total_sources": len(sources),
            "websites": 0,
            "pdfs": 0,
            "docx": 0,
            "txt": 0,
            "total_chunks": 0,
            "errors": []
        }
        
        for source in sources:
            try:
                print(f"\n📥 Processing: {source}")
                
                if self._is_url(source):
                    # Website crawling
                    chunks = await self._crawl_website(source)
                    stats["websites"] += 1
                elif source.lower().endswith('.pdf'):
                    # PDF parsing
                    chunks = self._parse_pdf(source)
                    stats["pdfs"] += 1
                elif source.lower().endswith(('.docx', '.doc')):
                    # Word document parsing
                    chunks = self._parse_docx(source)
                    stats["docx"] += 1
                elif source.lower().endswith('.txt'):
                    # Text file parsing
                    chunks = self._parse_txt(source)
                    stats["txt"] += 1
                else:
                    # Auto-detect file type
                    chunks = self._parse_auto(source)
                    stats["txt"] += 1
                
                all_chunks.extend(chunks)
                print(f"✅ Extracted {len(chunks)} chunks from {source}")
                
            except Exception as e:
                error_msg = f"Error processing {source}: {str(e)}"
                print(f"❌ {error_msg}")
                stats["errors"].append(error_msg)
        
        # Store in Pinecone and initialize BM25
        if all_chunks:
            await self._store_chunks(all_chunks)
            self._init_bm25(all_chunks)
            stats["total_chunks"] = len(all_chunks)
        
        return stats
    
    def _is_url(self, source: str) -> bool:
        """Check if source is a URL"""
        return source.startswith(('http://', 'https://', 'www.'))
    
    async def _crawl_website(self, base_url: str, depth=0) -> List[Dict]:
        """Deep crawl website with BFS"""
        if depth >= self.max_depth or len(self.visited_urls) >= self.max_pages:
            return []
        
        if base_url in self.visited_urls:
            return []
        
        self.visited_urls.add(base_url)
        all_chunks = []
        
        try:
            # Fetch page
            response = requests.get(
                base_url, 
                timeout=10, 
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text(separator=' ', strip=True)
            
            # Get title
            title = soup.title.string if soup.title else base_url
            
            # Chunk the content
            chunks = self._chunk_text(
                text,
                metadata={
                    "source": base_url,
                    "title": title,
                    "type": "website"
                }
            )
            all_chunks.extend(chunks)
            
            print(f"  📄 Crawled: {base_url} ({len(chunks)} chunks, depth {depth})")
            
            # Find and crawl links (if not at max depth)
            if depth < self.max_depth - 1:
                links = soup.find_all('a', href=True)
                parsed_base = urlparse(base_url)
                
                for link in links[:20]:  # Limit links per page
                    href = link['href']
                    full_url = urljoin(base_url, href)
                    parsed_url = urlparse(full_url)
                    
                    # Only crawl same domain
                    if parsed_url.netloc == parsed_base.netloc:
                        if full_url not in self.visited_urls:
                            sub_chunks = await self._crawl_website(full_url, depth + 1)
                            all_chunks.extend(sub_chunks)
                            
                            if len(self.visited_urls) >= self.max_pages:
                                break
        
        except Exception as e:
            print(f"  ❌ Crawl error: {e}")
        
        return all_chunks
    
    def _parse_pdf(self, file_path: str) -> List[Dict]:
        """Parse PDF file"""
        chunks = []
        
        try:
            doc = fitz.open(file_path)
            fitz.TOOLS.mupdf_display_errors(False)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text")
                
                if text.strip():
                    page_chunks = self._chunk_text(
                        text,
                        metadata={
                            "source": file_path,
                            "page": page_num + 1,
                            "type": "pdf",
                            "filename": Path(file_path).name
                        }
                    )
                    chunks.extend(page_chunks)
            
            doc.close()
            
        except Exception as e:
            print(f"  ❌ PDF parse error: {e}")
        
        return chunks
    
    # def _parse_docx(self, file_path: str) -> List[Dict]:
    #     """Parse Word document"""
    #     chunks = []
        
    #     try:
    #         doc = Document(file_path)
    #         text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            
    #         chunks = self._chunk_text(
    #             text,
    #             metadata={
    #                 "source": file_path,
    #                 "type": "docx",
    #                 "filename": Path(file_path).name
    #             }
    #         )
            
    #     except Exception as e:
    #         print(f"  ❌ DOCX parse error: {e}")
        
    #     return chunks
    
    def _parse_txt(self, file_path: str) -> List[Dict]:
        """Parse text file"""
        chunks = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            chunks = self._chunk_text(
                text,
                metadata={
                    "source": file_path,
                    "type": "txt",
                    "filename": Path(file_path).name
                }
            )
            
        except Exception as e:
            print(f"  ❌ TXT parse error: {e}")
        
        return chunks
    
    def _parse_auto(self, file_path: str) -> List[Dict]:
        """Auto-detect and parse file"""
        mime_type, _ = mimetypes.guess_type(file_path)
        
        if mime_type == 'application/pdf':
            return self._parse_pdf(file_path)
        elif mime_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
            return self._parse_docx(file_path)
        else:
            return self._parse_txt(file_path)
    
    def _chunk_text(self, text: str, metadata: Dict, chunk_size=500, overlap=50) -> List[Dict]:
        """Smart text chunking with overlap"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words).strip()
            
            if len(chunk_text) > 100:  # Skip tiny chunks
                chunk_data = {
                    "id": str(uuid.uuid4()),
                    "text": chunk_text,
                    **metadata,
                    "chunk_index": i // (chunk_size - overlap)
                }
                chunks.append(chunk_data)
        
        return chunks
    
    # async def _store_chunks(self, chunks: List[Dict]):
    #     """Store chunks in Pinecone with embeddings"""
    #     print(f"\n💾 Generating embeddings for {len(chunks)} chunks...")
        
    #     # Generate embeddings in batches
    #     texts = [chunk['text'] for chunk in chunks]
    #     embeddings = embedding_model.encode(texts, show_progress_bar=True)
        
    #     # Prepare vectors for Pinecone
    #     vectors = []
    #     for chunk, embedding in zip(chunks, embeddings):
    #         vectors.append({
    #             "id": chunk["id"],
    #             "values": embedding.tolist(),
    #             "metadata": {
    #                 "text": chunk["text"][:1000],  # Pinecone metadata limit
    #                 "source": str(chunk.get("source", "unknown")),
    #                 "type": chunk.get("type", "unknown"),
    #                 "page": str(chunk.get("page", "N/A")),
    #                 "title": chunk.get("title", "")[:100]
    #             }
    #         })
        
    #     # Batch upsert to Pinecone
    #     batch_size = 100
    #     for i in range(0, len(vectors), batch_size):
    #         batch = vectors[i:i + batch_size]
    #         self.index.upsert(vectors=batch)
    #         print(f"  📤 Uploaded batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
        
    #     print(f"✅ Stored {len(vectors)} vectors in Pinecone!")


    async def _store_chunks(self, chunks: List[Dict]):
        """Store text chunks into Pinecone (simple, fast, clean)."""

        if not chunks:
            print("⚠ No chunks provided. Skipping Pinecone storage.")
            return

        print(f"\n💾 Creating embeddings for {len(chunks)} chunks...")

        # 1. Generate embeddings
        texts = [c["text"] for c in chunks]
        embeddings = embedding_model.encode(texts)

        # 2. Build vector payloads
        vectors = [
            {
                "id": c["id"],
                "values": emb.tolist(),
                "metadata": {
                    "text": c["text"][:1000],
                    "source": c.get("source", "unknown"),
                    "type": c.get("type", "unknown"),
                    "page": str(c.get("page", "N/A")),
                    "title": c.get("title", "")[:100],
                    "url": c.get("url", "")
                }
            }
            for c, emb in zip(chunks, embeddings)
        ]

        # 3. Batch upload
        batch_size = 100
        total_batches = (len(vectors) + batch_size - 1) // batch_size

        print(f"📤 Uploading {len(vectors)} vectors in {total_batches} batches...")

        for batch_idx in range(total_batches):
            batch = vectors[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            self.index.upsert(vectors=batch)
            print(f"  ✔ Batch {batch_idx + 1}/{total_batches} uploaded")

        print(f"✅ Successfully stored {len(vectors)} vectors in Pinecone!")

    
    def _init_bm25(self, chunks: List[Dict]):
        """Initialize BM25 index"""
        global bm25_index, bm25_corpus, bm25_metadata
        
        bm25_corpus = [chunk['text'].lower().split() for chunk in chunks]
        bm25_metadata = chunks
        bm25_index = BM25Okapi(bm25_corpus)
        
        print(f"✅ BM25 initialized with {len(chunks)} documents")

# Retrieval functions
# async def hybrid_retrieve(query: str, top_k: int = 5, alpha: float = 0.5) -> str:
#     """Hybrid retrieval with Pinecone + BM25"""
#     index = pc.Index(INDEX_NAME)
    
#     # Vector search
#     query_embedding = embedding_model.encode([query])[0]
#     vector_results = index.query(
#         vector=query_embedding.tolist(),
#         top_k=10,
#         include_metadata=True
#     )
    
#     vector_docs = [{
#         "id": match['id'],
#         "score": match['score'],
#         "text": match['metadata']['text'],
#         "source": match['metadata'].get('source', 'unknown')
#     } for match in vector_results['matches']]
    
#     # BM25 search
#     bm25_docs = []
#     if bm25_index is not None:
#         tokenized_query = query.lower().split()
#         scores = bm25_index.get_scores(tokenized_query)
#         top_indices = np.argsort(scores)[-10:][::-1]
        
#         bm25_docs = [{
#             "id": bm25_metadata[i]['id'],
#             "score": float(scores[i]),
#             "text": bm25_metadata[i]['text'],
#             "source": bm25_metadata[i].get('source', 'unknown')
#         } for i in top_indices if scores[i] > 0]
    
#     # Reciprocal Rank Fusion
#     k = 60
#     fused_scores = {}
    
#     for rank, doc in enumerate(vector_docs):
#         doc_id = doc['id']
#         fused_scores[doc_id] = fused_scores.get(doc_id, 0) + alpha / (k + rank + 1)
    
#     for rank, doc in enumerate(bm25_docs):
#         doc_id = doc['id']
#         fused_scores[doc_id] = fused_scores.get(doc_id, 0) + (1 - alpha) / (k + rank + 1)
    
#     # Get top documents
#     all_docs = {doc['id']: doc for doc in vector_docs + bm25_docs}
#     sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)[:top_k]
    
#     # Format context
#     context = "\n\n".join([
#         f"[Source: {all_docs[doc_id]['source']}]\n{all_docs[doc_id]['text'][:600]}"
#         for doc_id in sorted_ids if doc_id in all_docs
#     ])
    
#     return context



# Retrieval functions
async def retrieve(query: str, top_k: int = 5) -> str:
    """Pure Pinecone vector retrieval (no BM25/sparse)"""
    index = pc.Index(INDEX_NAME)
    
    # Generate dense embedding only
    query_embedding = embedding_model.encode([query])[0]
    
    # Pure vector search - no sparse_vector or alpha
    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )
    
    # Format context
    context = "\n\n".join([
        f"[Source: {match['metadata'].get('source', 'unknown')}]\n{match['metadata']['text'][:600]}"
        for match in results['matches']
    ])
    
    return context

# async def hybrid_retrieve(query: str, top_k: int = 5) -> str:
#     """Alias for retrieve - pure vector search"""
#     return await retrieve(query, top_k)

    # # Upsert to Pinecone index
    # index.upsert(vectors=to_upsert)
    # print(f"Stored {len(to_upsert)} vectors to Pinecone index {INDEX_NAME}")

# CLI Interface
async def main():
    """Example usage"""
    engine = UniversalIngestionEngine(max_depth=2, max_pages=50)
    
    # Dynamic sources - mix of everything!
    sources = [
        "https://www.dcbbank.com",
        # r"C:\\Users\\admin\\Downloads\\Medical_book.pdf" # Website (deep crawl)
        # Text file
    ]
    
    print("🚀 Starting universal ingestion...")
    stats = await engine.ingest(sources)
    
    print("\n" + "="*50)
    print("📊 INGESTION SUMMARY")
    print("="*50)
    print(f"✅ Websites crawled: {stats['websites']}")
    print(f"✅ PDFs processed: {stats['pdfs']}")
    print(f"✅ DOCX processed: {stats['docx']}")
    print(f"✅ TXT processed: {stats['txt']}")
    print(f"✅ Total chunks: {stats['total_chunks']}")
    
    if stats['errors']:
        print(f"\n⚠️ Errors: {len(stats['errors'])}")
        for error in stats['errors']:
            print(f"  - {error}")
    
    # Test retrieval
    print("\n🔍 Testing retrieval...")
    context = await retrieve("What are the loan options?", top_k=3)
    print(context)


if __name__ == "__main__":
    asyncio.run(main())
