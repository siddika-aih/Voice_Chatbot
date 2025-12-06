"""Universal ingestion engine"""
import asyncio
from typing import List, Union
from pathlib import Path

from ingestion.pdf_parser import PDFParser
from ingestion.web_scraper import WebScraper
from retrieval.vector_store import VectorStore
from utils.config import config

class IngestionEngine:
    """Handles all data ingestion"""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.pdf_parser = PDFParser()
        self.web_scraper = WebScraper(
            max_depth=config.MAX_CRAWL_DEPTH,
            max_pages=config.MAX_PAGES
        )
    
    async def ingest(self, sources: Union[str, List[str]]):
        """
        Ingest data from multiple sources
        
        Args:
            sources: List of URLs or file paths
        """
        if isinstance(sources, str):
            sources = [sources]
        
        all_chunks = []
        
        for source in sources:
            if source.startswith(('http://', 'https://')):
                # Web source
                print(f"🌐 Scraping website: {source}")
                chunks = await self.web_scraper.scrape([source])
                all_chunks.extend(chunks)
            elif Path(source).suffix == '.pdf':
                # PDF source
                print(f"📄 Parsing PDF: {source}")
                chunks = self.pdf_parser.extract_chunks(source)
                all_chunks.extend(chunks)
            else:
                print(f"⚠️ Unsupported source type: {source}")
        
        # Store in vector database
        if all_chunks:
            print(f"\n💾 Storing {len(all_chunks)} chunks in vector database...")
            await self.vector_store.store_chunks(all_chunks)
            print("✅ Ingestion complete!")
        else:
            print("⚠️ No chunks to ingest")
