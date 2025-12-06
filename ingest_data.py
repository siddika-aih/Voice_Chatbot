"""Data ingestion script"""
import asyncio
from ingestion.ingestion_engine import IngestionEngine

async def main():
    """Ingest DCB Bank knowledge base"""
    engine = IngestionEngine()
    
    sources = [
        # DCB Bank website
        "https://www.dcb.bank.in/",
        
        # Add your PDF files
        # "data/dcb_products.pdf",
        # "data/dcb_policies.pdf",
    ]
    
    print("🚀 Starting data ingestion...")
    await engine.ingest(sources)
    print("\n✅ Ingestion complete! Ready to run bot.")

if __name__ == "__main__":
    asyncio.run(main())
