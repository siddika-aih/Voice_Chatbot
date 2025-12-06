"""PDF document parser"""
import fitz
import uuid
from typing import List, Dict
from pathlib import Path

class PDFParser:
    """Extract and chunk PDF documents"""
    
    @staticmethod
    def extract_chunks(
        file_path: str,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[Dict]:
        """
        Extract text from PDF and create overlapping chunks
        
        Args:
            file_path: Path to PDF file
            chunk_size: Number of words per chunk
            overlap: Number of overlapping words
            
        Returns:
            List of chunk dictionaries
        """
        doc = fitz.open(file_path)
        fitz.TOOLS.mupdf_display_errors(False)
        
        chunks = []
        file_name = Path(file_path).name
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            words = text.split()
            
            # Create overlapping chunks
            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i:i + chunk_size]
                chunk_text = " ".join(chunk_words).strip()
                
                if len(chunk_text) > 100:  # Filter tiny chunks
                    chunks.append({
                        "id": str(uuid.uuid4()),
                        "text": chunk_text,
                        "page_number": page_num + 1,
                        "source": file_name,
                        "type": "pdf"
                    })
        
        doc.close()
        print(f"✅ Extracted {len(chunks)} chunks from {file_name}")
        return chunks
