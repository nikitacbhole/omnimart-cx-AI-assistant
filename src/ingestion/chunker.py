import os
import glob
from src.config import settings

class DocumentChunker:
    def __init__(self, chunk_size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def chunk_text(self, text: str) -> list[str]:
        """Simple word-based chunking."""
        words = text.split()
        chunks = []
        if not words:
            return chunks
            
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            i += self.chunk_size - self.overlap
            
        return chunks
        
    def process_directory(self, dir_path: str) -> list[dict]:
        """Reads all txt files and chunks them."""
        documents = []
        files = glob.glob(os.path.join(dir_path, "*.txt"))
        
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                chunks = self.chunk_text(content)
                for chunk in chunks:
                    documents.append({
                        "text": chunk,
                        "source": os.path.basename(file_path)
                    })
        return documents
