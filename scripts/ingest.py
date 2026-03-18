import os
import pickle
import sys

# Add parent directory to path to allow importing src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.chunker import DocumentChunker
from src.ingestion.embedder import LocalEmbedder
from src.retrieval.faiss_store import FAISSStore
from src.retrieval.bm25_store import BM25Store
from src.config import settings

def main():
    print("Starting data ingestion pipeline...")
    
    docs_dir = "data/docs"
    os.makedirs(docs_dir, exist_ok=True)
    
    # Check if there are documents
    import glob
    if not glob.glob(os.path.join(docs_dir, "*.txt")):
        print("No documents found in data/docs. Creating a sample doc...")
        with open(os.path.join(docs_dir, "sample_policy.txt"), "w", encoding="utf-8") as f:
            f.write("Return Policy: You can return items within 30 days of purchase. "
                    "Shipping is free for standard delivery taking 5-7 business days. "
                    "Electronics have a 1-year limited warranty.")
    
    chunker = DocumentChunker()
    print(f"Chunking documents from {docs_dir}...")
    chunks = chunker.process_directory(docs_dir)
    
    if not chunks:
        print("No chunks created. Exiting.")
        return
        
    print(f"Created {len(chunks)} chunks. Generating embeddings...")
    embedder = LocalEmbedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts)
    
    print("Building FAISS index...")
    faiss_store = FAISSStore()
    faiss_store.add_embeddings(embeddings)
    
    print("Building BM25 index...")
    tokenized_corpus = [text.lower().split() for text in texts]
    bm25_store = BM25Store()
    bm25_store.fit(tokenized_corpus)
    
    print("Saving document metadata...")
    os.makedirs(os.path.dirname(settings.DOC_METADATA_PATH), exist_ok=True)
    with open(settings.DOC_METADATA_PATH, 'wb') as f:
        pickle.dump(chunks, f)
        
    print("Ingestion complete!")

if __name__ == "__main__":
    main()
