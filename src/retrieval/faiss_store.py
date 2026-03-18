import os
import faiss
import numpy as np
from src.config import settings

class FAISSStore:
    def __init__(self, index_path=settings.FAISS_INDEX_PATH):
        self.index_path = index_path
        self.index = None
        self.load_index()
        
    def load_index(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            
    def save_index(self):
        faiss.write_index(self.index, self.index_path)
        
    def add_embeddings(self, embeddings: np.ndarray):
        if self.index is None:
            # Assuming embedding dimension is derived from vector
            d = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(d)
        self.index.add(embeddings)
        self.save_index()
        
    def search(self, query_embedding: np.ndarray, top_k: int = settings.TOP_K_FAISS):
        if self.index is None or self.index.ntotal == 0:
            return [], []
        # Support batch search but assuming query_embedding is (1, d)
        distances, indices = self.index.search(query_embedding, top_k)
        return distances[0].tolist(), indices[0].tolist()
