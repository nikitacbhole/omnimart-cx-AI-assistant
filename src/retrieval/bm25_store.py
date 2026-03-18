import os
import pickle
from rank_bm25 import BM25Okapi
from src.config import settings

class BM25Store:
    def __init__(self, index_path=settings.BM25_INDEX_PATH):
        self.index_path = index_path
        self.bm25: BM25Okapi = None
        self.load_index()
        
    def load_index(self):
        if os.path.exists(self.index_path):
            with open(self.index_path, 'rb') as f:
                self.bm25 = pickle.load(f)
                
    def save_index(self):
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.bm25, f)
            
    def fit(self, tokenized_corpus: list[list[str]]):
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.save_index()
        
    def search(self, tokenized_query: list[str], top_k: int = settings.TOP_K_BM25):
        if self.bm25 is None:
            return [], []
        # Get scores
        scores = self.bm25.get_scores(tokenized_query)
        # Sort and get top_k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        top_scores = [scores[i] for i in top_indices]
        return top_scores, top_indices
