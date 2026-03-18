import os
import pickle
from src.retrieval.faiss_store import FAISSStore
from src.retrieval.bm25_store import BM25Store
from src.guardrails.retrieval_guardrails import RetrievalGuardrails
from src.config import settings

class HybridRetriever:
    def __init__(self, embedder):
        self.embedder = embedder
        self.faiss_store = FAISSStore()
        self.bm25_store = BM25Store()
        self.doc_metadata = []
        self.load_metadata()
        
    def load_metadata(self):
        if os.path.exists(settings.DOC_METADATA_PATH):
            with open(settings.DOC_METADATA_PATH, 'rb') as f:
                self.doc_metadata = pickle.load(f)
                
    def retrieve(self, query: str) -> tuple[list[dict], tuple[bool, str]]:
        """
        Retrieves documents using FAISS and BM25, blends results, and applies Retrieval Guardrails.
        """
        # FAISS search
        query_emb = self.embedder.encode([query])
        faiss_dists, faiss_indices = self.faiss_store.search(query_emb)
        
        # BM25 search
        tokenized_query = query.lower().split()
        bm25_scores, bm25_indices = self.bm25_store.search(tokenized_query)
        
        # Combine unique indices (union)
        valid_indices = set(faiss_indices + bm25_indices)
        valid_indices = [i for i in valid_indices if i != -1 and i < len(self.doc_metadata)]
        
        retrieved_docs = [
            {
                "id": i, 
                "text": self.doc_metadata[i]["text"], 
                "source": self.doc_metadata[i].get("source", "unknown")
            } for i in valid_indices
        ]
        
        # Guardrails validation
        best_faiss_dist = min(faiss_dists) if faiss_dists else float('inf')
        is_valid, message = RetrievalGuardrails.validate_retrieval(retrieved_docs, max_score=best_faiss_dist)
        
        if not is_valid:
            return [], (False, message)
            
        return retrieved_docs, (True, "")
