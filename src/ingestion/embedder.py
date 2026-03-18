from sentence_transformers import SentenceTransformer
from src.config import settings

class LocalEmbedder:
    def __init__(self, model_name=settings.EMBEDDING_MODEL_NAME):
        # We load a local embedding model
        # BAAI/bge-small-en is lightweight and highly capable
        self.model = SentenceTransformer(model_name)
        
    def encode(self, texts: list[str]):
        # normalize_embeddings=True is recommended for cosine similarity computations
        return self.model.encode(texts, normalize_embeddings=True)
