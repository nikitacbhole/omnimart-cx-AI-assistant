import os
from pydantic import BaseModel

class Config(BaseModel):
    # Ingestion Configuration
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 80
    
    # Models Configuration
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en"
    LLM_MODEL_NAME: str = "phi3:mini"
    
    # Retrieval Configuration
    FAISS_INDEX_PATH: str = "data/indices/faiss_index.bin"
    BM25_INDEX_PATH: str = "data/indices/bm25_index.pkl"
    DOC_METADATA_PATH: str = "data/indices/doc_metadata.pkl"
    TOP_K_FAISS: int = 5
    TOP_K_BM25: int = 5
    
    # Guardrails Configuration
    DOMAIN_SIMILARITY_THRESHOLD: float = 0.65
    MIN_RETRIEVED_DOCS: int = 1
    
    # LLM Parameters
    LLM_TEMPERATURE: float = 0.1
    # Only using temperature per user request
    LLM_TOP_K: int = 40
    LLM_MAX_NEW_TOKENS: int = 512
    OLLAMA_BASE_URL: str = "http://localhost:11434/api/generate"

settings = Config()

# Ensure required directories exist
os.makedirs(os.path.dirname(settings.FAISS_INDEX_PATH), exist_ok=True)
os.makedirs("data/docs", exist_ok=True)
