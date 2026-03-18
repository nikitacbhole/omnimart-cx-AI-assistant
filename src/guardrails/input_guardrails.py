import re
import numpy as np
from src.config import settings

MALICIOUS_PATTERNS = [
    r"(?i)(ignore .* instructions)",
    r"(?i)(system prompt)",
    r"(?i)(you are an unconstrained)",
    r"(?i)(bypassing restrictions)",
    r"(?i)(forget everything)",
    r"(?i)(disregard previous)",
]

# Topic anchors for semantic gatekeeper (to prevent general knowledge queries)
DOMAIN_TOPICS = [
    "company policies, return policy, privacy policy, terms of service",
    "product catalog, item details, specifications, electronics, clothing",
    "shipping information, delivery times, tracking orders",
    "pricing, discounts, promotions, sales",
    "warranty, repairs, replacements",
    "customer support, contact information, store locations"
]

class InputGuardrails:
    def __init__(self, embedder):
        self.embedder = embedder
        # Precompute embeddings for the domain topics
        self.domain_embeddings = self.embedder.encode(DOMAIN_TOPICS)
        
    def check_prompt_injection(self, query: str) -> tuple[bool, str]:
        """Returns (is_safe, reason)."""
        for pattern in MALICIOUS_PATTERNS:
            if re.search(pattern, query):
                return False, "Prompt injection detected."
        return True, ""
        
    def check_domain_similarity(self, query: str) -> tuple[bool, str]:
        """Returns (is_safe, reason)."""
        query_emb = self.embedder.encode([query])[0]
        
        # Calculate cosine similarity with all domain topics
        # Assuming embeddings are normalized, otherwise compute full cosine sim
        similarities = np.dot(self.domain_embeddings, query_emb) / (
            np.linalg.norm(self.domain_embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        max_sim = np.max(similarities)
        
        if max_sim < settings.DOMAIN_SIMILARITY_THRESHOLD:
            return False, f"Query out of domain. Similarity: {max_sim:.2f} < {settings.DOMAIN_SIMILARITY_THRESHOLD}"
            
        return True, ""
