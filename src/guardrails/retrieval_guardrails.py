from src.config import settings

class RetrievalGuardrails:
    @staticmethod
    def validate_retrieval(docs: list, max_score: float = None) -> tuple[bool, str]:
        """
        Retrieval Validation Guardrail
        Validates the quantity and quality of retrieved documents.
        """
        if not docs:
            return False, "No documents retrieved for the query."
            
        if len(docs) < settings.MIN_RETRIEVED_DOCS:
            return False, f"Insufficient context: found {len(docs)} docs, required {settings.MIN_RETRIEVED_DOCS}."
            
        # For FAISS L2 distance, a lower value is better (0 is exact match).
        # We reject if the best distance is too HIGH (> 1.2).
        if max_score is not None and max_score > 1.2:
            return False, f"Top retrieval distance ({max_score:.2f}) is too high (not relevant enough)."
            
        return True, ""
