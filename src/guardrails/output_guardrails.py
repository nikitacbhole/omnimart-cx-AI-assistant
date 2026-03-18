import re

class OutputGuardrails:
    @staticmethod
    def validate_output(answer: str, retrieved_docs: list) -> tuple[bool, str]:
        """
        Output Validation Guardrail (reject unsupported answers)
        Since we cannot use an LLM for validation, we utilize a deterministic approach:
        1. Numerical hallucination check: Any numbers/prices in the answer MUST exist in the context.
        2. Exact phrase matching could be expanded here.
        """
        context_text = " ".join([doc["text"] for doc in retrieved_docs]).lower()
        
        # Extract all numbers (including decimals) from answer
        numbers_in_answer = re.findall(r"\d+(?:\.\d+)?", answer)
        
        # Safe numbers that models might naturally output that shouldn't trigger hallucinations
        safe_numbers = {"404", "500", "0", "1", "11434"}
        
        for num in numbers_in_answer:
            if num not in safe_numbers and num not in context_text:
                # If there's an Ollama connection error string, bypass the hallucination check entirely
                if "error communicating" in answer.lower() or "localhost" in answer.lower():
                     return False, "Model connection failed."
                     
                return False, f"Output validation failed: Hallucinated number/quantity '{num}' not found in context."
                
        # Optional: Deny list words indication failure from LLM
        failure_phrases = [
            "i don't know", 
            "i am an ai", 
            "unable to provide",
            "cannot answer this based on the provided context",
            "not explicitly contained within the context",
            "cannot provide further details",
            "not explicitly mentioned",
            "does not specify"
        ]
        for phrase in failure_phrases:
            if phrase in answer.lower():
                return False, "Output indicates model failure/refusal."
                
        return True, ""
