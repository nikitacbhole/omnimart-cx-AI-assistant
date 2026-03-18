import requests
from src.config import settings

class Phi3Client:
    def __init__(self, base_url=settings.OLLAMA_BASE_URL, model_name=settings.LLM_MODEL_NAME):
        self.base_url = base_url
        self.model_name = model_name
        self.system_prompt = (
            "You are a strict company policy and product information assistant. "
            "You must answer the user's query ONLY using the provided context. "
            "If the answer is not contained in the context, you must reply with "
            "'I cannot answer this based on the provided context.' "
            "Do not hallucinate or include any external knowledge. "
            "Do not provide fabricated numbers or URLs."
        )

    def generate_response(self, query: str, context: str) -> str:
        prompt = f"""<|system|>
{self.system_prompt}<|end|>
<|user|>
Context Information:
{context}

Question: {query}

If the answer is not explicitly contained within the Context Information above, you MUST respond only with: "I cannot answer this based on the provided context." Do not guess, do not provide general knowledge, do not provide definitions of any objects, terms, or concepts, and do not explain what things are unless their definitions are explicitly stated in the context.<|end|>
<|assistant|>"""
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": settings.LLM_TEMPERATURE,
                "top_k": settings.LLM_TOP_K,
                "num_predict": settings.LLM_MAX_NEW_TOKENS
            }
        }
        
        try:
            response = requests.post(self.base_url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except Exception as e:
            return f"Error communicating with local LLM: {str(e)}"
