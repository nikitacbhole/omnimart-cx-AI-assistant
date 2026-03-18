import time
import logging
import sys
import os

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from src.config import settings
from src.guardrails.input_guardrails import InputGuardrails
from src.guardrails.retrieval_guardrails import RetrievalGuardrails
from src.guardrails.output_guardrails import OutputGuardrails
from src.retrieval.hybrid_retriever import HybridRetriever
from src.ingestion.embedder import LocalEmbedder
from src.llm.phi3_client import Phi3Client

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Omnimart CX Assistant API", version="1.0")

# Initialize global components (Lazily or at startup, here at startup for simplicity)
try:
    logger.info("Initializing Embedder...")
    embedder = LocalEmbedder()
    logger.info("Initializing Guardrails...")
    input_guardrails = InputGuardrails(embedder=embedder)
    logger.info("Initializing Retriever...")
    retriever = HybridRetriever(embedder=embedder)
    logger.info("Initializing LLM Client...")
    llm_client = Phi3Client()
    logger.info("All components initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing components: {e}")
    # Components might not load if indices do not exist yet

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    confidence: Optional[float] = None
    status: str
    latency_ms: float

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    start_time = time.time()
    query = request.query
    logger.info(f"Processing query: {query}")
    
    # 1. Input Guardrails
    is_safe_injection, reason_inj = input_guardrails.check_prompt_injection(query)
    if not is_safe_injection:
        logger.warning(f"Blocked by Input Guardrail (Injection): {reason_inj}")
        return QueryResponse(
            answer="Blocked due to security policy.",
            sources=[],
            status="blocked",
            latency_ms=(time.time() - start_time) * 1000
        )
        
    is_domain_match, reason_dom = input_guardrails.check_domain_similarity(query)
    if not is_domain_match:
        logger.warning(f"Blocked by Input Guardrail (Domain): {reason_dom}")
        return QueryResponse(
            answer="I am a company policy and product assistant. I cannot answer queries outside this domain.",
            sources=[],
            status="blocked",
            latency_ms=(time.time() - start_time) * 1000
        )
        
    # 2. Hybrid Retrieval
    docs, (retrieval_valid, reason_ret) = retriever.retrieve(query)
    if not retrieval_valid:
        logger.warning(f"Blocked by Retrieval Guardrail: {reason_ret}")
        return QueryResponse(
            answer="I'm sorry, I couldn't find enough relevant information in our knowledge base to answer that.",
            sources=[],
            status="fallback",
            latency_ms=(time.time() - start_time) * 1000
        )
        
    # 3. LLM Generation
    context_str = "\n".join([f"- {d['text']}" for d in docs])
    llm_start = time.time()
    raw_answer = llm_client.generate_response(query, context_str)
    llm_latency = time.time() - llm_start
    logger.info(f"LLM Generation took {llm_latency:.2f}s")
    logger.info(f"RAW LLM ANSWER: {raw_answer}")
    
    # 4. Output Guardrails
    is_output_valid, reason_out = OutputGuardrails.validate_output(raw_answer, docs)
    if not is_output_valid:
        logger.warning(f"Blocked by Output Guardrail: {reason_out}")
        return QueryResponse(
            answer="I'm sorry, I could not generate a reliable answer based on the available information.",
            sources=docs,
            status="blocked",
            latency_ms=(time.time() - start_time) * 1000
        )
        
    # 5. Return Success
    return QueryResponse(
        answer=raw_answer,
        sources=docs,
        status="success",
        latency_ms=(time.time() - start_time) * 1000
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
