import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.embedder import LocalEmbedder
from src.guardrails.input_guardrails import InputGuardrails
from src.guardrails.retrieval_guardrails import RetrievalGuardrails
from src.guardrails.output_guardrails import OutputGuardrails
from src.retrieval.hybrid_retriever import HybridRetriever
from src.llm.phi3_client import Phi3Client

def run_evaluation():
    print("Initializing Evaluation Pipeline...")
    try:
        embedder = LocalEmbedder()
        input_guardrails = InputGuardrails(embedder=embedder)
        retriever = HybridRetriever(embedder=embedder)
        llm_client = Phi3Client()
    except Exception as e:
        print(f"Error initializing components: {e}")
        print("Please ensure you have run `python scripts/ingest.py` first to create the indices.")
        return
    
    dataset_path = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
        
    metrics = {
        "gatekeeper_correct": 0,
        "gatekeeper_total_evals": 0,
        "queries_blocked_before_llm": 0,
        "total_queries": len(dataset),
        "retrieval_success": 0,
        "retrieval_total": 0,
        "answers_generated": 0,
        "hallucinations": 0,
        "total_latency": 0.0,
        "total_llm_latency": 0.0,
        "keyword_matches": 0, # Used for pseudo Answer Accuracy
        "evals_with_keywords": 0
    }
    
    print(f"Evaluating {metrics['total_queries']} queries...\n")
    import time
    
    for i, data in enumerate(dataset):
        query = data["query"]
        expected_type = data["expected_type"]
        expected_keywords = data.get("expected_keywords", [])
        
        print(f"[{i+1}/{metrics['total_queries']}] Query: '{query}' ({expected_type})")
        
        start_time = time.time()
        
        # 1. Gatekeeper / Input Guardrails
        is_safe_injection, _ = input_guardrails.check_prompt_injection(query)
        is_domain_match, _ = input_guardrails.check_domain_similarity(query)
        
        passed_gatekeeper = is_safe_injection and is_domain_match
        
        # Check gatekeeper accuracy
        should_pass = expected_type in ["in_scope", "edge_case"]
        if passed_gatekeeper == should_pass:
            metrics["gatekeeper_correct"] += 1
        metrics["gatekeeper_total_evals"] += 1
        
        if not passed_gatekeeper:
            metrics["queries_blocked_before_llm"] += 1
            metrics["total_latency"] += (time.time() - start_time)
            print("  -> Blocked by Gatekeeper.")
            continue
            
        # 2. Retrieval
        docs, (retrieval_valid, _) = retriever.retrieve(query)
        metrics["retrieval_total"] += 1
        
        # Recall Check (Are the expected keywords in the retrieved context?)
        if retrieval_valid:
            context_text = " ".join([d['text'] for d in docs]).lower()
            if expected_keywords:
                matches = sum(1 for kw in expected_keywords if kw.lower() in context_text)
                if matches == len(expected_keywords):
                    metrics["retrieval_success"] += 1
            else:
                metrics["retrieval_success"] += 1
        else:
            metrics["queries_blocked_before_llm"] += 1
            metrics["total_latency"] += (time.time() - start_time)
            print("  -> Blocked by Retrieval Guardrail (No context).")
            continue
            
        # 3. LLM Generation
        context_str = "\n".join([f"- {d['text']}" for d in docs])
        llm_start = time.time()
        raw_answer = llm_client.generate_response(query, context_str)
        metrics["total_llm_latency"] += (time.time() - llm_start)
        metrics["answers_generated"] += 1
        
        # Answer Accuracy (pseudo-eval via keyword presence in answer)
        if expected_keywords:
            metrics["evals_with_keywords"] += 1
            if all(kw.lower() in raw_answer.lower() for kw in expected_keywords):
                metrics["keyword_matches"] += 1
        
        # 4. Output validation
        is_output_valid, _ = OutputGuardrails.validate_output(raw_answer, docs)
        if not is_output_valid:
            metrics["hallucinations"] += 1
            print(f"  -> Hallucination detected! Answer validation failed.")
        else:
            print(f"  -> Answered securely.")
            
        metrics["total_latency"] += (time.time() - start_time)
            
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    
    gatekeeper_acc = (metrics["gatekeeper_correct"] / max(metrics["gatekeeper_total_evals"], 1)) * 100
    retrieval_acc = (metrics["retrieval_success"] / max(metrics["retrieval_total"], 1)) * 100
    llm_reduction = (metrics["queries_blocked_before_llm"] / metrics["total_queries"]) * 100
    hallucination_rate = (metrics["hallucinations"] / max(metrics["answers_generated"], 1)) * 100
    grounding_score = 100 - hallucination_rate
    answer_acc = (metrics["keyword_matches"] / max(metrics["evals_with_keywords"], 1)) * 100 if metrics["evals_with_keywords"] > 0 else 100.0
    
    avg_latency = metrics["total_latency"] / metrics["total_queries"]
    avg_llm_latency = metrics["total_llm_latency"] / max(metrics["answers_generated"], 1)
    
    print(f"Gatekeeper Accuracy:     {gatekeeper_acc:.1f}%")
    print(f"Retrieval Recall@5:      {retrieval_acc:.1f}%")
    print(f"Answer Accuracy:         {answer_acc:.1f}%")
    print(f"Answer Grounding Score:  {grounding_score:.1f}%")
    print(f"Hallucination Rate:      {hallucination_rate:.1f}%")
    print(f"LLM Call Reduction:      {llm_reduction:.1f}%")
    print(f"Avg End-to-End Latency:  {avg_latency:.2f}s")
    print(f"Avg Time-To-Generated:   {avg_llm_latency:.2f}s")
    print("="*40)
    
if __name__ == "__main__":
    run_evaluation()
