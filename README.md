# Omnimart CX Assistant

A production-grade, cost-optimized, Retrieval-Augmented Generation (RAG) chatbot system for company policy and product information queries.

## Features
- **Deterministic Guardrails**: Prompt injection, semantic domain, retrieval, and output guardrails running locally without LLM usage.
- **Hybrid Retrieval**: Combines FAISS (dense embeddings) and BM25 (keyword matching) for robust semantic and lexical search.
- **Local LLM**: Uses Phi-3 Mini strictly for compliant answer generation based ONLY on the retrieved context. No hallucinations allowed.
- **Full Evaluation Framework**: Built-in scripts to evaluate hallucination rates, gatekeeper accuracy, and retrieval success against a 30-query dataset.

## Setup Instructions

1. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
3. **Setup Ollama**
   Ensure you have [Ollama](https://ollama.com/) installed and running locally. We will be using the `phi3:mini` model.
   ```bash
   ollama pull phi3:mini
   ollama serve
   ```

4. **Run Data Ingestion**
   This script chunks documents in `data/docs`, generates BGE embeddings, and builds FAISS and BM25 indices.
   ```bash
   python scripts/ingest.py
   ```

## Running the Application

1. **Start the FastAPI Backend**
   ```bash
   python api/main.py
   ```
   The API will be available at `http://localhost:8000`. Let it run in the background.

2. **Start the Streamlit Frontend**
   In a new terminal, run:
   ```bash
   streamlit run frontend/app.py
   ```

## Evaluation
Run the automated evaluation pipeline to test the RAG system against the dataset:
```bash
python scripts/evaluate.py
```
