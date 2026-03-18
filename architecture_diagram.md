# Omnimart CX Assistant Architecture

Below is the complete system architecture diagram representing the flow of data through the deterministic guardrails, retrieval system, and local LLM.

```mermaid
flowchart TD
    classDef ui fill:#326ce5,stroke:#fff,stroke-width:2px,color:#fff
    classDef api fill:#4caf50,stroke:#fff,stroke-width:2px,color:#fff
    classDef guardrail fill:#ff9800,stroke:#fff,stroke-width:2px,color:#fff
    classDef vector fill:#9c27b0,stroke:#fff,stroke-width:2px,color:#fff
    classDef llm fill:#e91e63,stroke:#fff,stroke-width:2px,color:#fff
    classDef block fill:#f44336,stroke:#fff,stroke-width:2px,color:#fff

    subgraph User Interface
        UI[Streamlit Frontend <br> 'app.py']:::ui
    end

    subgraph Backend Services
        API[FastAPI Backend <br> 'main.py']:::api
    end

    subgraph Input Phase
        IG1[Input Guardrail <br> Prompt Injection Det.]:::guardrail
        IG2[Semantic Gatekeeper <br> Domain Validation]:::guardrail
    end

    subgraph Retrieval Phase
        Embed[BGE-Small-EN <br> Embedder]:::vector
        HybridRetriever[Hybrid Retriever Layer]:::vector
        FAISS[(FAISS Index <br> Dense Vector)]:::vector
        BM25[(BM25 Index <br> Keyword Sparse)]:::vector
    end

    subgraph Generation Phase
        RG[Retrieval Guardrail <br> L2 Distance / Context Count]:::guardrail
        PHI3[Local Phi-3 Mini <br> Ollama Client]:::llm
        OG[Output Guardrail <br> Num. Hallucination Check]:::guardrail
    end

    OutReject[Reject & Fallback]:::block
    OutSafe[Display Clean Answer]:::ui

    UI --> |User Query| API
    
    %% Input Guardrails
    API --> IG1
    IG1 --> |Injection Found| OutReject
    IG1 --> |Safe| IG2
    IG2 --> |Out of Scope| OutReject
    IG2 --> |In Domain| Embed

    %% Retrieval Pipeline
    Embed --> HybridRetriever
    HybridRetriever --> FAISS
    HybridRetriever --> BM25
    FAISS -.-> HybridRetriever
    BM25 -.-> HybridRetriever

    %% Generation & Quality Control
    HybridRetriever --> RG
    RG --> |Low Relevance| OutReject
    RG --> |Validated Context| PHI3
    
    PHI3 --> |Raw LLM Answer| OG
    OG --> |Definitions / Numbers Fail| OutReject
    OG --> |Accurate & Grounded| OutSafe
```

## System Components

1. **Streamlit UI (`frontend/app.py`)**: The chat interface connecting users to the pipeline.
2. **FastAPI (`api/main.py`)**: The asynchronous backend engine routing queries.
3. **Deterministic Guardrails**:
   * *Input*: Blocks prompt injections and completely out-of-domain conversational requests.
   * *Retrieval*: Aborts request if the retrieved chunks are irrelevant (high L2 distance).
   * *Output*: Catches numerical hallucinations or leaked general knowledge before the user sees the answer!
4. **Hybrid Retrieval**: Queries both a Dense Vector DB (FAISS) and Sparse DB (BM25) for extreme recall precision.
5. **Local LLM (`src/llm/phi3_client.py`)**: Runs entirely locally via Ollama with strict `phi3` ChatML formatting boundaries to minimize pre-training leakages.
