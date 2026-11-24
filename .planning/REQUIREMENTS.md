# Requirements: DocuMind

**Defined:** 2026-11-22  
**Core Value:** Deliver accurate, grounded answers with verifiable page-level citations from local documents under a strict ₹0 cloud API budget and local-first privacy boundary.

## v1 Requirements (Milestone 1 — DocuMind MVP)

Requirements for the initial MVP release. Each requirement maps to a specific roadmap phase.

### Phase 1: WP-02 Independent Audit and Environment Repair

- [ ] **AUDIT-01**: Establish a working Python virtual environment (`.venv`) on Windows CPU with verified dependency compatibility.
- [ ] **AUDIT-02**: Execute all existing unit and regression tests in `tests/`, verifying 100% pass rate.
- [ ] **AUDIT-03**: Audit PDF extraction, text normalization, chunking, deterministic hashing, page attribution, and privacy boundaries.
- [ ] **AUDIT-04**: Resolve any confirmed blocking defects and produce an evidence-based audit report requiring explicit approval before beginning WP-03.

### Phase 2: WP-03 Local Embeddings and Persistent Vector Index

- [ ] **EMBED-01**: Evaluate and select a CPU-compatible local embedding model operating without external API keys or cloud services.
- [ ] **EMBED-02**: Evaluate and select a local persistence backend (FAISS or Chroma) that runs embedded in-process.
- [ ] **EMBED-03**: Implement embedding generation and persistent vector indexing preserving document ID, filename, and physical page metadata.
- [ ] **EMBED-04**: Add automated tests verifying vector generation, index persistence, metadata roundtripping, and search retrieval.

### Phase 3: WP-04 Semantic Retrieval

- [ ] **RETR-01**: Implement query embedding generation matching the vector index space.
- [ ] **RETR-02**: Implement Top-K vector similarity retrieval with relevance score filtering.
- [ ] **RETR-03**: Implement formatted context window construction preserving document boundaries and page citations.
- [ ] **RETR-04**: Add retrieval evaluation testing retrieval accuracy and latency across sample benchmark queries.

### Phase 4: WP-05 Local LLM and Grounded Generation

- [ ] **GEN-01**: Integrate a lightweight, CPU-compatible local LLM (e.g. via llama-cpp-python or Ollama) running locally on CPU.
- [ ] **GEN-02**: Generate answers strictly grounded in retrieved document context with precise document and page number citations.
- [ ] **GEN-03**: Implement explicit abstention mechanisms for questions not supported by the retrieved evidence.
- [ ] **GEN-04**: Add automated tests validating grounding faithfulness, citation precision, and abstention behavior.

### Phase 5: WP-06 Streamlit Application

- [ ] **APP-01**: Build a responsive Streamlit UI supporting drag-and-drop upload of multiple PDF files.
- [ ] **APP-02**: Implement document management and indexing controls showing real-time status and chunk counts.
- [ ] **APP-03**: Implement a conversational chat interface with expandable source citations displaying exact page references.
- [ ] **APP-04**: Provide clear, user-friendly error handling for malformed documents, empty queries, or processing timeouts.

### Phase 6: WP-07 Evaluation and Release

- [ ] **EVAL-01**: Implement end-to-end integration tests validating the complete pipeline from PDF upload to cited answer.
- [ ] **EVAL-02**: Measure and document retrieval quality, citation accuracy, and hardware envelope ($\le 24$ GB RAM, CPU latency).
- [ ] **EVAL-03**: Finalize comprehensive README, architecture diagrams, and portfolio demonstration runbook.

---

## v2 Requirements (Milestone 2 — Advanced DocuMind)

Deferred to future release. Tracked but explicitly not part of the Milestone 1 MVP roadmap.

### Hybrid Retrieval & Ranking

- **HYBRID-01**: Implement BM25 lexical search alongside dense vector retrieval.
- **HYBRID-02**: Merge keyword and semantic retrieval results using Reciprocal Rank Fusion (RRF).
- **RERANK-01**: Implement cross-encoder reranking over fused candidate chunks.

### Advanced Query Processing

- **QUERY-01**: Implement query rewriting, sub-query decomposition, and HyDE (Hypothetical Document Embeddings).
- **COMPARE-01**: Implement multi-document comparison and cross-document synthesis workflows.

### Observability & Evaluation

- **DASH-01**: Build an interactive evaluation dashboard displaying RAG triad metrics (faithfulness, answer relevance, context precision).

---

## Out of Scope

Explicitly excluded to protect the ₹0 budget, local-first privacy boundary, and MVP focus.

| Feature / Pattern | Reason |
| :--- | :--- |
| Paid cloud LLM APIs (OpenAI, Anthropic, Gemini) | Strict ₹0 budget constraint and interview reproducibility |
| Cloud vector databases (Pinecone, Weaviate Cloud) | Privacy violation; external network dependency |
| Cloud hosting / remote web servers | Local desktop/developer execution only |
| Agentic multi-hop query routing (LangGraph/CrewAI) | Unnecessary complexity before baseline RAG is verified |
| Complex distributed processing (Celery/Redis/Spark) | Overkill for single-node local execution within 24 GB RAM |

---

## Traceability Matrix

| Requirement | Phase | Status |
| :--- | :--- | :--- |
| AUDIT-01 | Phase 1: WP-02 Independent Audit and Environment Repair | Pending |
| AUDIT-02 | Phase 1: WP-02 Independent Audit and Environment Repair | Pending |
| AUDIT-03 | Phase 1: WP-02 Independent Audit and Environment Repair | Pending |
| AUDIT-04 | Phase 1: WP-02 Independent Audit and Environment Repair | Pending |
| EMBED-01 | Phase 2: WP-03 Local Embeddings and Persistent Vector Index | Pending |
| EMBED-02 | Phase 2: WP-03 Local Embeddings and Persistent Vector Index | Pending |
| EMBED-03 | Phase 2: WP-03 Local Embeddings and Persistent Vector Index | Pending |
| EMBED-04 | Phase 2: WP-03 Local Embeddings and Persistent Vector Index | Pending |
| RETR-01 | Phase 3: WP-04 Semantic Retrieval | Pending |
| RETR-02 | Phase 3: WP-04 Semantic Retrieval | Pending |
| RETR-03 | Phase 3: WP-04 Semantic Retrieval | Pending |
| RETR-04 | Phase 3: WP-04 Semantic Retrieval | Pending |
| GEN-01 | Phase 4: WP-05 Local LLM and Grounded Generation | Pending |
| GEN-02 | Phase 4: WP-05 Local LLM and Grounded Generation | Pending |
| GEN-03 | Phase 4: WP-05 Local LLM and Grounded Generation | Pending |
| GEN-04 | Phase 4: WP-05 Local LLM and Grounded Generation | Pending |
| APP-01 | Phase 5: WP-06 Streamlit Application | Pending |
| APP-02 | Phase 5: WP-06 Streamlit Application | Pending |
| APP-03 | Phase 5: WP-06 Streamlit Application | Pending |
| APP-04 | Phase 5: WP-06 Streamlit Application | Pending |
| EVAL-01 | Phase 6: WP-07 Evaluation and Release | Pending |
| EVAL-02 | Phase 6: WP-07 Evaluation and Release | Pending |
| EVAL-03 | Phase 6: WP-07 Evaluation and Release | Pending |

**Coverage:**
- v1 requirements: 23 total
- Mapped to phases: 23
- Unmapped: 0 ✓

---
*Requirements defined: 2026-11-22*
*Last updated: 2026-11-22 after project initialization*
