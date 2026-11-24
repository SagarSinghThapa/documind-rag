# Roadmap: DocuMind

## Overview

DocuMind is an offline, local-first Retrieval-Augmented Generation (RAG) system built from scratch to deliver grounded answers with verifiable citations over diverse PDF documents. The project executes in strict dependency order: beginning with an independent audit and environment repair of the existing ingestion pipeline (WP-02), followed by CPU-compatible local embeddings and vector indexing (WP-03), semantic retrieval (WP-04), local LLM grounded generation with citations (WP-05), an interactive Streamlit UI (WP-06), and culminating in end-to-end evaluation, benchmarking, and release preparation (WP-07).

## Milestone 1: DocuMind MVP

- [x] **Phase 1: WP-02 Independent Audit and Environment Repair** — Complete (1/1 plans executed and verified). Establish working Python venv, verify dependency compatibility, execute all tests, audit extraction/chunking/metadata, and produce audit report.
  - Summaries:
    - [01-01-SUMMARY.md](file:///D:/rag/.planning/phases/DOCU-01-wp-02-independent-audit-and-environment-repair/01-01-SUMMARY.md)
  - Audit Report:
    - [WP_02_INDEPENDENT_AUDIT_REPORT.md](file:///D:/rag/docs/project_reports/WP_02_INDEPENDENT_AUDIT_REPORT.md)
- [x] **Phase 2: WP-03 Local Embeddings and Persistent Vector Index** — Complete (2/2 plans executed and verified). Select CPU-compatible embedding model and indexing backend (FAISS), implement persistent storage with metadata, two-level deduplication, crash-consistent snapshots, and tests.
  - Summaries:
    - [02-01-SUMMARY.md](file:///D:/rag/.planning/phases/DOCU-02-wp-03-local-embeddings-and-persistent-vector-index/02-01-SUMMARY.md)
    - [02-02-SUMMARY.md](file:///D:/rag/.planning/phases/DOCU-02-wp-03-local-embeddings-and-persistent-vector-index/02-02-SUMMARY.md)
- [x] **Phase 3: WP-04 Semantic Retrieval** — Complete (2/2 plans executed and verified). Implement query embeddings, Top-K vector retrieval, relevance filtering, context construction, and retrieval evaluation.
  - Summaries:
    - [03-01-SUMMARY.md](file:///D:/rag/.planning/phases/DOCU-03-wp-04-semantic-retrieval/03-01-SUMMARY.md)
    - [03-02-SUMMARY.md](file:///D:/rag/.planning/phases/DOCU-03-wp-04-semantic-retrieval/03-02-SUMMARY.md)
- [x] **Phase 4: WP-05 Local LLM and Grounded Generation** — Complete (2/2 plans executed and verified). Integrate CPU-compatible local LLM, generate grounded answers, implement 3-layer citation verification, dual-tier abstention, and grounding evaluation.
  - Summaries:
    - [04-01-SUMMARY.md](file:///D:/rag/.planning/phases/DOCU-04-wp-05-local-llm-and-grounded-generation/04-01-SUMMARY.md)
    - [04-02-SUMMARY.md](file:///D:/rag/.planning/phases/DOCU-04-wp-05-local-llm-and-grounded-generation/04-02-SUMMARY.md)
  - Plans:
    - [04-01-PLAN.md](file:///D:/rag/.planning/phases/DOCU-04-wp-05-local-llm-and-grounded-generation/04-01-PLAN.md)
    - [04-02-PLAN.md](file:///D:/rag/.planning/phases/DOCU-04-wp-05-local-llm-and-grounded-generation/04-02-PLAN.md)
- [x] **Phase 5: WP-06 Streamlit Application** — Complete. Web interface with multi-PDF upload, document management, indexing controls, conversational Q&A, and citation display.
  - Acceptance Report:
    - [WP_06_STREAMLIT_ACCEPTANCE.md](file:///D:/rag/docs/project_reports/WP_06_STREAMLIT_ACCEPTANCE.md)
  - Plans:
    - [05-01-PLAN.md](file:///D:/rag/.planning/phases/DOCU-05-wp-06-streamlit-application/05-01-PLAN.md)
    - [05-02-PLAN.md](file:///D:/rag/.planning/phases/DOCU-05-wp-06-streamlit-application/05-02-PLAN.md)
- [ ] **Phase 6: WP-07 Evaluation and Release** — End-to-end integration testing, retrieval and citation evaluation, RAM and latency benchmarking, documentation, and demonstration prep.
  - Plans:
    - [06-01-PLAN.md](file:///D:/rag/.planning/phases/DOCU-06-wp-07-evaluation-and-release/06-01-PLAN.md)
    - [06-02-PLAN.md](file:///D:/rag/.planning/phases/DOCU-06-wp-07-evaluation-and-release/06-02-PLAN.md)

---

## Phase Details

### Phase 1: WP-02 Independent Audit and Environment Repair
**Goal**: Inspect existing WP-02 implementation, establish a functional Python virtual environment (`.venv`), verify dependency compatibility on Windows CPU, execute all tests, audit extraction/chunking/metadata/privacy, resolve confirmed blocking defects, and produce an evidence-based audit report.  
**Depends on**: Nothing (initial executable phase).  
**Requirements**: `AUDIT-01`, `AUDIT-02`, `AUDIT-03`, `AUDIT-04`  
**Success Criteria** (what must be TRUE):
  1. A dedicated Python virtual environment (`.venv`) is created, active, and dependencies from `requirements.txt` (`pymupdf`, `pytest`) are installed and verified.
  2. All existing tests in `tests/` execute via `pytest` and pass with 0 failures, establishing the verified baseline test count.
  3. PDF extraction, conservative text normalization, character offset slicing, strict chunk bounds, and deterministic SHA-256 IDs are independently audited and confirmed on sample PDFs.
  4. Privacy boundary is verified: zero external cloud calls, and `resumes/` remains completely local and excluded from version control.
  5. An evidence-based audit report is produced, and user approval is explicitly obtained before proceeding to Phase 2.  
**Plans**: 1 plan

Plans:
- [x] 01-01: Environment repair, test execution, WP-02 audit, defect remediation, and audit report generation.

---

### Phase 2: WP-03 Local Embeddings and Persistent Vector Index
**Goal**: Benchmark CPU-compatible embedding models, compare FAISS and Chroma for local persistence, select a model and indexing backend, implement embedding generation and persistent vector storage with complete document and page metadata, and add automated tests.  
**Depends on**: Phase 1 (requires explicit audit approval).  
**Requirements**: `EMBED-01`, `EMBED-02`, `EMBED-03`, `EMBED-04`  
**Success Criteria** (what must be TRUE):
  1. A lightweight, high-performance CPU embedding model (`all-MiniLM-L6-v2`) is selected and integrated without paid APIs.
  2. A local persistence backend (`faiss-cpu`) is implemented with immutable versioned snapshots running entirely in-process.
  3. Chunks are embedded and indexed with immutable metadata preserved: `document_id`, `chunk_id`, `document_name`, `page_numbers`, `start_page`, `end_page`, `char_start`, `char_end`, `text`.
  4. Automated unit tests verify vector generation, index persistence to disk, index reload, and metadata integrity.  
**Plans**: 2 plans

Plans:
- [x] 02-01: Embedding Engine & Configuration (`all-MiniLM-L6-v2`, storage schemas, `LocalEmbedder`, embedder unit tests).
- [x] 02-02: Persistent FAISS Vector Store, Deduplication & High-Level `DocumentIndex` (`FaissVectorStore`, atomic snapshots, `DocumentIndex`, integration tests & benchmarks).

---

### Phase 3: WP-04 Semantic Retrieval
**Goal**: Implement query embedding generation, Top-K vector similarity search, relevance threshold filtering, formatted context window construction, and retrieval evaluation.  
**Depends on**: Phase 2.  
**Requirements**: `RETR-01`, `RETR-02`, `RETR-03`, `RETR-04`  
**Success Criteria** (what must be TRUE):
  1. User queries are encoded using the identical embedding model and dimension space as the indexed documents.
  2. Top-K vector search returns candidate chunks with similarity scores and page metadata.
  3. Relevance filtering discards low-confidence matches below a defined similarity threshold.
  4. Clean context blocks are constructed with unambiguous source headers (`[Document: {name}, Page: {p}]`) ready for downstream generation.
  5. Automated retrieval tests evaluate precision and recall against sample queries.  
**Plans**: 2 plans

Plans:
- [x] 03-01: Core Semantic Retriever, Schemas & Threshold Filtering (`SemanticRetriever`, query validation, Top-K ranking, relevance filtering, unit tests).
- [x] 03-02: ContextBuilder, Evaluation Framework & Calibration Suite (`ContextBuilder`, overlap deduplication, citation headers, ground-truth dataset, `RetrievalEvaluator`, benchmark script).

---

### Phase 4: WP-05 Local LLM and Grounded Generation
**Goal**: Integrate a CPU-compatible local LLM, generate answers strictly bounded by retrieved context, format explicit document and page citations, and enforce abstention when evidence is absent.  
**Depends on**: Phase 3.  
**Requirements**: `GEN-01`, `GEN-02`, `GEN-03`, `GEN-04`  
**Success Criteria** (what must be TRUE):
  1. A quantized local LLM (e.g., Llama 3.2 1B/3B or Qwen 2.5 1.5B/3B via `llama-cpp-python` or Ollama) runs locally on CPU within the 24 GB RAM envelope.
  2. Generated answers cite source documents and physical page numbers corresponding to retrieved chunks.
  3. When queried on topics absent from the indexed documents, the generator explicitly abstains (e.g., "I cannot answer this based on the provided documents").
  4. Automated grounding tests confirm that answers do not hallucinate outside the retrieved context.  
**Plans**: 2 plans

Plans:
- [x] 04-01: Local LLM Engine, Configuration, Prompt Templates & Mocking (`TransformersLLM`, `MockLLM`, `GenerationConfig`, prompt contracts, offline safety).
- [x] 04-02: Grounded Generation, Citation Verification, Abstention & Evaluation (`CitationParser`, `GroundedGenerator`, dual-tier abstention, `GroundingEvaluator`, test suites).

---

### Phase 5: WP-06 Streamlit Application
**Goal**: Build a clean, responsive Streamlit web application supporting multi-PDF upload, document management, indexing status, conversational Q&A, and interactive citation inspection.  
**Depends on**: Phase 4.  
**Requirements**: `APP-01`, `APP-02`, `APP-03`, `APP-04`  
**Success Criteria** (what must be TRUE):
  1. Users can upload multiple PDF documents through the web interface with immediate validation feedback.
  2. Document manager displays indexed files, page counts, chunk counts, and allows re-indexing or deletion.
  3. Interactive chat UI displays assistant responses with expandable citation accordions showing exact page numbers and chunk text.
  4. Non-fatal errors (corrupted files, empty inputs, unanswerable queries) are surfaced gracefully without crashing the UI.  
**Plans**: 2 plans

Plans:
- [x] 05-01: Core UI State, Multi-PDF Ingestion Pipeline & Document Manager (`state.py`, `sidebar.py`, file upload, index controls, document table, and unit tests).
- [x] 05-02: Conversational Q&A, Token Streaming, Citation Accordions & Headless AppTest Suite (`chat.py`, `citations.py`, `app.py`, streaming integration, and comprehensive `AppTest` suite).

---

### Phase 6: WP-07 Evaluation and Release
**Goal**: Conduct end-to-end integration testing, evaluate retrieval and generation quality, benchmark latency and RAM consumption, finalize documentation, and prepare portfolio presentation materials.  
**Depends on**: Phase 5.  
**Requirements**: `EVAL-01`, `EVAL-02`, `EVAL-03`  
**Success Criteria** (what must be TRUE):
  1. Index consistency is fully verified and documented: 436 vectors match exactly 436 registered chunks across 3 documents, reconciling FINZAA's 241 chunks.
  2. Disaggregated quality evaluation across positive ($Q_+$) and negative ($Q_-$) queries confirms $100\%$ citation validity, $\ge 90\%$ evidence support, and $0\%$ false abstentions.
  3. CPU hardware benchmarks profile cold load, query embedding, FAISS search, context build, TTFT, token decoding throughput, and peak RSS ($\le 16$ GB RAM).
  4. Security audit confirms localhost-only binding, CORS/XSRF protection, zero external cloud data transfer, safe quarantine file paths, and `.gitignore` coverage.
  5. Multi-PDF end-to-end integration verifies simultaneous retrieval, citation, and abstention across multiple distinct indexed documents.
  6. Comprehensive release documentation (`README.md`, `docs/ARCHITECTURE.md`, `docs/WINDOWS_SETUP_GUIDE.md`, `docs/DEMONSTRATION_RUNBOOK.md`, `docs/TROUBLESHOOTING.md`, `docs/RELEASE_CHECKLIST.md`, `docs/project_reports/WP_07_FINAL_RELEASE_REPORT.md`) is complete.
  7. All 139 existing automated tests remain 100% passing across isolated test runs.  
**Plans**: 2 plans

Plans:
- [x] 06-01: Index Consistency, Quality Benchmarking, CPU Performance Profiling & Security Audit (`verify_index_integrity.py`, `profile_hardware_envelope.py`, `evaluate_grounding_quality.py`, security checks).
- [x] 06-02: Multi-PDF End-to-End Acceptance, Comprehensive Technical Documentation, Portfolio Demo Runbook & Release Gate (`verify_e2e_acceptance.py`, `README.md`, `ARCHITECTURE.md`, `WINDOWS_SETUP_GUIDE.md`, `DEMONSTRATION_RUNBOOK.md`, `TROUBLESHOOTING.md`, `RELEASE_CHECKLIST.md`, final release report).  
