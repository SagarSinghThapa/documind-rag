# Phase 6: WP-07 Evaluation and Release — Context & Architecture Contracts

**Gathered:** 2026-11-24  
**Status:** Ready for planning  
**Source:** User Requirements, WP-07 Scope Specification, Phase 1–5 Audited Implementations, ₹0 Budget & Offline Privacy Boundaries

<domain>
## Phase Boundary

Phase 6 (WP-07) is the final evaluation, benchmarking, security hardening, and release preparation phase for Milestone 1 (DocuMind MVP).

It validates the complete integrated system across:
1. Multi-PDF document ingestion and index persistence (`PDFLoader`, `RecursiveCharacterChunker`, `DocumentIndex`, `FaissVectorStore`).
2. Calibrated semantic retrieval and context assembly (`SemanticRetriever`, `ContextBuilder`).
3. Local CPU generation, 3-layer citation verification, and dual-tier abstention (`GroundedGenerator`, `TransformersLLM`, `CitationParser`).
4. Interactive Streamlit user interface (`app.py`, `state.py`, `sidebar.py`, `chat.py`, `citations.py`).

Phase 6 executes quality benchmarks, measures CPU performance metrics, verifies index consistency, performs a full security audit, and finalizes production documentation and demonstration runbooks.

Phase 6 does **NOT**:
- Modify core retrieval or generation algorithms unless a critical defect is uncovered during final evaluation.
- Download new or alternative neural network models (strictly locks to `all-MiniLM-L6-v2` and `Qwen/Qwen2.5-0.5B-Instruct`).
- Implement Milestone 2 features (BM25 lexical search, RRF rank fusion, cross-encoder rerankers, HyDE, agentic graph routing).
- Alter the ₹0 budget or local-first privacy perimeter.

</domain>

<decisions>
## Implementation Decisions & Quality Contracts

### 1. Workstream 1 — Final End-to-End Acceptance (EVAL-01)
- **Multi-PDF Upload & Indexing:** Verify the UI ingestion manager correctly ingests and indexes multiple disparate PDFs (`BASWE-15-RAG-Concepts-Guide.pdf`, `Everything-to-know-about-RAG.pdf`, `FINZAA.pdf`) into a unified FAISS index.
- **Factual & Negative Inquiry Verification:**
  - Positive inquiries must produce answers naming verified entities with valid `[S1]`, `[S2]` source identifiers resolving to accurate physical pages.
  - Negative/unsupported inquiries (e.g. out-of-domain topics) must trigger clean, explicit abstentions with zero hallucinated citations.
- **Persistence Across Restarts:** Verify that quitting Streamlit and restarting instantly restores the complete multi-document registry and FAISS index from disk without re-extracting or re-embedding PDFs.
- **Safe Reset Boundary:** Verify that triggering two-step confirmed reset purges only `data/uploads/` and `data/index/`, while strictly preserving `models/llm/` and application source code.
- **Browser Interaction Protocol:** Document exact manual browser verification steps for the user, and do not claim automated browser control unless an actual browser was driven.

### 2. Workstream 2 — Index Consistency & Snapshot Reconciliation (EVAL-01, EVAL-02)
- **Reconciliation Audit:** Document that the active index contains exactly 436 vectors corresponding to:
  - `BASWE-15-RAG-Concepts-Guide.pdf`: 30 chunks
  - `Everything-to-know-about-RAG.pdf`: 165 chunks
  - `FINZAA.pdf`: 241 chunks (indices 0 to 240)
  - Reconcile the clerical "32 chunks" typo in WP-06 documentation with the verified 241 chunks on disk.
- **Integrity Verification Script:** Create an automated diagnostic tool (`scripts/verify_index_integrity.py`) that checks for orphaned chunks, duplicate hashes, missing metadata records, or index-registry desynchronization.

### 3. Workstream 3 — Retrieval & Generation Quality Benchmarking (EVAL-02)
- **Evaluation Dataset:** Utilize the ground-truth benchmark suite comprising positive queries ($Q_+$) and negative distractor queries ($Q_-$) across public technical references.
- **Metric Definitions:**
  - **Hit@K / Recall@K:** Fraction of positive queries where the ground-truth document chunk is retrieved in top $K$.
  - **Citation Emission Rate:** Fraction of non-abstained answers containing $\ge 1$ inline citation (`[S1]`). Target: $100\%$.
  - **Citation Metadata Validity Rate:** Fraction of citations linking to true active `ContextReference` records. Target: $100\%$.
  - **Evidence Support Verification:** Verified through non-stopword entity/noun token matching across both the citing sentence and the full concise answer ($\le 500$ chars). Target: $\ge 90\%$.
  - **False Abstention Rate:** Fraction of positive queries improperly rejected. Target: $0\%$.
  - **Unsupported Answer Rate:** Fraction of factual answers lacking valid citations. Target: $0\%$.

### 4. Workstream 4 — CPU Performance & Resource Profiling (EVAL-02)
- **Automated Profiling Suite:** Create `scripts/profile_hardware_envelope.py` to measure:
  - Cold model load time for `LocalEmbedder` and `TransformersLLM`.
  - Disaggregated latency: query embedding ($t_{\text{embed}}$), FAISS search ($t_{\text{search}}$), context assembly ($t_{\text{ctx}}$).
  - TTFT (prefill latency) and autoregressive decoding throughput (tokens/sec).
  - End-to-end response latency for short (20 tokens), medium (50 tokens), and long (100 tokens) answers.
  - Peak RSS memory footprint during active inference using `psutil`.

### 5. Workstream 5 — Security, Privacy & Localhost Boundary (EVAL-01, EVAL-02)
- **Localhost Enclosure:** Verify `.streamlit/config.toml` enforces `address = "localhost"`, `port = 8501`, `enableCORS = true`, `allowedHosts = ["localhost", "127.0.0.1"]`, `enableXsrfProtection = true`.
- **Zero Cloud Transmission:** Code audit verifying zero network calls (`requests`, `urllib`, `socket`) to external IP addresses or third-party cloud APIs.
- **Safe File Handling:** Path traversal sanitization, 50 MB upload limits, quarantine isolation.
- **Git Exclusion Verification:** Confirm `.gitignore` comprehensively covers `data/`, `models/`, `.venv/`, `scratch/`, `__pycache__/`, `.pytest_cache/`, and `.streamlit/secrets.toml`.

### 6. Workstream 6 — Testing & Execution Guarantees (EVAL-01)
- **Preserve All 139 Tests:** Ensure zero regressions across the existing 139 passing tests.
- **Sandbox Boundary Protocol:**
  - Execute non-UI unit and integration tests (128 tests) inside the sandbox.
  - Document the host PowerShell command for the 11 Streamlit UI AppTests (`tests/test_ui_app.py`, `tests/test_ui_state.py`) to bypass Windows loopback IPC socket restrictions.
- **Bounded Execution:** Zero unbounded background tasks; strict per-command timeouts.

### 7. Workstream 7 — Comprehensive Documentation & Release Artifacts (EVAL-03)
- **Documentation Deliverables:**
  1. `README.md`: Modern project overview, architectural badges, feature highlights, ₹0 cost model, quickstart, demo screenshots/flow.
  2. `docs/ARCHITECTURE.md`: Subsystem block diagrams, data flow specifications, memory lifecycle, session isolation contracts.
  3. `docs/WINDOWS_SETUP_GUIDE.md`: Step-by-step Windows 11 installation, Python 3.11 venv, OpenMP/delvewheel DLL handling, local model provisioning.
  4. `docs/DEMONSTRATION_RUNBOOK.md`: Scripted presentation walkthrough, sample queries, citation verification tour, and Q&A edge cases.
  5. `docs/TROUBLESHOOTING.md`: Common operational issues (port 8501 conflicts, loopback socket limits, CORS warnings, scanned PDF OCR requirements).
  6. `docs/RELEASE_CHECKLIST.md`: Pre-flight verification checklist for Milestone 1 MVP sign-off.
  7. `docs/project_reports/WP_07_FINAL_RELEASE_REPORT.md`: Comprehensive audit report with empirical tables and final release verdict.

</decisions>

<requirements>
## Traceability Matrix

| Requirement | Phase Scope | Plan Mapping | Acceptance Gate |
| :--- | :--- | :--- | :--- |
| **EVAL-01** | End-to-end integration tests & pipeline validation | Plan 06-01, Plan 06-02 | Multi-PDF upload, Q&A, citation linking, persistence, reset, and 139 passing tests verified |
| **EVAL-02** | Retrieval quality, citation accuracy & CPU hardware envelope | Plan 06-01 | Hit@K $\ge 90\%$, citation validity $100\%$, false abstention $0\%$, RAM $\le 24$ GB, TPS $> 1.5$ |
| **EVAL-03** | Technical documentation, architecture, demo runbook & release guide | Plan 06-02 | README, ARCHITECTURE, WINDOWS_SETUP, DEMO, TROUBLESHOOTING, and RELEASE checklist complete |

</requirements>
