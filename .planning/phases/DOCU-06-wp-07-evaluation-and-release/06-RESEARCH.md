# Phase 6: WP-07 Evaluation and Release — Research & Empirical Baseline

**Date:** 2026-11-24  
**Author:** Senior RAG Engineer, QA Lead & Systems Architect  
**Scope:** Phase 6 (WP-07 Evaluation, Performance Profiling, Quality Benchmarking, Security Audit, and Release Preparation)

---

## 1. Index Consistency & Snapshot Reconciliation

### 1.1 Empirical Audit of Active Vector Store & Document Registry
A critical investigation was conducted into the reported chunk count discrepancy for `FINZAA.pdf` (noted as 32 in a markdown table in WP-06 vs 241 in the system registry).

Direct inspection of `data/index/document_registry.json` and the active FAISS snapshot via Python yielded:
- **`data/index/document_registry.json`:**
  - `BASWE-15-RAG-Concepts-Guide.pdf`: 14 pages, **30 chunks** (ID: `6a30092e0998...`)
  - `Everything-to-know-about-RAG.pdf`: 92 pages, **165 chunks** (ID: `606ad137b843...`)
  - `FINZAA.pdf`: 118 pages, **241 chunks** (ID: `95cca1e4774a...`)
  - **Total Registered Chunks:** $30 + 165 + 241 = \mathbf{436\text{ chunks}}$
- **Active FAISS Vector Store (`data/index/`):**
  - **`total_vectors`:** $\mathbf{436}$
  - **Records Matching `FINZAA.pdf`:** $\mathbf{241}$
  - **Chunk Indices:** Exact contiguous sequence from `0` to `240` (`chunk_index: 0 .. 240`)
  - **Dimension:** 384 (`float32`), metric: inner product (`IndexFlatIP`)
- **Finding & Root Cause:**
  - The vector store, JSON registry, and serialized manifest are **100% synchronized and bit-for-bit consistent** at 241 chunks for `FINZAA.pdf` and 436 total vectors.
  - The "32 chunks" reference was a clerical transcription typo in the markdown verification checklist of `WP_06_STREAMLIT_ACCEPTANCE.md`.
  - Zero index corruption, zero duplicate chunk IDs, and zero stale vectors exist in the active snapshot.

---

## 2. Retrieval & Generation Quality Evaluation Methodology

### 2.1 Limitations of Naive Lexical Overlap
Phase 4 and Phase 5 revealed subtle traps when evaluating grounded generation with small language models (Qwen2.5-0.5B-Instruct):
1. **The Multi-Sentence Citation Problem:** Small LLMs frequently split answers into two sentences:
   - Sentence 1: Factual assertion (*"Anjali Singh is the author of FINZAA.pdf."*)
   - Sentence 2: Citation placement (*"The first page of the paper is located at [S1]"*)
   A naive claim extractor isolating only the sentence containing `[S1]` evaluates Sentence 2, completely missing the factual content words (*Anjali*, *Singh*) in Sentence 1.
2. **Lexical Matching Pitfalls:**
   - Common stopwords or generic query terms (*"author"*, *"paper"*, *"document"*) match almost any academic front-matter, leading to false-positive evidence support.
   - Conversely, slight morphological shifts or OCR token boundaries cause false-negative evidence rejection.
3. **Rigorous Quality Protocol for Phase 6:**
   - **Disaggregated Metrics:** Separate Positive Queries ($Q_+$, requiring verified citations) from Negative/Distractor Queries ($Q_-$, requiring immediate abstention).
   - **Citation Syntax Validity:** $\frac{\text{Valid } [S1] \text{ Citations}}{\text{Total Emitted Citations}}$
   - **Citation Metadata Resolution:** $\frac{\text{Citations linked to true active ContextReference}}{\text{Valid Syntax Citations}}$
   - **Evidence Support Verification:** Verified through content token intersection ($\ge 1$ specific non-stopword entity/noun token) evaluated across both the immediate sentence and the full concise answer ($\le 500$ chars).
   - **False Abstention Rate:** Percentage of positive queries where the model or pipeline improperly abstains despite sufficient retrieved evidence.
   - **Unsupported Answer Rate:** Percentage of queries where factual claims are made with missing or invalid citations.

---

## 3. Local CPU Performance & Hardware Envelope Profiling

### 3.1 Hardware Envelope Target
- **Platform:** Windows 11 x64, 8-Core CPU, 24 GB Physical RAM.
- **Constraints:** ₹0 cloud cost, zero external API keys, 100% offline local inference.

### 3.2 Key Latency & Resource Benchmarks to Measure
To establish production-grade reproducibility for portfolio demonstration, Phase 6 must measure and document:
1. **Cold Initialization Latency:**
   - `LocalEmbedder` initialization and weights load (~1.8s).
   - FAISS snapshot load from disk (~0.05s).
   - `TransformersLLM` cold load from Safetensors (`models/llm`, ~2.1s).
2. **Warm Query Pipeline Breakdown:**
   - Query embedding time on CPU ($t_{\text{embed}} \approx 10\text{--}18\text{ ms}$).
   - FAISS inner product search time ($t_{\text{search}} \approx 0.5\text{--}2.0\text{ ms}$).
   - Context assembly & deduplication ($t_{\text{context}} \approx 1.0\text{--}3.0\text{ ms}$).
   - Pre-generation fast-path abstention latency for distractors ($t_{\text{fast-path}} < 15\text{ ms}$).
3. **LLM Generation Dynamics on CPU:**
   - Prompt prefill latency (Time to First Token / TTFT) across context lengths (600 to 1,800 characters).
   - Autoregressive token decoding rate (Target: $> 1.5\text{ tokens/sec}$ on CPU, up to $3.5\text{ tokens/sec}$).
   - End-to-end response latency for 50–100 token answers ($t_{\text{total}} \approx 30\text{--}48\text{ s}$).
4. **Memory Footprint (Peak RSS):**
   - Baseline OS + Python venv: ~80 MB.
   - Post-embedding & FAISS load: ~350 MB.
   - Post-LLM load & active generation: $\approx 1.6\text{ GB}$ (well within the $\le 24\text{ GB}$ ceiling).

---

## 4. Security, Privacy & Localhost Boundary

### 4.1 Security Baseline
1. **Localhost Binding:** Enforced in `.streamlit/config.toml` (`server.address = "localhost"`, `server.port = 8501`).
2. **CORS & Origin Restriction:** `server.enableCORS = true`, `server.allowedHosts = ["localhost", "127.0.0.1"]`.
3. **XSRF Protection:** `server.enableXsrfProtection = true` active on all file upload endpoints.
4. **Local Data Privacy:** Zero network transmission of uploaded PDF bytes. Zero external telemetry (`browser.gatherUsageStats = false`).
5. **Storage Isolation:** Uploaded files quarantined to `data/uploads/{hash[:12]}_{safe_filename}` with strict path traversal stripping (`..`, `/`, `\`).
6. **Destructive Action Safety:** Sidebar index reset requires explicit two-step confirmation (`confirm_reset_checkbox`), deleting only `data/uploads/` and `data/index/snapshots/`, while strictly preserving `models/llm/` and source code.
7. **Version Control Exclusion:** `.gitignore` must be verified to exclude all sensitive directories (`data/`, `models/`, `.venv/`, `scratch/`).

---

## 5. Testing & Verification Safeguards

### 5.1 Test Suite Organization & Sandbox Boundary
- **Baseline Test Suite:** **139 passing tests** across 14 test modules.
- **Windows Sandbox Loopback Constraint:**
  - Standard Antigravity sandboxes restrict Windows loopback TCP sockets (`127.0.0.1`), causing headless Streamlit `AppTest.run()` to hang waiting for internal IPC socket bindings.
  - **Operating Protocol:** Non-UI unit tests (128 tests across ingestion, chunker, vector store, retriever, LLM, prompts, and citations) run cleanly inside the sandbox in $< 35\text{ seconds}$.
  - The 11 Streamlit UI tests (`tests/test_ui_app.py`, `tests/test_ui_state.py`) must be executed in host PowerShell outside the sandbox (`.venv\Scripts\python.exe -m pytest -v tests/test_ui_app.py tests/test_ui_state.py`), taking $\approx 19.2\text{ seconds}$.

---

## 6. Phase 6 Deliverables & Documentation Matrix

| Deliverable | Purpose | Target Location |
| :--- | :--- | :--- |
| **Comprehensive README** | High-level architectural overview, features, ₹0 cost model, quickstart | `README.md` |
| **Architecture Guide** | Deep dive into subsystem contracts, memory lifecycle, data flows | `docs/ARCHITECTURE.md` |
| **Windows Setup Runbook** | Python 3.11, venv, delvewheel/OpenMP DLL handling, local model provisioning | `docs/WINDOWS_SETUP_GUIDE.md` |
| **Portfolio Demo Runbook** | Scripted demonstration flow, sample queries, citation verification tour | `docs/DEMONSTRATION_RUNBOOK.md` |
| **Troubleshooting Guide** | Solutions for port conflicts, loopback sockets, CORS, OCR limitations | `docs/TROUBLESHOOTING.md` |
| **Release Checklist** | Pre-flight verification checklist for Milestone 1 release | `docs/RELEASE_CHECKLIST.md` |
| **Final Audit Report** | Independent verification report covering EVAL-01 through EVAL-03 | `docs/project_reports/WP_07_FINAL_RELEASE_REPORT.md` |
