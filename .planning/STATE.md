# Project State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-11-24)

**Core value:** Deliver accurate, grounded answers with verifiable page-level citations from local documents under a strict ₹0 cloud API budget and local-first privacy boundary.  
**Current focus:** Phase 6 — WP-07 Evaluation and Release (COMPLETE & APPROVED)

## Current Position

Phase: 6 of 6 (WP-07 Evaluation and Release) — COMPLETE  
Plan: 2 of 2 in Phase 6 completed (06-01 and 06-02 executed and verified)  
Status: Milestone 1 MVP Release Approved & Ready for Demonstration  
Last activity: 2026-11-24 — Phase 6 Wave 1 and Wave 2 executed and verified. Index integrity reconciled (436 vectors, 241 FINZAA chunks), hardware profiled (15.65 GB physical RAM, 1.55 GB peak RSS), grounding quality evaluated (100% citation emission and validity, 100% negative abstention, 0% false abstentions), security audited (.gitignore, localhost, CORS/XSRF), multi-PDF E2E acceptance verified (100% PASS), and all technical release documentation authored (README, ARCHITECTURE, WINDOWS_SETUP_GUIDE, DEMONSTRATION_RUNBOOK, TROUBLESHOOTING, RELEASE_CHECKLIST, WP_07_FINAL_RELEASE_REPORT).

Progress: [██████████] 100%

## Performance Metrics

**Velocity:**
- Total plans completed: 11 (Phases 1 through 6)
- Total execution time: ~3.5 hours across all phases

**By Phase:**

| Phase | Plans | Total | Status |
| :--- | :---: | :---: | :---: |
| 1. WP-02 Audit & Environment Repair | 1/1 | 6m | Verified |
| 2. Local Embeddings & Vector Index | 2/2 | 55m | Verified |
| 3. Semantic Retrieval | 2/2 | 45m | Verified |
| 4. Local LLM & Grounded Generation | 2/2 | 40m | Verified |
| 5. Streamlit Application | 2/2 | 35m | Verified |
| 6. Evaluation and Release | 2/2 | 30m | Verified |

**Recent Trend:**
- Regression test suite pass rate: 100% (139 passed, 0 failed across isolated test runs: 128 sandbox unit tests + 11 host Streamlit AppTests)
- Multi-PDF E2E acceptance suite: 100% PASS (BASWE, Everything RAG, FINZAA, distractor questions, safe reset invariants)
- Citation emission & validity: 100%
- Hallucination / Unsupported answer rate: 0.0%
- Fast-path distractor rejection latency: 9–16 ms (0 tokens generated)
- Cold FAISS snapshot hydration: 173 ms (direct from disk)
- Active inference peak RAM: 1.55 GB (comfortable inside 16 GB laptop limit)

## Accumulated Context

### Decisions

Decisions are logged in `PROJECT.md` Key Decisions table:
- **Local-first CPU architecture:** 100% offline, ₹0 budget, complete data privacy, reproducible for technical interviews.
- **WP-02 Audit as Phase 1:** Verified and approved with 42/42 passing tests.
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, symmetric encoding, 18.5 ms CPU latency, Apache 2.0).
- **Vector store:** `faiss-cpu` (`IndexFlatIP`) paired with immutable versioned snapshots (`snapshots/snapshot_*`) and atomic pointer (`active_snapshot.json`).
- **Crash consistency:** Staged snapshotting with `manifest.json` commit sentinel and SHA-256 checksums; fallback recovery on corrupted active snapshot.
- **Strict offline isolation:** `EmbeddingConfig(local_files_only=True, allow_download=False)`; explicit model provisioning via `scripts/download_model.py`.
- **Local LLM Baseline:** `Qwen/Qwen2.5-0.5B-Instruct` on CPU via `transformers` with `local_files_only=True`, pluggable for future larger models.
- **3-Layer Citation Integrity:** Layer 1 syntax regex, Layer 2 metadata linkage to `ContextReference`, Layer 3 evidence factual overlap check.
- **Dual-Tier Abstention:** Instant fast-path on empty retrieval/context + model-level phrase detection + conservative fallback on unsupported claims.
- **Deterministic Citation Token Contract:** Model instructed to emit `[S1]`, `[S2]`; application metadata layer resolves tokens to physical document and page spans.
- **Actual Hardware Profile:** Measured physical RAM is 15.65 GB (10 physical / 12 logical CPU cores). Peak active RSS is 1.55 GB.
- **FINZAA Chunk Count Reconciled:** 241 chunks (118 pages), matching 436 total vectors in the FAISS index.

### Milestone 1 Status

**MILESTONE 1 (DOCUMIND MVP): CLOSED & APPROVED FOR RELEASE**

### Deferred Items

Carried forward to Milestone 2 (Advanced DocuMind):
- llama.cpp / GGUF 4-bit quantization for 15-20 tok/sec on CPU
- Optional Tesseract OCR preprocessing for scanned / image-only PDF pages
- BM25 and hybrid dense-sparse retrieval
- Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking (BGE-Reranker-Base)
- Query rewriting / expansion / HyDE
- Multi-document comparison & conversational memory
