# Phase 5: WP-06 Streamlit Application - Technical Research & Safety Refinement

**Auditor/Architect:** Senior Python Full-Stack Engineer, Streamlit Architect, and RAG Systems Auditor  
**Date:** 2026-11-23  
**Target:** WP-06 Interactive Streamlit Application (Refined with 15 Senior Production Safeguards)

---

## 1. Inspection of LLM Runtime & Streaming Feasibility (Refinements 1, 2, 3)

### Codebase Inspection Findings
- `TransformersLLM` in `src/documind/generation/llm.py` defines `stream_generate()` using Hugging Face's `TextIteratorStreamer` and a worker thread (`threading.Thread(target=self._model.generate, ...)`).
- **Testing Inventory:** In `tests/test_llm.py`, only `MockLLM.stream_generate` is tested (`test_mock_llm_streaming`). `TransformersLLM.stream_generate` has **not been tested in automated test suites** with the real model (only tested manually for a 1-token probe in `scripts/verify_real_llm.py:measure_ttft`).
- **Grounding Integrity Bottleneck:** Citation extraction, source ID mapping (`[S1]`), and 3-layer verification (`CitationParser`) can ONLY execute after the **entire completion text** is available.
- **Architectural Decision:**
  1. **Primary MVP Path:** Synchronous verified answer generation via `GroundedGenerator.generate_answer()` is the **primary, default, and guaranteed rendering path for MVP**. The user sees a clean status spinner, and the final answer is rendered only after 100% of grounding and citation verification is complete.
  2. **Provisional Streaming Distinction:** If token streaming is optionally enabled in settings, streaming text must be explicitly badged and labeled as *"Generating provisional draft..."*. It is strictly provisional until generation completes, citations are resolved against `ContextReference`, and verified badges are rendered. Under no circumstances is streaming output presented as verified truth before verification completes.

---

## 2. Model Caching, Thread Safety, and Session State Isolation (Refinements 4, 5)

### Streamlit Concurrency & State Analysis
In Streamlit:
- `@st.cache_resource` creates a global singleton shared across all browser sessions and threads.
- `st.session_state` is private to each browser tab/session.

### Potential Failure Modes & Safeguards:
1. **Cross-Session Data Leakage:** If `DocumentIndex`, `FaissVectorStore`, or uploaded documents were stored in `@st.cache_resource`, Session B would see Session A's private documents, violating the core local privacy guarantee.
   - *Safeguard:* `DocumentIndex`, `indexed_docs`, `messages`, `retriever`, and `generator` are **strictly placed in `st.session_state`** (per-session isolation).
2. **CPU Thread Safety of Cached Models:** PyTorch models on CPU are largely thread-safe for forward passes with `torch.inference_mode()`, but concurrent generation requests on an 8-core CPU can cause CPU thrashing or race conditions in internal KV caches.
   - *Safeguard:* In `src/documind/ui/state.py`, access to cached model forward inference is guarded by a standard Python `threading.Lock()` to ensure clean serial execution on CPU.
3. **Model Weight Caching:** `LocalEmbedder` and `TransformersLLM` instances are cached globally via `@st.cache_resource` so that heavy model weights (~1.5 GB RAM) are loaded once and never re-read from disk on UI reruns.

---

## 3. Durable Index Lifecycle & Disk Persistence (Refinement 6)

### Lifecycle Across UI Reruns and Application Restarts
- **Streamlit Reruns:** On every button click or query submission, Streamlit reruns the script from line 1. `st.session_state` preserves the active `DocumentIndex`, `indexed_docs`, and `messages` seamlessly.
- **Application Restarts:** When the user stops and restarts `streamlit run src/documind/app.py`:
  - `init_session_state()` checks whether `data/index/active_snapshot.json` and `data/index/document_registry.json` exist.
  - If present, it automatically calls `document_index.vector_store.load("data/index")` and hydrates `st.session_state.indexed_docs` from `document_registry.json`.
  - When new documents are uploaded, `IngestionManager` automatically commits a new snapshot via `document_index.vector_store.save("data/index")` and updates `document_registry.json`.

---

## 4. Multi-PDF Upload, Duplicate Prevention & File Handling (Refinements 7, 9)

### Upload Safety Matrix

| Guardrail | Implementation | Protection |
| :--- | :--- | :--- |
| **Upload Size Limit** | Enforce `max_file_size_bytes = 50 * 1024 * 1024` (50 MB) | Prevents memory exhaustion from giant PDFs. |
| **Filename Sanitization** | `pathlib.Path(name).name` + safe regex filter | Prevents directory traversal attacks (`../`, `..\\`). |
| **Quarantine Storage** | Write to `data/uploads/{file_hash[:12]}_{safe_filename}` | Safe filesystem isolation; orphan files cleaned on error. |
| **Duplicate Prevention (Doc Level)** | Compute SHA-256 hash of byte buffer before parsing | If hash exists in registry, skip parsing with informational notice. |
| **Duplicate Prevention (Chunk Level)** | `DocumentIndex.add_chunks()` 2-level deduplication | Filters duplicate chunks intra-batch and against vector store. |
| **Encrypted / Password PDFs** | Caught by `PDFLoader` (`is_encrypted=True`) | Emits non-fatal warning card; session continues unharmed. |
| **Corrupt / 0-Byte PDFs** | Caught by `PDFLoader.validate_pdf()` | Emits non-fatal error card; session continues unharmed. |
| **Scanned Image-Only PDFs** | Checked via `extractable_characters == 0` | Emits notice explaining that OCR is scheduled for Milestone 2. |

---

## 5. Safe Index Reset Behavior (Refinement 8)

Accidental deletion of indexes or collateral deletion of model files is a catastrophic failure mode in local AI tools.

### Safe Reset Policy:
1. **Explicit Two-Step Confirmation:**
   - In the sidebar, the "Clear Index & Reset" button is disabled by default.
   - It is enabled only when the user checks: `st.checkbox("Confirm permanent deletion of indexed vectors")`.
2. **Strict Deletion Scope:**
   - Clears in-memory FAISS index (`_index.reset()`, `_records.clear()`, `_chunk_id_set.clear()`).
   - Clears session state: `st.session_state.indexed_docs.clear()`, `st.session_state.messages.clear()`.
   - Clears disk persistence: deletes files in `data/uploads/` and snapshots in `data/index/snapshots/`.
   - **ABSOLUTE SAFETY INVARIANT:** Under NO circumstances touches `models/` (model weights), `research/` (technical references), or codebase files.

---

## 6. Citation Grounding & Grounding Status Visualization (Refinements 10, 11)

### Visual Presentation Standards:
- **Preserve Source Identifiers:** Emitted citations `[S1]`, `[S2]` are parsed by `CitationParser` and resolved to `ContextReference` attributes:
  - Document Name (`ref.document_name`)
  - Physical Page Span (`ref.start_page` to `ref.end_page`)
  - Chunk ID (`ref.chunk_id`)
  - Text Excerpt (`ref.text`)
- **Three-Layer Badging Inside Accordions:**
  - Syntax: `[S1] format valid ✅`
  - Metadata Linkage: `Resolved to ContextReference ✅`
  - Evidence Support: `Claim terms confirmed in chunk text ✅`
- **Strict Grounding Status Display:**
  - `verified`: Green badge (`Verified Grounding ✅`).
  - `partially_supported`: Amber badge (`Partially Supported ℹ️`).
  - `unsupported`: Orange warning callout (`Unsupported Claims Intercepted ⚠️`).
  - `abstained`: Neutral callout (`Abstained (No Conclusive Evidence) ℹ️`).
  - **Non-Negotiable Rule:** Unsupported or partially supported claims are NEVER displayed with a green verified checkmark.

---

## 7. Automated and Manual Testing Strategy (Refinements 12, 13, 14)

### Automated Headless Tests (`tests/test_ui_app.py`)
- Uses `streamlit.testing.v1.AppTest` with `MockLLM` to execute the app headlessly in Python without spinning up a live browser or opening network sockets.
- Tests verify:
  1. Initial UI render (page title, sidebar, upload widget, chat input).
  2. Empty-index query handling (triggers instant fast-path abstention with zero errors).
  3. Query submission with mock answer and citation accordion rendering.
  4. Safe reset button behavior (clears vectors without touching models).
  5. Graceful handling of empty or malformed inputs.
- **Timing Safeguard (Refinement 13):** Tests assert functional correctness, state mutations, and DOM elements without imposing arbitrary sub-second execution thresholds.
- **Zero-Regression Guarantee (Refinement 14):** All 116 existing automated tests must continue passing.

### Manual Real-Model Acceptance Runbook (`docs/runbooks/PHASE_5_STREAMLIT_ACCEPTANCE.md`)
- A comprehensive, documented human acceptance test protocol using the real local `Qwen2.5-0.5B-Instruct` model on `localhost:8501`:
  - Test 1: Uploading technical reference PDFs (`BASWE` and `Everything-to-know`) and verifying chunk counts.
  - Test 2: Asking factual query ("What is chunk overlap?") and verifying `[S1]` citation accordion, page number, and green verified badge.
  - Test 3: Asking out-of-domain query ("How to bake sourdough bread?") and verifying sub-15ms fast-path abstention.
  - Test 4: Verifying safe index reset.

---

## 8. Localhost Configuration & Zero Telemetry (Refinement 15)

Streamlit by default attempts to send anonymous usage statistics. DocuMind strictly disables this via `.streamlit/config.toml`:
```toml
[server]
address = "localhost"
port = 8501
headless = true
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false  # Disable external telemetry beacon
```
The application runs strictly on `localhost:8501` with zero external network traffic, preserving the ₹0 budget and complete local data privacy boundary.
