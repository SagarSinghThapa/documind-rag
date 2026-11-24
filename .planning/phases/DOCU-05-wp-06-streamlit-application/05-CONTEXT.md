# Phase 5: WP-06 Streamlit Application - Context

**Gathered:** 2026-11-23  
**Status:** Ready for execution (Refined with 15 Senior Full-Stack, Streamlit Architecture & RAG Safety Safeguards)  
**Source:** User Requirements, WP-01 Pipeline Specification, Phase 1–4 Verified Architecture, ₹0 Budget & Offline Constraints

<domain>
## Phase Boundary

Phase 5 designs and implements the interactive web application for DocuMind using Streamlit. It integrates the four previously verified subsystems:
1. Document ingestion and chunking (`IngestionPipeline`, `PDFLoader`, `RecursiveCharacterChunker`) from Phase 1.
2. Local embeddings and FAISS vector indexing (`LocalEmbedder`, `FaissVectorStore`, `DocumentIndex`) from Phase 2.
3. Semantic retrieval and context assembly (`SemanticRetriever`, `ContextBuilder`) from Phase 3.
4. Grounded answer generation, 3-layer citation verification, and dual-tier abstention (`GroundedGenerator`, `TransformersLLM`, `CitationParser`) from Phase 4.

Phase 5 is **user interface, lifecycle persistence, and end-to-end interactive workflow orchestration**. It provides multi-PDF upload, real-time indexing status, conversational Q&A, visual citation inspection via expandable accordions, document management controls, and defensive error handling.

This phase does NOT alter the core ingestion or generation algorithms. Large-scale benchmark suites, release documentation, and portfolio demonstration runbooks belong to Phase 6 (WP-07).

</domain>

<decisions>
## Implementation Decisions & Production Safeguards

### 1. Application Architecture & Localhost Boundary (Safeguards 4, 15)
- **Localhost Only:** Configured in `.streamlit/config.toml` (`server.address = "localhost"`, `server.port = 8501`, `browser.gatherUsageStats = false`). Strictly zero external telemetry beacons, zero cloud inference, and zero paid APIs.
- **UI Component Modularization:**
  - `src/documind/app.py`: Main application entry point executable via `streamlit run src/documind/app.py`.
  - `src/documind/ui/state.py`: Session state initialization, cached resource resolution, and thread-safety locks.
  - `src/documind/ui/ingestion_manager.py`: File upload quarantine, SHA-256 deduplication, and index persistence.
  - `src/documind/ui/components/sidebar.py`: Upload widget, active document summary table, two-step confirmed index reset, and live CPU/RAM telemetry.
  - `src/documind/ui/components/chat.py`: Message history rendering, sample query pills, dockable chat input, and verified answer presentation.
  - `src/documind/ui/components/citations.py`: Expandable citation accordions, evidence snippet formatting, and 3-layer verification badges.
- **Dependency Management:** Add `streamlit>=1.35.0` to `requirements.txt`.

### 2. Model Caching & Session Isolation Strategy (Safeguards 4, 5)
- **Global Cached Resources (`@st.cache_resource`):**
  - Read-only neural network weights (`LocalEmbedder`, `TransformersLLM`) are cached globally to eliminate redundant disk reads (~1.8s) and duplicate RAM allocations (~1.5 GB).
  - CPU inference is serialized using a threading lock (`threading.Lock()`) to guarantee thread safety during model execution.
- **Strict Session Isolation (`st.session_state`):**
  - To prevent cross-session data leakage and allow per-session lifecycle control:
    - `st.session_state.document_index`: The active FAISS vector index and metadata records.
    - `st.session_state.indexed_docs`: Registry of uploaded documents.
    - `st.session_state.messages`: Conversational chat history.
    - `st.session_state.retriever` and `st.session_state.generator`: Retrieval and generation instances bound to the active session index.
    - `st.session_state.error_notice`: Transient UI error alerts.

### 3. Durable Index Lifecycle & Disk Persistence (Safeguard 6)
- **Persistence Across Reruns & Restarts:**
  - On application startup, `init_session_state()` checks if an active index exists at `data/index/active_snapshot.json` and `data/index/document_registry.json`. If found, it automatically restores the active index and document registry.
  - When new files are ingested, `IngestionManager` flushes the updated vector store snapshot to `data/index/snapshots/` and commits `active_snapshot.json`.
  - State survives both Streamlit UI reruns (via `st.session_state`) and full process restarts (via disk snapshot restoration).

### 4. Multi-PDF Upload, Deduplication & Edge-Case Handling (Safeguards 7, 9)
- **Upload Constraints & Filename Sanitization:**
  - 50 MB per-file upload limit (`max_file_size_bytes = 50 * 1024 * 1024`).
  - Filename sanitization via `pathlib.Path(name).name` and safe character filtering to prevent path traversal attacks (`../`, `..\\`).
  - Quarantine: Saved safely to `data/uploads/{file_hash[:12]}_{safe_filename}`. Temporary files are cleaned up on failure.
- **Two-Level Safe Duplicate Detection:**
  - Level 1 (Document Hash): Computes SHA-256 hash of incoming bytes before extraction. If hash exists in registry, skips ingestion with an informational toast.
  - Level 2 (Chunk ID): `DocumentIndex.add_chunks` enforces intra-batch and persistent store chunk ID deduplication.
- **Graceful PDF Edge-Case Handling:**
  - Encrypted / Password-protected PDFs: Caught by `PDFLoader`, surfaces non-fatal warning card (`"Cannot read encrypted PDF '{name}'. Skipping."`).
  - 0-Byte or Corrupt Header PDFs: Caught by `PDFLoader.validate_pdf()`, surfaces non-fatal error card.
  - Scanned Image-Only PDFs: Detected when extractable text character count is 0 across all pages. Surfaces informative notice explaining that OCR is scheduled for Milestone 2.

### 5. Safe Reset Behavior (Safeguard 8)
- **Two-Step Explicit Confirmation:**
  - A confirmation checkbox (`st.checkbox("Confirm permanent deletion of indexed vectors")`) must be checked before the "Clear Index & Reset" button is enabled.
- **Strict Deletion Scope:**
  - Clears in-memory vectors (`_index.reset()`, `_records.clear()`, `_chunk_id_set.clear()`).
  - Clears `st.session_state.indexed_docs` and `st.session_state.messages`.
  - Deletes persisted files strictly in `data/uploads/` and `data/index/snapshots/`.
  - **NON-NEGOTIABLE SAFETY GUARANTEE:** NEVER deletes `models/` (model weights), `research/` (technical references), or codebase files.

### 6. Verified Answer Rendering Strategy (Safeguards 1, 2, 3, 10, 11)
- **Synchronous Verified Generation as Primary Default:**
  - While `TransformersLLM.stream_generate` exists, it has only been tested for 1 token in profiling. Token streaming emits provisional, unverified text before citation resolution is possible.
  - To prevent displaying unvalidated hallucinated output as verified truth, **synchronous verified generation via `GroundedGenerator.generate_answer()` is the primary, default, and guaranteed rendering path for MVP**.
  - Generation occurs behind a clean spinner: `st.spinner("Retrieving evidence & generating grounded answer...")`.
- **Provisional Streaming Distinction (If Optional Streaming is Toggled):**
  - If streaming is enabled by the user in settings, the streaming container must be explicitly labeled *"Generating provisional draft..."*.
  - Only AFTER generation completes and `CitationParser` validates citations against `ContextReference` is the provisional container replaced with the finalized verified answer card.
- **3-Layer Citation Verification Display:**
  - Citations preserve deterministic `[S1]`, `[S2]` source IDs, mapped filenames, physical page numbers, and chunk text excerpts.
  - Accordions display 3-layer verification badges:
    - Syntax Validity: Valid `[S1]` format ✅
    - Metadata Linkage: Mapped to `ContextReference` ✅
    - Evidence Support: Claim terms confirmed in chunk text ✅
- **Strict Grounding Status Display:**
  - `verified`: Green badge (`Verified Grounding ✅`).
  - `partially_supported`: Amber badge (`Partially Supported ℹ️`) with explicit warning.
  - `unsupported`: Orange warning callout (`Unsupported Claims Intercepted ⚠️`).
  - `abstained`: Neutral callout (`Abstained (No Evidence) ℹ️`).
  - **Rule:** Unsupported or partially supported content is NEVER displayed with a green verified badge.
- **Fast-Path Abstention:**
  - Renders instantly in 8–12 ms with 0 LLM tokens generated when 0 chunks exceed the similarity threshold.

### 7. Automated and Manual Testing Strategy (Safeguards 12, 13, 14)
- **Headless Automated Testing via `AppTest`:**
  - Uses `streamlit.testing.v1.AppTest` with `MockLLM` in `tests/test_ui_app.py` to test rendering, input submission, citation display, abstention rendering, and error handling without launching a browser or opening network ports.
  - **No Arbitrary Timing Assertions:** Tests assert functional correctness, state mutations, and element presence without fragile sub-second latency assertions.
- **Manual Real-Model Acceptance Testing:**
  - Documented in `docs/runbooks/PHASE_5_STREAMLIT_ACCEPTANCE.md` covering real `Qwen2.5-0.5B-Instruct` execution on `localhost:8501` across multi-PDF upload, factual queries, fast-path abstention, and safe reset.
- **Regression Safety:**
  - Zero regressions across the 116 existing automated tests.

</decisions>

<canonical_refs>
## Canonical References

- `src/documind/ingestion/pipeline.py` — `IngestionPipeline` for multi-PDF chunking and reporting.
- `src/documind/storage/index.py` — `DocumentIndex` for vector insertion and persistence.
- `src/documind/storage/vector_store.py` — `FaissVectorStore` snapshot persistence and crash-recovery.
- `src/documind/retrieval/retriever.py` — `SemanticRetriever` for Top-K and threshold filtering.
- `src/documind/retrieval/context_builder.py` — `ContextBuilder` for citation-grounded context windows.
- `src/documind/generation/llm.py` — `TransformersLLM` and `MockLLM` for token generation.
- `src/documind/generation/generator.py` — `GroundedGenerator` for dual-tier abstention and answer production.
- `src/documind/generation/citation.py` — `CitationParser` for `[S1]` source ID resolution and verification.
- `docs/architecture/BASELINE_RAG_SPECIFICATION.md` — Stage 1 (PDF Upload) & Stage 10 (Grounded Generation).
- `docs/project_reports/WP_05_REAL_LLM_VERIFICATION.md` — Verified Phase 4 real-model performance baseline.

</canonical_refs>
