# Phase 3: WP-04 Semantic Retrieval - Context

**Gathered:** 2026-11-22  
**Status:** Ready for execution (Refined with 10 evaluation, latency & untrusted-text safeguards)  
**Source:** User Requirements, WP-02 & WP-03 System Architecture, Technical Research & Local-First Constraints

<domain>
## Phase Boundary

Phase 3 designs and implements the semantic retrieval subsystem for DocuMind. It connects natural-language user queries to the local `DocumentIndex` and `FaissVectorStore` built in Phase 2, retrieves relevant text chunks with similarity scores, supports both an unthresholded baseline and candidate threshold filtering, constructs citation-rich context windows while treating document text as untrusted content, and measures retrieval effectiveness against an evidence-based evaluation dataset.

This phase is **retrieval only**. LLM text generation, prompt templating with LLMs, Streamlit UI, and advanced multi-hop RAG belong to Phase 4 (WP-05), Phase 5 (WP-06), and Phase 6 (WP-07) and are strictly out of scope for Phase 3.

</domain>

<decisions>
## Implementation Decisions

### 1. Retrieval Architecture & Component Separation
- **Package location:** `src/documind/retrieval/`.
- **Core Components:**
  - `SemanticRetriever` (`src/documind/retrieval/retriever.py`): Encapsulates query embedding generation, index querying, and score filtering behind an abstract `BaseRetriever` interface.
  - `ContextBuilder` (`src/documind/retrieval/context_builder.py`): Formats retrieved chunks into clean, citation-ready context blocks with explicit document/page boundaries, enforces character/token budgets, eliminates verified overlapping chunk redundancy, and sandboxes untrusted text.
  - `RetrievalEvaluator` (`src/documind/retrieval/evaluator.py`): Computes formal IR metrics (Recall@K, Precision@K, MRR, NRR, latency percentiles, attribution accuracy) across separate positive and negative evaluation splits.
- **Query Embedding:** Reuses `LocalEmbedder` from Phase 2 with strict `local_files_only=True` to ensure identical 384-dimensional vector space and zero runtime network access.

### 2. Relevance Score Filtering & Threshold Calibration Policy (Safeguard 1)
- **Metric:** Cosine similarity via `faiss.IndexFlatIP` on L2-normalized embeddings ($S_C \in [-1.0, 1.0]$).
- **Baseline vs Candidate Threshold:**
  - `similarity_threshold=None` provides a pure unthresholded Top-K baseline returning all nearest neighbors.
  - `similarity_threshold=0.35` is treated strictly as an **initial exploratory candidate**, not a pre-validated optimal threshold.
  - Optimal threshold is calibrated empirically via `scripts/evaluate_retrieval.py` using a parameter sweep over $\tau \in [0.10, 0.60]$ on the calibration split.
- **Empty & Edge Case Handling:**
  - Empty or whitespace-only queries raise `ValueError`.
  - Searches against an empty index or queries where all chunks fall below the threshold return empty `RetrievalResult` objects without crashing.

### 3. Stage-by-Stage Latency Profiling (Safeguard 2)
- Disaggregates retrieval latency into:
  - `embed_latency_ms`: Pure query embedding time via `LocalEmbedder`.
  - `search_latency_ms`: Pure FAISS inner-product nearest-neighbor lookup time.
  - `context_build_latency_ms`: Pure context formatting and overlap deduplication time.
  - `total_latency_ms`: End-to-end pipeline latency.
- Evaluator reports **mean, median, and P95** for all stages.
- Hardware-dependent latency pass/fail thresholds are strictly decoupled from unit tests (avoiding flaky unit-test failures) and placed into evaluation benchmarks.

### 4. Context Construction, Verified Deduplication & Untrusted Content (Safeguards 5, 7)
- **Dynamic Overlap Verification:** Never assumes fixed 150-char overlap. Suffixes of chunk $i$ and prefixes of chunk $i+1$ are dynamically scanned; text is deduplicated only when a matching substring $\ge \text{min\_overlap\_chars}$ (20 chars) is verified.
- **Untrusted Text Sandboxing:** Retrieved document text is treated as untrusted external content. Chunks are encapsulated in structural blockquotes:
  ```markdown
  ### Source: {document_name} | Physical Page(s): {page_span} | Ref ID: {chunk_id}
  > [Document Content - Untrusted Context]
  > {escaped_chunk_text}
  ```
- **Budget Enforcement:** Configurable `max_context_chars` (default: 4000 chars $\approx 1000$ tokens). Chunks are packed in rank order until the budget limit is reached without splitting words.
- **Structured References:** Emits structured `ContextReference` records containing all 12 metadata attributes for Phase 4 citation generation.

### 5. Ground-Truth Dataset & Leakage Prevention (Safeguards 3, 4, 6)
- **Verified Ground Truth:** All 20 evaluation queries are grounded in physical pages and verbatim text excerpts from:
  - `research/technical_references/BASWE-15-RAG-Concepts-Guide.pdf` (14 pages)
  - `research/technical_references/Everything-to-know-about-RAG.pdf` (92 pages)
- **Leakage Prevention Split:**
  - **Calibration Set (Split A - 10 queries: 6 pos, 4 neg):** Used exclusively for threshold sweeps and tuning.
  - **Evaluation Set (Split B - 10 queries: 6 pos, 4 neg):** Held out for unbiased evaluation reporting.
- **Separate Positive/Negative Metrics:**
  - `Recall@K`: Computed strictly on positive queries ($Q_+$) where $|\text{Relevant}| > 0$.
  - `Precision@K`: Computed on positive queries ($Q_+$).
  - `MRR`: Computed on positive queries ($Q_+$) (0.0 if not found in top K).
  - `Negative Rejection Rate (NRR)`: Fraction of negative queries ($Q_-$) returning 0 chunks above threshold.
  - `False Positive Rate (FPR)`: $1.0 - \text{NRR}$.

### 6. Non-Negotiable Safeguards
- Zero regressions across existing 68 tests (Phase 1: 42 tests, Phase 2: 26 tests).
- Pure local CPU execution — no external APIs, cloud services, or paid keys.
- Strictly offline runtime execution (`local_files_only=True`).
- No modifications to production code during planning.
- Stop and await approval before implementing.

</decisions>

<canonical_refs>
## Canonical References

- `src/documind/storage/schemas.py` — `VectorRecord`, `SearchResult`, `IndexManifest`.
- `src/documind/storage/embedder.py` — `LocalEmbedder`, `validate_vectors()`.
- `src/documind/storage/vector_store.py` — `FaissVectorStore`, `IndexFlatIP`.
- `src/documind/storage/index.py` — `DocumentIndex`.
- `src/documind/ingestion/schemas.py` — `TextChunk`.
- `research/technical_references/BASWE-15-RAG-Concepts-Guide.pdf` — Evaluation source PDF 1.
- `research/technical_references/Everything-to-know-about-RAG.pdf` — Evaluation source PDF 2.
- `docs/project_reports/WP_03_INDEPENDENT_AUDIT.md` — Phase 2 audit evidence and baseline.

</canonical_refs>
