# Phase 2: WP-03 Local Embeddings & Persistent Vector Index - Context

**Gathered:** 2026-11-22  
**Status:** Ready for execution (Refined with 10 production safeguards)  
**Source:** User Requirements, Phase 1 Audit Evidence, Technical Research & Robustness Safeguards

<domain>
## Phase Boundary

Phase 2 designs and implements the local embedding generation and persistent vector indexing subsystem for DocuMind. It takes normalized `TextChunk` records from Phase 1 (WP-02) and produces dense vector representations stored in a persistent local FAISS index alongside complete document and page metadata.

Query transformation, BM25 hybrid fusion, and LLM text generation belong to Phase 3 (WP-04) and Phase 4 (WP-05) and are strictly out of scope for Phase 2.

</domain>

<decisions>
## Implementation Decisions

### 1. Embedding Model Selection & Strict Offline Operation
- **Selected model:** `sentence-transformers/all-MiniLM-L6-v2`.
- **Rationale:** Lowest CPU latency (~10–15 ms per chunk on Intel i5), 384-dimensional dense vectors, lightweight RAM footprint (<150 MB), Apache 2.0 license, and symmetric encoding (zero risk of missing-prefix degradation bugs).
- **Offline enforcement (Safeguard 3):** Runtime inference defaults to `local_files_only=True` and `allow_download=False`. If model weights are missing from the local cache, `LocalEmbedder` raises `ModelNotFoundError` detailing how to run `python scripts/download_model.py`. The runtime will never make silent outbound HTTP requests.
- **Abstraction:** Implemented behind an abstract `BaseEmbedder` base class in `src/documind/storage/embedder.py`.

### 2. Vector Indexing Backend Selection
- **Selected backend:** `faiss-cpu` paired with an atomic JSON metadata sidecar and index manifest.
- **Rationale:** High performance C++ SIMD execution on Windows CPU, zero background daemons, zero telemetry (unlike Chroma's PostHog integration), minimal dependencies (only `numpy`), and sub-millisecond vector lookup.
- **Index type:** `faiss.IndexFlatIP` (Exact Inner Product over L2-normalized vectors = exact cosine similarity).

### 3. Metadata Preservation & Invariant Schemas
- Preserve all 12 existing fields from WP-02 `TextChunk`:
  - `document_id`: SHA-256 hash of source PDF.
  - `chunk_id`: Deterministic unique chunk string.
  - `document_name`: Source PDF filename.
  - `source_path`: Source file path.
  - `chunk_index`: 0-indexed position within document.
  - `page_numbers`: Exact list of contributing 1-indexed pages.
  - `start_page`: First contributing page.
  - `end_page`: Final contributing page.
  - `char_start`: Character start in unified document text.
  - `char_end`: Character end in unified document text.
  - `text`: Plain text content.
  - `metadata`: Extensible metadata dictionary.
- Schemas defined in `src/documind/storage/schemas.py`:
  - `VectorRecord`: Holds all 12 attributes + optional vector list.
  - `IndexManifest`: Tracks model identity, dimension, metric, normalization policy, total counts, timestamps, commit status, and checksums.
  - `SearchResult`: Holds score and `VectorRecord`.
  - `IndexingResult`: Telemetry for batch ingestion (total received, newly indexed, intra-batch duplicates skipped, existing duplicates skipped, empty chunks skipped).

### 4. Index Lifecycle, Crash Consistency & Defensive Loading
- **Create Index:** Initializes FAISS index with model dimensions and inner-product metric.
- **Two-Level Deduplication & Empty Filtering (Safeguards 4, 6):**
  - Detects and drops empty or whitespace-only chunks (never indexes zero or NaN vectors).
  - Drops duplicate `chunk_id`s within the incoming batch (`seen_in_batch`).
  - Drops duplicate `chunk_id`s already existing in the vector store (`vector_store.contains_chunk()`).
- **Vector Validation (Safeguard 5):**
  - Enforces `dtype=np.float32`, C-contiguity, shape `(N, dim)`, finiteness (no `NaN` or `Inf`), and Euclidean norm $\approx 1.0 \pm 1e-3$ before touching FAISS C++ memory.
- **Crash-Consistent Snapshot Strategy (Safeguard 1):**
  - Writes `index.faiss` and `metadata.json` into an isolated staging folder (`.snapshot_staging_<token>`).
  - Computes file hashes and writes `manifest.json` with commit sentinel `status: "committed"` and checksums.
  - Promotes staging files to the target directory with atomic replacements; `manifest.json` is replaced last as the commit marker.
  - If a process crashes mid-save, the existing committed index is untouched.
  - Explicit limitations documented: On Windows, active open file handles prevent replacements; crash consistency protects against process crashes and partial writes via the commit sentinel and staging isolation.
- **Defensive Load Verification (Safeguard 2, 9):**
  - Requires all 3 files (`manifest.json`, `index.faiss`, `metadata.json`).
  - Verifies `manifest.status == "committed"`, rejecting uncommitted or corrupted snapshots with `IndexCorruptionError`.
  - Validates `model_name`, `dimension`, `metric`, and `normalize_embeddings` against expected values, raising `IncompatibleIndexError` on any mismatch.
  - Validates FAISS vector count against metadata sidecar record count.
- **Empty Index & Empty Query Handling (Safeguard 6, 9):**
  - Empty search queries raise `ValueError`.
  - Searching an empty index returns `[]` immediately without throwing errors.

### 5. Numerical Tolerance & Decoupled Benchmarks (Safeguards 7, 8)
- Unit tests use `np.allclose(a, b, rtol=1e-5, atol=1e-6)` and `pytest.approx` for floating-point comparisons rather than fragile bitwise equality.
- Hardware-dependent CPU latency thresholds are decoupled into a dedicated benchmark script (`scripts/benchmark_vector_store.py`) reporting throughput (chunks/sec), batch latency (ms), search latency (ms, p95), and RAM RSS delta, rather than acting as brittle unit test failure gates.

### 6. Non-Negotiable Rules
- Preserve all 42 passing tests from WP-02 with zero regressions (Safeguard 10).
- No package installations or code modifications until execution is explicitly approved.
- Do not create Git commits or push changes without approval.
- Do not automatically advance to Phase 3.

</decisions>

<canonical_refs>
## Canonical References

- `src/documind/ingestion/schemas.py` — Source `TextChunk` and `DocumentPage` data models.
- `src/documind/config.py` — Ingestion and pipeline configuration parameters.
- `docs/architecture/BASELINE_RAG_SPECIFICATION.md` — Stage 5 (Embeddings) & Stage 6 (Vector Storage) architecture specifications.
- `docs/project_reports/WP_02_INDEPENDENT_AUDIT_REPORT.md` — Verified Phase 1 baseline and test suite evidence.
- `.planning/phases/DOCU-02-wp-03-local-embeddings-and-persistent-vector-index/02-RESEARCH.md` — Detailed technical evaluation.

</canonical_refs>

<specifics>
## Specific Ideas

- Provide a high-level `DocumentIndex` orchestrator combining `LocalEmbedder` and `FaissVectorStore` for clean integration with `IngestionPipeline`.
- Support batch vector insertion (`batch_size=32`) to optimize CPU vectorized operations.
- Include structured telemetry in `IndexingResult` to give callers full visibility into skipped duplicates and empty chunks.

</specifics>

<deferred>
## Deferred Ideas

- Hybrid retrieval with BM25 — Deferred to Milestone 2.
- Cross-encoder reranking — Deferred to Milestone 2.
- Asymmetric query instructions (`bge-small` / `e5`) — Handled in Phase 3 if alternative models are enabled.

</deferred>
