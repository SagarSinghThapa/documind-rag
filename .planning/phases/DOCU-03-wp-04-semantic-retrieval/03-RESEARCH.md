# Phase 3: WP-04 Semantic Retrieval - Research & Architecture Analysis

**Auditor/Architect:** Senior RAG Retrieval Engineer & Evaluation Architect  
**Date:** 2026-11-22  
**Target:** WP-04 Semantic Retrieval Subsystem (Refined with 10 Production Safeguards)

---

## 1. Analysis of Existing Ingestion & Storage Contracts

The semantic retrieval layer sits directly on top of the components implemented in Phase 1 (WP-02) and Phase 2 (WP-03):

```
User Query (str)
       │
       ▼
┌───────────────────────────┐
│     SemanticRetriever     │
│  - Query Validation       │
│  - LocalEmbedder (384-d)  │
│  - FaissVectorStore search│
│  - Baseline / Threshold   │
└─────────────┬─────────────┘
              │ List[RetrievedChunk]
              ▼
┌───────────────────────────┐
│      ContextBuilder       │
│  - Verified Overlap Dedup │
│  - Untrusted Text Guard   │
│  - Token/Char Budget      │
│  - Citation Header Format │
└─────────────┬─────────────┘
              │ FormattedContext
              ▼
 downstream LLM (Phase 4)
```

### Existing Building Blocks
1. **`LocalEmbedder` (`src/documind/storage/embedder.py`):**
   - Model: `sentence-transformers/all-MiniLM-L6-v2`.
   - Dimension: strictly 384, `float32`, finite, L2-normalized.
   - Safety: `local_files_only=True`, strictly offline CPU execution.
   - Latency: ~10–18 ms per single query on CPU.
2. **`FaissVectorStore` (`src/documind/storage/vector_store.py`):**
   - Index type: `faiss.IndexFlatIP(384)`.
   - Exact inner product on L2-normalized vectors = **exact cosine similarity**:
     $$\text{similarity} = \mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{384} u_i v_i \in [-1.0, 1.0]$$
   - Metadata alignment: Every vector has an aligned `VectorRecord` containing all 12 metadata fields.
3. **`DocumentIndex` (`src/documind/storage/index.py`):**
   - Provides `search(query: str, top_k: int = 5) -> List[SearchResult]`.
   - Each `SearchResult` yields `score: float` and `record: VectorRecord`.

---

## 2. Relevance Thresholding: Baseline vs. Calibration

### Baseline vs. Candidate Filtering
- **No-Threshold Baseline (`similarity_threshold=None`):** Standard vector search returns the top $K$ nearest neighbors unconditionally. This provides an essential reference baseline for measuring the impact of filtering.
- **Candidate Filtering (`similarity_threshold=\tau`):** Cosine score threshold $\tau \in [-1.0, 1.0]$ filters out candidate chunks below $\tau$.
- **Exploratory Candidate ($\tau = 0.35$):** $0.35$ is treated strictly as an initial candidate, not a pre-validated optimal value. Real optimal threshold must be determined via calibration.

### Empirical Calibration Methodology
1. **Calibration vs. Evaluation Split:** To prevent evaluation-data leakage, the 20-query ground-truth dataset is partitioned into:
   - **Split A — Calibration Set (10 queries: 6 positive, 4 negative):** Used exclusively to sweep $\tau \in [0.10, 0.60]$ (step 0.05).
   - **Split B — Holdout Evaluation Set (10 queries: 6 positive, 4 negative):** Held out for unbiased evaluation of final retrieval performance under $\tau^*$.
2. **Optimization Criterion:**
   $$\tau^* = \arg\max_\tau F_1(\tau) \quad \text{subject to } \text{NRR}(\tau) \ge 0.75$$
   Where $F_1 = 2 \cdot \frac{\text{Precision@}K \cdot \text{Recall@}K}{\text{Precision@}K + \text{Recall@}K}$.

---

## 3. Disaggregated Latency Profiling

Retrieval latency must be measured at each distinct pipeline stage:
- $T_{\text{embed}}$: Query vectorization using `LocalEmbedder`.
- $T_{\text{search}}$: FAISS inner-product nearest-neighbor lookup.
- $T_{\text{context}}$: Context formatting, overlap verification, and budget packing.
- $T_{\text{total}}$: Full end-to-end retrieval latency ($T_{\text{embed}} + T_{\text{search}} + T_{\text{context}}$).

**Statistical Reporting:**
For each stage, the evaluation pipeline computes:
- **Mean Latency**
- **Median (P50) Latency**
- **95th Percentile (P95) Latency**

**Decoupled Verification:** Hardware-dependent latency targets (e.g., $P95 \le 45$ ms) are measured in benchmark scripts (`scripts/evaluate_retrieval.py`), preventing flaky test failures on resource-constrained environments.

---

## 4. Verified Overlap Deduplication & Untrusted Text Guard

### Verified Overlap Deduplication
In Phase 1, recursive chunking applied a nominal 150-character overlap. However, depending on paragraph and sentence boundaries, the actual overlap between Chunk $i$ and Chunk $i+1$ can vary or be 0.
- **Algorithm:**
  1. For consecutive chunks from the same document (`curr.document_id == prev.document_id` and `curr.chunk_index == prev.chunk_index + 1`):
  2. Compute maximal common suffix of `prev.text` and prefix of `curr.text`:
     $$M = \text{longest\_matching\_substring}(\text{prev.text}[-K:], \text{curr.text}[:K])$$
  3. Deduplicate **only if** $|M| \ge \text{min\_overlap\_chars}$ (20 characters).
  4. If verified, strip $M$ from `curr.text`.
  5. If no verified match exists, keep `curr.text` intact. Never assume a fixed 150-char overlap.

### Untrusted Content Sandboxing
Document text extracted from PDFs comes from external files and cannot be trusted blindly:
- Could contain prompt injection attempts or formatting artifacts.
- `ContextBuilder` wraps all chunk content in distinct structural quote blocks:
  ```markdown
  ### Source: {document_name} | Physical Page(s): {page_span} | Ref ID: {chunk_id}
  > [Document Content - Untrusted Context]
  > {escaped_chunk_text}
  ```
- Downstream LLM system prompts in Phase 4 will instruct the model to treat content within these blocks strictly as factual reference data.
- Full metadata fidelity (`char_start`, `char_end`, `start_page`, `end_page`, `chunk_id`) is stored independently in structured `ContextReference` objects.

---

## 5. Formal Information Retrieval Evaluation Framework

Evaluation separates positive factual queries ($Q_+$) from negative distractor queries ($Q_-$):

### Positive Query Metrics ($Q_+$)
1. **Recall@K:**
   $$\text{Recall@}K = \frac{1}{|Q_+|} \sum_{q \in Q_+} \frac{|\text{Retrieved@}K(q) \cap \text{Relevant}(q)|}{|\text{Relevant}(q)|}$$
   - Denominator: total ground-truth relevant chunks for query $q$.
   - Empty result behavior: if $|\text{Retrieved@}K(q)| = 0$, $\text{Recall@}K(q) = 0.0$.
2. **Precision@K:**
   $$\text{Precision@}K = \frac{1}{|Q_+|} \sum_{q \in Q_+} \frac{|\text{Retrieved@}K(q) \cap \text{Relevant}(q)|}{K}$$
3. **Mean Reciprocal Rank (MRR):**
   $$\text{MRR} = \frac{1}{|Q_+|} \sum_{q \in Q_+} \text{RR}(q)$$
   Where $\text{RR}(q) = \frac{1}{\text{rank}_1(q)}$ if a relevant chunk appears at 1-indexed rank $\le K$; otherwise $\text{RR}(q) = 0.0$.
4. **Attribution Accuracy:**
   $$\text{Attribution Accuracy} = \frac{\text{Retrieved chunks with correct document and page matches}}{\text{Total retrieved chunks evaluated}}$$

### Negative / Distractor Query Metrics ($Q_-$)
1. **Negative Rejection Rate (NRR):**
   $$\text{NRR} = \frac{\sum_{q \in Q_-} \mathbb{I}(|\text{RetrievedAboveThreshold}(q)| = 0)}{|Q_-|}$$
   Measures the fraction of out-of-domain queries successfully returning 0 chunks.
2. **False Positive Rate (FPR):**
   $$\text{FPR} = 1.0 - \text{NRR}$$

---

## 6. Curated Ground-Truth Dataset Design

The evaluation dataset `data/evaluation/retrieval_ground_truth.json` contains 20 manually verified queries grounded in physical PDF pages:
- **6 Calibration Positives:** Grounded in `BASWE-15-RAG-Concepts-Guide.pdf` (pages 1–14).
- **4 Calibration Negatives:** Out-of-domain topics (e.g. sourdough bread baking, tire pressure).
- **6 Holdout Positives:** Grounded in `Everything-to-know-about-RAG.pdf` (pages 1–92).
- **4 Holdout Negatives:** Out-of-domain topics (e.g. corporate tax law, quantum computing).

Every positive item includes physical page numbers, exact verbatim text excerpts, and verified ground-truth answer spans. No synthetic or assumed labels are permitted.
