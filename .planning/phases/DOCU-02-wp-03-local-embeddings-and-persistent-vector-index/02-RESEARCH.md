# Phase 2: WP-03 Local Embeddings & Persistent Vector Index - Research

**Date:** 2026-11-22  
**Target Hardware:** Intel Core i5-1135G7 (4C/8T, 2.4-4.2 GHz), 24 GB RAM, No discrete GPU  
**Operating System:** Windows 11 x86_64  
**Runtime:** Python 3.11.9 (.venv)  
**Budget & Network Constraint:** ₹0 financial budget; 100% offline local inference after initial model fetch

---

## 1. Local Embedding Model Technical Evaluation

We evaluated three leading CPU-compatible open-source embedding models from the Hugging Face / MTEB ecosystem:

| Evaluation Metric | `sentence-transformers/all-MiniLM-L6-v2` | `BAAI/bge-small-en-v1.5` | `intfloat/e5-small-v2` |
| :--- | :--- | :--- | :--- |
| **Parameters** | 22.7M | 33.5M | 33.0M |
| **Embedding Dimension** | **384** | **384** | **384** |
| **Max Sequence Length** | 256 tokens (~1,000 characters) | 512 tokens (~2,000 characters) | 512 tokens (~2,000 characters) |
| **Model Size (Disk)** | **~80 MB** (PyTorch / ONNX) | ~133 MB | ~133 MB |
| **RAM Footprint (Loaded)** | **~120–150 MB** | ~200–230 MB | ~200–230 MB |
| **CPU Latency (per chunk)** | **~10–15 ms** (fastest) | ~20–30 ms | ~20–28 ms |
| **License** | Apache 2.0 (Permissive) | MIT (Permissive) | MIT (Permissive) |
| **Encoding Scheme** | **Symmetric** (raw text in, vector out) | **Asymmetric** (Requires query prefix: `"Represent this sentence for searching relevant passages: "`) | **Asymmetric** (Requires `"query: "` and `"passage: "` prefixes) |
| **Failure Modes** | Truncation past 256 tokens if chunks exceed target | Prefix omission degrades retrieval by 3–5% | Prefix omission degrades retrieval severely (disjoint vector spaces) |
| **Python 3.11 Windows Support** | Native binary wheel via PyTorch / ONNX | Native binary wheel via PyTorch / ONNX | Native binary wheel via PyTorch / ONNX |
| **Offline Operation** | `local_files_only=True` via HuggingFace Hub cache | `local_files_only=True` via HuggingFace Hub cache | `local_files_only=True` via HuggingFace Hub cache |

### Key Findings & Model Recommendation

1. **Symmetric vs. Asymmetric Complexity:**
   - Both `bge-small-en-v1.5` and `e5-small-v2` require asymmetric instruction prefixes. In production code, forgetting to prepend `"query: "` or the passage prefix breaks mathematical similarity.
   - `all-MiniLM-L6-v2` uses symmetric encoding: `model.encode(text)`. Identical vector space logic applies to both documents and search queries.

2. **Chunk Size Alignment:**
   - DocuMind's WP-02 chunking configuration uses `chunk_size = 1000` characters. At standard English densities (4–4.5 characters per token), 1,000 characters corresponds to approximately 220–250 tokens, fitting cleanly within `all-MiniLM-L6-v2`'s 256-token limit.

3. **Inference Latency on Intel i5 CPU:**
   - `all-MiniLM-L6-v2` demonstrates the lowest CPU inference latency (~10–15 ms per chunk), allowing a 100-page document (195 chunks) to be embedded in under 2.5 seconds on local CPU.

**Recommendation:**  
Default to **`sentence-transformers/all-MiniLM-L6-v2`** as the primary embedding engine for Milestone 1 MVP. The architecture encapsulates embedding generation behind an abstract `BaseEmbedder` interface so `bge-small-en-v1.5` can be configured via a single parameter without changing indexing logic.

---

## 2. Vector Database Technical Evaluation: FAISS vs. Chroma

| Architectural Metric | `FAISS` (`faiss-cpu`) | `Chroma` (`chromadb`) |
| :--- | :--- | :--- |
| **Core Architecture** | C++ optimized vector indexing library (`IndexFlatIP`, `IndexHNSWFlat`) | Embedded client/server database over SQLite + `hnswlib` |
| **Windows Compatibility** | Precompiled PyPI wheels (`faiss-cpu`) install cleanly without compilation | Installs on Windows, but frequently encounters SQLite version errors (<3.35) requiring `pysqlite3` workarounds |
| **CPU Execution Efficiency** | Direct AVX2/AVX-512 SIMD vector operations; sub-millisecond retrieval | In-process HNSW lookup via SQLite table lookups |
| **Memory Footprint** | Minimal (~20–30 MB for index structures) | Moderate (~100–180 MB due to SQLite runtime, pydantic, and background services) |
| **Persistence Mechanism** | Native binary file (`index.faiss`) + atomic JSON metadata sidecar (`metadata.json`) | Directory of SQLite files, parquet files, and header locks |
| **Metadata Preservation** | Mapped by integer vector ID to structured `VectorRecord` metadata | Stored as SQLite column attributes; limited nested metadata typing |
| **Dependency Tree & Bloat** | **Zero heavy dependencies** (only `numpy`) | **Heavy dependencies** (FastAPI, Uvicorn, PostHog, OpenTelemetry, Pydantic, Typer, DuckDB/ClickHouse) |
| **Telemetry & Privacy** | **Zero telemetry**; completely offline C++ library | Transmits anonymous usage telemetry by default (must be manually suppressed via env vars) |
| **Inspectability for Interviews**| High: Index file and human-readable metadata sidecar can be directly inspected | Low: Black-box SQLite internal schema |

### Key Findings & Vector Store Recommendation

1. **Dependency Hygiene & Zero Bloat:**
   - `Chroma` pulls in a web server stack (`uvicorn`, `fastapi`, `starlette`) and telemetry trackers (`posthog`). This violates DocuMind's explicit principle of "no unnecessary frameworks" and "₹0 budget local privacy".
   - `faiss-cpu` installs cleanly with zero auxiliary dependencies other than `numpy`.

2. **Persistence Simplicity:**
   - FAISS natively saves and loads vector index binaries via `faiss.write_index()` and `faiss.read_index()`.
   - Pairing `faiss-cpu` with a dedicated, atomic `metadata.json` sidecar guarantees 100% metadata preservation for all WP-02 chunk attributes (`document_id`, `chunk_id`, `page_numbers`, `char_start`, `char_end`, `text`).

**Recommendation:**  
Adopt **FAISS (`faiss-cpu`)** paired with a typed JSON metadata store. This delivers the fastest local CPU search, minimal dependencies, zero telemetry, and complete metadata transparency.

---

## 3. Storage Format & Index Architecture

The persistent index directory (`data/processed/vector_index/`) will contain:

```text
data/processed/vector_index/
├── manifest.json       # Index configuration, embedding model name, dimension, timestamp
├── index.faiss         # FAISS IndexFlatIP (cosine similarity over L2-normalized vectors)
└── metadata.json       # Array of VectorRecord dictionaries indexed by vector ID (0..N-1)
```

### Deterministic Deduplication Logic
- Every chunk has an invariant `chunk_id` derived deterministically from the document SHA-256 hash, ordinal chunk index, and chunking strategy (e.g. `6a30092e0998_c0000_rec`).
- Before embedding new chunks, the index checks the existing `chunk_id` registry. Already indexed chunks are skipped, preventing duplicate vectors and duplicate memory consumption.

---

## 4. Hardware Benchmarking Methodology

To quantify performance on the user's Intel Core i5-1135G7 CPU:
1. **Cold Start & Load Time:** Measure wall-clock latency to load model weights into RAM.
2. **Embedding Throughput:** Benchmark single chunk vs. batch (batch size 16, 32, 64) inference speed in chunks/second.
3. **Index Build Time:** Benchmark vector insertion and FAISS indexing for the 195 chunks from Phase 1 reference PDFs.
4. **Search Latency:** Measure top-K (k=5) query vector search latency in milliseconds.
5. **Memory Consumption:** Track RAM delta (RSS) before and after model and index loading, confirming usage remains far below the 24 GB hardware ceiling (<1 GB total).
