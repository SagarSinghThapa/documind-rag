"""CPU performance and memory benchmarking suite for DocuMind embeddings and FAISS vector index.

Decoupled from standard unit tests to ensure hardware-dependent timings
do not cause spurious CI test failures while providing empirical evidence
of local performance on the user's Intel Core i5 CPU.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List
import numpy as np

# Ensure src is on discovery path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from documind.config import EmbeddingConfig, VectorStoreConfig
from documind.ingestion.schemas import TextChunk
from documind.storage.embedder import LocalEmbedder
from documind.storage.index import DocumentIndex
from documind.storage.vector_store import FaissVectorStore

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def get_process_memory_mb() -> float:
    """Return resident memory (RSS) in megabytes."""
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    return 0.0


def generate_benchmark_chunks(num_chunks: int = 150) -> List[TextChunk]:
    """Generate realistic synthetic chunks for benchmarking."""
    base_sentences = [
        "Retrieval-Augmented Generation combines dense semantic search with generative language models.",
        "Chunking strategies partition documents using recursive character separators to preserve paragraph cohesion.",
        "FAISS IndexFlatIP executes vectorized SIMD inner products over L2-normalized dense embeddings.",
        "Local CPU inference avoids external network latency, cloud API billing, and private document exposure.",
        "Metadata sidecars guarantee full attribution to source filenames, page ranges, and character offsets.",
        "Deterministic chunk identifiers prevent duplicate vector entries across repeated indexing passes.",
        "Immutable versioned snapshots provide crash-consistent vector persistence without in-place corruption.",
    ]
    chunks = []
    for i in range(num_chunks):
        sentence_idx = i % len(base_sentences)
        text = f"Chunk {i:04d}: {base_sentences[sentence_idx]} Additional context span to pad length to ~350 characters for realistic embedding latency benchmarks."
        chunk = TextChunk(
            chunk_id=f"bench_doc_c{i:04d}_rec",
            document_id="bench_doc_hash_001",
            document_name="benchmark_document.pdf",
            source_path="benchmark/benchmark_document.pdf",
            chunk_index=i,
            text=text,
            page_numbers=[(i // 3) + 1],
            start_page=(i // 3) + 1,
            end_page=(i // 3) + 1,
            chunking_strategy="recursive",
            chunk_size=1000,
            chunk_overlap=150,
            char_start=i * 350,
            char_end=(i + 1) * 350,
            metadata={"benchmark_iteration": i},
        )
        chunks.append(chunk)
    return chunks


def run_benchmark():
    print("=" * 70)
    print("DocuMind Phase 2 — CPU Performance and Memory Benchmark")
    print("=" * 70)
    
    mem_initial = get_process_memory_mb()
    print(f"Initial Process Memory: {mem_initial:.2f} MB")

    # 1. Model Loading Benchmark
    print("\n[Benchmark 1/4] Embedding Model Initialization (CPU)...")
    config = EmbeddingConfig(local_files_only=True, allow_download=False)
    t0 = time.perf_counter()
    embedder = LocalEmbedder(config)
    t_load = time.perf_counter() - t0
    mem_after_model = get_process_memory_mb()
    model_ram_delta = mem_after_model - mem_initial
    print(f"  Model Load Latency:   {t_load:.3f} s")
    print(f"  RAM RSS after model:  {mem_after_model:.2f} MB (Delta: +{model_ram_delta:.2f} MB)")

    # 2. Batch Embedding Benchmark
    print("\n[Benchmark 2/4] CPU Batch Embedding Throughput...")
    chunks = generate_benchmark_chunks(150)
    texts = [c.text for c in chunks]

    batch_sizes = [16, 32, 64]
    throughput_results = {}
    for bs in batch_sizes:
        t_start = time.perf_counter()
        _ = embedder.embed_texts(texts, batch_size=bs)
        t_batch = time.perf_counter() - t_start
        throughput = len(texts) / t_batch
        latency_per_chunk_ms = (t_batch / len(texts)) * 1000.0
        throughput_results[bs] = {
            "total_time": t_batch,
            "throughput": throughput,
            "latency_ms": latency_per_chunk_ms,
        }
        print(f"  Batch size {bs:2d}: {t_batch:.3f} s ({throughput:.1f} chunks/s, {latency_per_chunk_ms:.2f} ms/chunk)")

    # 3. Vector Indexing Benchmark
    print("\n[Benchmark 3/4] Vector Store Indexing...")
    vector_store = FaissVectorStore(
        dimension=embedder.dimension,
        metric="cosine",
        normalize_embeddings=True,
        snapshot_retention=2,
    )
    doc_index = DocumentIndex(embedder=embedder, vector_store=vector_store)

    t_idx_start = time.perf_counter()
    res = doc_index.add_chunks(chunks)
    t_indexing = time.perf_counter() - t_idx_start
    mem_after_index = get_process_memory_mb()
    index_ram_delta = mem_after_index - mem_after_model
    print(f"  Indexed {res.newly_indexed} chunks in {t_indexing:.3f} s")
    print(f"  Indexing throughput:  {res.newly_indexed / t_indexing:.1f} chunks/s")
    print(f"  RAM RSS after index:  {mem_after_index:.2f} MB (Delta: +{index_ram_delta:.2f} MB)")

    # 4. Search Latency Benchmark
    print("\n[Benchmark 4/4] Top-K (k=5) Search Latency (50 iterations)...")
    sample_queries = [
        "What is Retrieval-Augmented Generation?",
        "How does recursive chunking work?",
        "Explain FAISS vector indexing on CPU",
        "Deterministic identifiers for duplicate chunk prevention",
        "Local memory and latency benchmarks",
    ]
    latencies_ms = []
    # Warmup
    for q in sample_queries:
        _ = doc_index.search(q, top_k=5)

    # Timed runs
    for i in range(50):
        q = sample_queries[i % len(sample_queries)]
        t_q0 = time.perf_counter()
        results = doc_index.search(q, top_k=5)
        t_q = (time.perf_counter() - t_q0) * 1000.0
        latencies_ms.append(t_q)

    mean_lat = np.mean(latencies_ms)
    median_lat = np.median(latencies_ms)
    p95_lat = np.percentile(latencies_ms, 95)
    min_lat = np.min(latencies_ms)
    max_lat = np.max(latencies_ms)

    print(f"  Mean Search Latency:   {mean_lat:.2f} ms")
    print(f"  Median Search Latency: {median_lat:.2f} ms")
    print(f"  P95 Search Latency:    {p95_lat:.2f} ms")
    print(f"  Min / Max Latency:     {min_lat:.2f} ms / {max_lat:.2f} ms")

    print("\n" + "=" * 70)
    print("Benchmark Summary Table:")
    print("=" * 70)
    print(f"| Metric | Measured Value | Target / Hardware Limit |")
    print(f"| :--- | :--- | :--- |")
    print(f"| Model RAM Footprint | {model_ram_delta:.1f} MB | < 250 MB |")
    print(f"| Peak Process RAM | {mem_after_index:.1f} MB | < 1,000 MB (Hardware: 24 GB) |")
    best_bs = 32 if 32 in throughput_results else 16
    print(f"| Inference Throughput (bs={best_bs}) | {throughput_results[best_bs]['throughput']:.1f} chunks/s | > 30 chunks/s |")
    print(f"| CPU Latency per Chunk | {throughput_results[best_bs]['latency_ms']:.2f} ms | < 25 ms |")
    print(f"| Top-K Search Latency (Mean) | {mean_lat:.2f} ms | < 10 ms |")
    print(f"| Top-K Search Latency (P95) | {p95_lat:.2f} ms | < 20 ms |")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
