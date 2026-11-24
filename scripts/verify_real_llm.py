"""Real Local LLM Verification Script for DocuMind Phase 4 (WP-05).

Executes genuine offline CPU inference with Qwen/Qwen2.5-0.5B-Instruct across:
1. Factual questions with direct supporting evidence.
2. Paraphrased questions.
3. Multi-chunk questions.
4. Questions requiring correct page citations.
5. Unsupported questions.
6. Empty retrieval results.
7. Contradictory / insufficient evidence.
8. Full Split B holdout benchmark (N=10).
"""

import json
import os
import sys
import time
from typing import Any, Dict, List

import psutil

# Ensure src is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from documind.config import EmbeddingConfig, GenerationConfig, RetrievalConfig
from documind.generation.evaluator import GroundingEvaluator, GroundingMetrics
from documind.generation.generator import GroundedGenerator
from documind.generation.llm import TransformersLLM
from documind.ingestion.pipeline import IngestionPipeline
from documind.retrieval.context_builder import ContextBuilder
from documind.retrieval.retriever import SemanticRetriever
from documind.storage import DocumentIndex, FaissVectorStore, LocalEmbedder


def get_current_ram_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)


def measure_ttft(llm: TransformersLLM, prompt_messages: Any) -> float:
    """Measure Time To First Token (TTFT) in milliseconds using stream_generate."""
    t0 = time.perf_counter()
    for _ in llm.stream_generate(prompt_messages, max_new_tokens=10):
        ttft_ms = (time.perf_counter() - t0) * 1000.0
        return ttft_ms
    return 0.0


def main():
    print("=" * 80)
    print("DOCUMIND PHASE 4 — REAL LOCAL LLM VERIFICATION RUNNER")
    print("=" * 80)

    start_ram = get_current_ram_mb()
    print(f"Process initial RSS memory: {start_ram:.2f} MB")

    # 1. Ingest Technical Reference PDFs
    print("\n[1/4] Ingesting Technical Reference PDFs into FAISS Index...")
    embedder = LocalEmbedder()
    store = FaissVectorStore(dimension=embedder.dimension)
    document_index = DocumentIndex(embedder=embedder, vector_store=store)
    pipeline = IngestionPipeline(chunk_size=1000, chunk_overlap=150)

    eval_pdfs = [
        "research/technical_references/BASWE-15-RAG-Concepts-Guide.pdf",
        "research/technical_references/Everything-to-know-about-RAG.pdf",
    ]
    for p in eval_pdfs:
        chunks, _ = pipeline.ingest_file(p)
        document_index.add_chunks(chunks)
        print(f"  - Ingested {os.path.basename(p)}: {len(chunks)} chunks")

    print(f"Total indexed vectors: {document_index.total_vectors}")

    retrieval_cfg = RetrievalConfig(top_k=5, similarity_threshold=0.35)
    retriever = SemanticRetriever(document_index=document_index, config=retrieval_cfg)

    # 2. Cold Model Load Measurement
    print("\n[2/4] Loading Qwen/Qwen2.5-0.5B-Instruct on CPU (local_files_only=True)...")
    gen_cfg = GenerationConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        cache_folder="models/llm",
        device="cpu",
        local_files_only=True,
        temperature=0.0,
        max_new_tokens=256,
        max_context_chars=1800,
        num_threads=8,
        include_exemplar=True,
    )
    context_builder = ContextBuilder(max_chars=gen_cfg.max_context_chars)

    t_cold_start = time.perf_counter()
    llm = TransformersLLM(config=gen_cfg)
    llm._ensure_loaded()
    cold_load_ms = (time.perf_counter() - t_cold_start) * 1000.0
    post_load_ram = get_current_ram_mb()
    load_ram_delta = post_load_ram - start_ram

    print(f"Cold model load latency: {cold_load_ms:.2f} ms")
    print(f"Post-load RSS memory: {post_load_ram:.2f} MB (Delta: +{load_ram_delta:.2f} MB)")

    generator = GroundedGenerator(
        retriever=retriever,
        llm=llm,
        context_builder=context_builder,
        config=gen_cfg,
    )

    # 3. Categorized Real-Model Test Suite
    print("\n[3/4] Executing Categorized Real-Model Test Inquiries...")
    test_cases = [
        {
            "category": "1. Factual with direct supporting evidence",
            "query": "What is the typical recommended range for chunk overlap percentage?",
            "expected_behavior": "Answers 10 to 20 percent with citation to BASWE guide p. 3",
        },
        {
            "category": "2. Paraphrased question",
            "query": "How much repetition should occur between adjacent text segments in chunking?",
            "expected_behavior": "Answers 10-20% overlap based on same source evidence",
        },
        {
            "category": "3. Multi-chunk question",
            "query": "What is hybrid search and how does it combine embeddings with keyword search?",
            "expected_behavior": "Explains dense + sparse retrieval across pages 6-7",
        },
        {
            "category": "4. Questions requiring correct page citations",
            "query": "How does an LLM count text and what is the rule of thumb for tokens to English words?",
            "expected_behavior": "Answers 100 tokens is about 75 English words with citation to Everything-to-know p. 5",
        },
        {
            "category": "5. Unsupported / Out-of-domain question",
            "query": "What is the proper water temperature and fermentation time for sourdough bread?",
            "expected_behavior": "Fast-path pre-generation abstention",
        },
        {
            "category": "6. Empty retrieval results / Below threshold",
            "query": "How do I replace a high-voltage lithium electric vehicle battery pack?",
            "expected_behavior": "Fast-path pre-generation abstention (0 chunks above 0.35)",
        },
        {
            "category": "7. Contradictory or insufficient evidence",
            "query": "What is the exact subscription price of DocuMind enterprise edition and its refund policy?",
            "expected_behavior": "Abstains due to lack of evidence in context",
        },
    ]

    detailed_results = []
    for tc in test_cases:
        print(f"\n--- Running: {tc['category']} ---")
        print(f"Query: {tc['query']}")
        t0 = time.perf_counter()
        ans = generator.generate_answer(tc["query"], top_k=5, similarity_threshold=0.35)
        total_time_ms = (time.perf_counter() - t0) * 1000.0

        rec = {
            "category": tc["category"],
            "query": tc["query"],
            "expected_behavior": tc["expected_behavior"],
            "raw_answer": ans.answer,
            "is_abstained": ans.is_abstained,
            "abstention_type": ans.abstention_type,
            "grounding_status": ans.grounding_status,
            "cited_source_ids": ans.cited_source_ids,
            "citations": [c.to_dict() for c in ans.citations],
            "retrieval_chunks_count": len(ans.context.references),
            "retrieval_references": [
                {"doc": r.document_name, "pages": f"{r.start_page}-{r.end_page}", "chunk_id": r.chunk_id}
                for r in ans.context.references
            ],
            "telemetry": ans.telemetry,
            "total_latency_ms": total_time_ms,
        }
        detailed_results.append(rec)

        print(f"Answer: {ans.answer}")
        print(f"Abstained: {ans.is_abstained} (Type: {ans.abstention_type})")
        print(f"Grounding Status: {ans.grounding_status}")
        print(f"Citations: {[c.raw_text for c in ans.citations]}")
        print(f"Latency: {total_time_ms:.1f} ms | Gen TPS: {ans.telemetry.get('tokens_per_sec', 0.0):.2f}")

    # 4. Measure TTFT on a standard factual prompt
    sample_ctx = context_builder.build(retriever.retrieve(test_cases[0]["query"]).chunks)
    from documind.generation.prompts import build_grounded_prompt
    sample_prompt = build_grounded_prompt(test_cases[0]["query"], sample_ctx)
    ttft_ms = measure_ttft(llm, sample_prompt)
    print(f"\nMeasured Time To First Token (TTFT): {ttft_ms:.2f} ms")

    # 5. Full Evaluation Benchmark on Split B Holdout (N=10)
    print("\n[4/4] Running Full Evaluation Benchmark on Split B Holdout (N=10)...")
    evaluator = GroundingEvaluator(generator=generator, eval_data_path="data/evaluation/retrieval_ground_truth.json")
    split_b_metrics: GroundingMetrics = evaluator.evaluate(split="evaluation", top_k=5, similarity_threshold=0.35)

    peak_ram = get_current_ram_mb()
    total_ram_growth = peak_ram - start_ram

    output_payload = {
        "hardware_metrics": {
            "cold_model_load_ms": round(cold_load_ms, 2),
            "ttft_ms": round(ttft_ms, 2),
            "mean_generation_latency_ms": round(split_b_metrics.mean_generation_latency_ms, 2),
            "mean_total_latency_ms": round(split_b_metrics.mean_total_latency_ms, 2),
            "mean_tokens_per_sec": round(split_b_metrics.mean_tokens_per_sec, 2),
            "start_ram_mb": round(start_ram, 2),
            "peak_ram_mb": round(peak_ram, 2),
            "peak_ram_growth_mb": round(total_ram_growth, 2),
        },
        "categorized_tests": detailed_results,
        "split_b_metrics": split_b_metrics.to_dict(),
        "query_evaluations": split_b_metrics.query_evaluations,
    }

    out_path = "docs/project_reports/real_llm_verification_data.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\nSaved raw verification data to: {out_path}")
    print("Verification execution complete.")


if __name__ == "__main__":
    main()
