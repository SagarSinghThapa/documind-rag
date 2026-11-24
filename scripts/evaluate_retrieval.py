"""Standalone CLI to calibrate retrieval thresholds on Split A and evaluate holdout Split B."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from documind.config import IngestionConfig, RetrievalConfig
from documind.ingestion.pipeline import IngestionPipeline
from documind.retrieval import ContextBuilder, RetrievalEvaluator, SemanticRetriever
from documind.storage.embedder import LocalEmbedder
from documind.storage.index import DocumentIndex
from documind.storage.vector_store import FaissVectorStore


def main():
    parser = argparse.ArgumentParser(description="Evaluate DocuMind Semantic Retrieval Subsystem")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("data/evaluation/retrieval_ground_truth.json"),
        help="Path to ground-truth dataset JSON",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K nearest neighbors to retrieve",
    )
    args = parser.parse_args()

    if not args.ground_truth.exists():
        print(f"Error: Ground truth file not found at {args.ground_truth}")
        sys.exit(1)

    with open(args.ground_truth, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    calib_set = [item for item in ground_truth if item.get("split") == "calibration"]
    eval_set = [item for item in ground_truth if item.get("split") == "evaluation"]

    print("=" * 80)
    print("DOCUMIND SEMANTIC RETRIEVAL BENCHMARK & THRESHOLD CALIBRATION")
    print("=" * 80)
    print(f"Total Ground-Truth Queries: {len(ground_truth)}")
    print(f"  - Calibration Split (Split A): {len(calib_set)} queries (6 positive, 4 negative)")
    print(f"  - Holdout Evaluation Split (Split B): {len(eval_set)} queries (6 positive, 4 negative)")

    # 1. Ingest evaluation PDFs
    pdf1 = Path("research/technical_references/BASWE-15-RAG-Concepts-Guide.pdf")
    pdf2 = Path("research/technical_references/Everything-to-know-about-RAG.pdf")

    print("\n[1/3] Ingesting evaluation PDFs into in-memory vector index...")
    embedder = LocalEmbedder()
    store = FaissVectorStore(dimension=embedder.dimension)
    index = DocumentIndex(embedder=embedder, vector_store=store)
    pipeline = IngestionPipeline(chunk_size=1000, chunk_overlap=150)

    for pdf_path in [pdf1, pdf2]:
        if pdf_path.exists():
            chunks, rep = pipeline.ingest_file(pdf_path)
            res = index.add_chunks(chunks)
            print(f"  - Ingested {pdf_path.name}: {len(chunks)} chunks ({res.newly_indexed} indexed)")
        else:
            print(f"  - Warning: Missing PDF {pdf_path}")

    print(f"Total indexed vectors: {index.total_vectors}")

    retriever = SemanticRetriever(index)
    builder = ContextBuilder(max_chars=4000, deduplicate_overlap=True)
    evaluator = RetrievalEvaluator(context_builder=builder)

    # 2. Calibration sweep on Split A
    print("\n[2/3] Sweeping candidate thresholds on Split A (Calibration)...")
    print(f"{'Threshold (tau)':<16} | {'Recall@5':<10} | {'Precision@5':<12} | {'F1':<8} | {'MRR':<8} | {'NRR':<8} | {'FPR':<8}")
    print("-" * 80)

    best_tau = None
    best_f1 = -1.0
    best_metrics = None

    threshold_candidates = [None, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    for tau in threshold_candidates:
        metrics = evaluator.evaluate(retriever, calib_set, top_k=args.top_k, similarity_threshold=tau)
        p = metrics.precision_at_k
        r = metrics.recall_at_k
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        tau_str = f"{tau:.2f}" if tau is not None else "None (baseline)"
        print(
            f"{tau_str:<16} | {metrics.recall_at_k:<10.3f} | {metrics.precision_at_k:<12.3f} | {f1:<8.3f} | "
            f"{metrics.mrr:<8.3f} | {metrics.negative_rejection_rate:<8.3f} | {metrics.false_positive_rate:<8.3f}"
        )

        # Optimize F1 while requiring NRR >= 0.75 on negatives
        if tau is not None and metrics.negative_rejection_rate >= 0.75:
            if f1 > best_f1:
                best_f1 = f1
                best_tau = tau
                best_metrics = metrics

    if best_tau is None:
        best_tau = 0.35

    print(f"\n=> Calibrated Optimal Threshold: tau* = {best_tau:.2f} (Calibration F1: {best_f1:.3f})")

    # 3. Unbiased Evaluation on Split B (Holdout)
    print("\n[3/3] Evaluating holdout Split B under calibrated threshold (tau*)...")
    eval_metrics_baseline = evaluator.evaluate(retriever, eval_set, top_k=args.top_k, similarity_threshold=None)
    eval_metrics_calibrated = evaluator.evaluate(retriever, eval_set, top_k=args.top_k, similarity_threshold=best_tau)

    print("\n" + "=" * 80)
    print(f"HOLDOUT EVALUATION RESULTS (Split B - Unbiased Generalization)")
    print("=" * 80)
    print(f"{'Metric':<30} | {'No-Threshold Baseline':<22} | {'Calibrated (tau=' + str(best_tau) + ')':<22}")
    print("-" * 80)
    print(f"{'Recall@5':<30} | {eval_metrics_baseline.recall_at_k:<22.3f} | {eval_metrics_calibrated.recall_at_k:<22.3f}")
    print(f"{'Precision@5':<30} | {eval_metrics_baseline.precision_at_k:<22.3f} | {eval_metrics_calibrated.precision_at_k:<22.3f}")
    print(f"{'Mean Reciprocal Rank (MRR)':<30} | {eval_metrics_baseline.mrr:<22.3f} | {eval_metrics_calibrated.mrr:<22.3f}")
    print(f"{'Negative Rejection Rate (NRR)':<30} | {eval_metrics_baseline.negative_rejection_rate:<22.3f} | {eval_metrics_calibrated.negative_rejection_rate:<22.3f}")
    print(f"{'False Positive Rate (FPR)':<30} | {eval_metrics_baseline.false_positive_rate:<22.3f} | {eval_metrics_calibrated.false_positive_rate:<22.3f}")
    print(f"{'Attribution Accuracy':<30} | {eval_metrics_baseline.attribution_accuracy:<22.3f} | {eval_metrics_calibrated.attribution_accuracy:<22.3f}")

    print("\n" + "=" * 80)
    print("DISAGGREGATED LATENCY PROFILES (ms)")
    print("=" * 80)
    print(f"{'Pipeline Stage':<25} | {'Mean (ms)':<15} | {'Median (ms)':<15} | {'P95 (ms)':<15}")
    print("-" * 80)
    m = eval_metrics_calibrated
    print(f"{'Query Embedding':<25} | {m.embed_latency_mean:<15.2f} | {m.embed_latency_median:<15.2f} | {m.embed_latency_p95:<15.2f}")
    print(f"{'FAISS Search':<25} | {m.search_latency_mean:<15.2f} | {m.search_latency_median:<15.2f} | {m.search_latency_p95:<15.2f}")
    print(f"{'Context Construction':<25} | {m.context_latency_mean:<15.2f} | {m.context_latency_median:<15.2f} | {m.context_latency_p95:<15.2f}")
    print(f"{'Total Retrieval Pipeline':<25} | {m.total_latency_mean:<15.2f} | {m.total_latency_median:<15.2f} | {m.total_latency_p95:<15.2f}")


if __name__ == "__main__":
    main()
