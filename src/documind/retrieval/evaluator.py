"""Retrieval evaluation framework computing formal IR metrics and stage latencies."""

import statistics
import time
from typing import Any, Dict, List, Optional

from documind.retrieval.context_builder import ContextBuilder
from documind.retrieval.retriever import BaseRetriever
from documind.retrieval.schemas import RetrievalMetrics, RetrievalResult, RetrievedChunk


def _percentile(data: List[float], p: float) -> float:
    """Compute the p-th percentile of a list of floats (0 <= p <= 100)."""
    if not data:
        return 0.0
    if len(data) == 1:
        return float(data[0])
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    if f == c:
        return float(sorted_data[f])
    fraction = k - f
    return float(sorted_data[f] * (1.0 - fraction) + sorted_data[c] * fraction)


class RetrievalEvaluator:
    """Evaluates retrieval quality, relevance calibration, and latency benchmarks."""

    def __init__(self, context_builder: Optional[ContextBuilder] = None) -> None:
        self._context_builder = context_builder or ContextBuilder()

    @staticmethod
    def is_chunk_relevant(chunk: RetrievedChunk, ground_truth_item: Dict[str, Any]) -> bool:
        """Determine if a retrieved chunk matches ground-truth document and content evidence."""
        expected_doc = ground_truth_item.get("expected_document")
        if not expected_doc:
            return False

        if chunk.document_name != expected_doc:
            return False

        # Match against physical page numbers or verified excerpt substring
        expected_pages = ground_truth_item.get("expected_pages", [])
        page_match = False
        if expected_pages:
            for p in expected_pages:
                if p in chunk.page_numbers or (chunk.start_page <= p <= chunk.end_page):
                    page_match = True
                    break

        verified_excerpt = ground_truth_item.get("verified_excerpt")
        excerpt_match = False
        if verified_excerpt:
            if verified_excerpt.lower() in chunk.text.lower():
                excerpt_match = True

        return page_match or excerpt_match

    def evaluate(
        self,
        retriever: BaseRetriever,
        ground_truth_dataset: List[Dict[str, Any]],
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
    ) -> RetrievalMetrics:
        """Run formal evaluation over a ground-truth dataset split."""
        positive_items = [item for item in ground_truth_dataset if not item.get("is_negative", False)]
        negative_items = [item for item in ground_truth_dataset if item.get("is_negative", False)]

        # Positive query metric accumulators
        recalls: List[float] = []
        precisions: List[float] = []
        reciprocal_ranks: List[float] = []

        # Negative query metric accumulators
        negatives_rejected = 0

        # Latency metric accumulators
        embed_times: List[float] = []
        search_times: List[float] = []
        context_times: List[float] = []
        total_times: List[float] = []

        # Attribution accuracy accumulator
        total_chunks_evaluated = 0
        correct_attributions = 0

        # 1. Evaluate Positive Queries
        for item in positive_items:
            res: RetrievalResult = retriever.retrieve(
                query=item["query"],
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )
            embed_times.append(res.embed_latency_ms)
            search_times.append(res.search_latency_ms)

            # Context construction latency
            ctx = self._context_builder.build(res.chunks)
            context_times.append(ctx.context_build_latency_ms)
            total_times.append(res.total_latency_ms + ctx.context_build_latency_ms)

            # Determine relevant matches in retrieved chunks
            relevant_chunks_found = 0
            first_rank: Optional[int] = None

            for rank_idx, chunk in enumerate(res.chunks, start=1):
                is_rel = self.is_chunk_relevant(chunk, item)
                if is_rel:
                    relevant_chunks_found += 1
                    if first_rank is None:
                        first_rank = rank_idx

                # Check attribution metadata completeness
                total_chunks_evaluated += 1
                if chunk.document_name and chunk.page_numbers and chunk.start_page > 0:
                    correct_attributions += 1

            # Recall@K (we assume target has at least 1 relevant chunk)
            # In our benchmark, each positive query targets 1 primary concept in the doc
            target_relevant_count = 1
            query_recall = min(1.0, relevant_chunks_found / target_relevant_count)
            query_precision = relevant_chunks_found / top_k
            query_rr = (1.0 / first_rank) if first_rank is not None else 0.0

            recalls.append(query_recall)
            precisions.append(query_precision)
            reciprocal_ranks.append(query_rr)

        # 2. Evaluate Negative / Distractor Queries
        for item in negative_items:
            res = retriever.retrieve(
                query=item["query"],
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )
            embed_times.append(res.embed_latency_ms)
            search_times.append(res.search_latency_ms)

            ctx = self._context_builder.build(res.chunks)
            context_times.append(ctx.context_build_latency_ms)
            total_times.append(res.total_latency_ms + ctx.context_build_latency_ms)

            # Negative query is successfully rejected if 0 chunks are returned above threshold
            if res.total_retrieved == 0:
                negatives_rejected += 1

        # 3. Aggregate Statistical Metrics
        mean_recall = float(statistics.mean(recalls)) if recalls else 0.0
        mean_precision = float(statistics.mean(precisions)) if precisions else 0.0
        mrr = float(statistics.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

        nrr = (negatives_rejected / len(negative_items)) if negative_items else 1.0
        fpr = 1.0 - nrr

        attrib_acc = (
            (correct_attributions / total_chunks_evaluated)
            if total_chunks_evaluated > 0
            else 1.0
        )

        return RetrievalMetrics(
            total_queries=len(ground_truth_dataset),
            positive_queries=len(positive_items),
            negative_queries=len(negative_items),
            recall_at_k=mean_recall,
            precision_at_k=mean_precision,
            mrr=mrr,
            negative_rejection_rate=nrr,
            false_positive_rate=fpr,
            attribution_accuracy=attrib_acc,
            embed_latency_mean=float(statistics.mean(embed_times)) if embed_times else 0.0,
            embed_latency_median=float(statistics.median(embed_times)) if embed_times else 0.0,
            embed_latency_p95=_percentile(embed_times, 95),
            search_latency_mean=float(statistics.mean(search_times)) if search_times else 0.0,
            search_latency_median=float(statistics.median(search_times)) if search_times else 0.0,
            search_latency_p95=_percentile(search_times, 95),
            context_latency_mean=float(statistics.mean(context_times)) if context_times else 0.0,
            context_latency_median=float(statistics.median(context_times)) if context_times else 0.0,
            context_latency_p95=_percentile(context_times, 95),
            total_latency_mean=float(statistics.mean(total_times)) if total_times else 0.0,
            total_latency_median=float(statistics.median(total_times)) if total_times else 0.0,
            total_latency_p95=_percentile(total_times, 95),
        )
