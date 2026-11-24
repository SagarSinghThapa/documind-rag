"""Automated unit and integration tests for RetrievalEvaluator and ground-truth metrics."""

import json
from pathlib import Path
from typing import Optional, Union
import pytest

from documind.config import IngestionConfig
from documind.ingestion.pipeline import IngestionPipeline
from documind.retrieval import (
    BaseRetriever,
    ContextBuilder,
    RetrievalEvaluator,
    RetrievalMetrics,
    RetrievalQuery,
    RetrievalResult,
    RetrievedChunk,
    SemanticRetriever,
)
from documind.storage.embedder import LocalEmbedder
from documind.storage.index import DocumentIndex
from documind.storage.vector_store import FaissVectorStore


class MockRetriever(BaseRetriever):
    """Mock retriever emitting pre-configured chunks for metric math verification."""

    def __init__(self, responses):
        self.responses = responses

    def retrieve(
        self,
        query: Union[str, RetrievalQuery],
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> RetrievalResult:
        query_str = query if isinstance(query, str) else query.text
        chunks = self.responses.get(query_str, [])
        if similarity_threshold is not None:
            chunks = [c for c in chunks if c.score >= similarity_threshold]

        return RetrievalResult(
            query=query_str,
            chunks=chunks,
            total_retrieved=len(chunks),
            total_below_threshold=0,
            embed_latency_ms=10.0,
            search_latency_ms=5.0,
            total_latency_ms=15.0,
        )


def make_chunk(doc_name: str, pages: list[int], text: str = "text", score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c_test",
        document_id="d_test",
        document_name=doc_name,
        source_path=f"data/{doc_name}",
        chunk_index=0,
        text=text,
        page_numbers=pages,
        start_page=min(pages),
        end_page=max(pages),
        char_start=0,
        char_end=len(text),
        score=score,
    )


class TestRetrievalEvaluatorUnit:
    """Unit tests for metric math, positive/negative separation, and percentile calculations."""

    def test_evaluator_metrics_math_synthetic(self):
        """Validates exact math for Recall@K, Precision@K, MRR, NRR, and FPR."""
        # Query 1: relevant chunk at rank 1 (RR = 1.0)
        # Query 2: relevant chunk at rank 2 (RR = 0.5)
        # Query 3: relevant chunk not in top K (RR = 0.0)
        # Neg 1: 0 chunks returned (rejected = True)
        # Neg 2: 1 chunk returned (rejected = False)
        responses = {
            "Q1": [
                make_chunk("doc1.pdf", [1], "combines retrieval with generation"),
                make_chunk("other.pdf", [1], "noise"),
            ],
            "Q2": [
                make_chunk("other.pdf", [1], "noise"),
                make_chunk("doc1.pdf", [2], "chunk overlap ranges 10 to 20 percent"),
            ],
            "Q3": [
                make_chunk("other.pdf", [1], "noise"),
                make_chunk("other.pdf", [2], "noise"),
            ],
            "NEG1": [],
            "NEG2": [make_chunk("other.pdf", [1], "hallucinated noise", score=0.2)],
        }

        dataset = [
            {
                "query": "Q1",
                "is_negative": False,
                "expected_document": "doc1.pdf",
                "expected_pages": [1],
                "verified_excerpt": "combines retrieval",
            },
            {
                "query": "Q2",
                "is_negative": False,
                "expected_document": "doc1.pdf",
                "expected_pages": [2],
                "verified_excerpt": "10 to 20 percent",
            },
            {
                "query": "Q3",
                "is_negative": False,
                "expected_document": "doc1.pdf",
                "expected_pages": [3],
                "verified_excerpt": "nonexistent text",
            },
            {"query": "NEG1", "is_negative": True, "expected_document": None},
            {"query": "NEG2", "is_negative": True, "expected_document": None},
        ]

        mock_retriever = MockRetriever(responses)
        evaluator = RetrievalEvaluator()
        metrics: RetrievalMetrics = evaluator.evaluate(mock_retriever, dataset, top_k=2)

        assert metrics.total_queries == 5
        assert metrics.positive_queries == 3
        assert metrics.negative_queries == 2

        # Recall@2: Q1 (1.0), Q2 (1.0), Q3 (0.0) -> Mean = 2.0 / 3 = 0.6667
        assert metrics.recall_at_k == pytest.approx(2.0 / 3.0, abs=1e-4)

        # Precision@2: Q1 (1/2 = 0.5), Q2 (1/2 = 0.5), Q3 (0/2 = 0.0) -> Mean = 1.0 / 3 = 0.3333
        assert metrics.precision_at_k == pytest.approx(1.0 / 3.0, abs=1e-4)

        # MRR: Q1 (1/1 = 1.0), Q2 (1/2 = 0.5), Q3 (0.0) -> Mean = 1.5 / 3 = 0.5
        assert metrics.mrr == pytest.approx(0.5, abs=1e-4)

        # NRR on 2 negative queries: NEG1 (0 chunks => 1), NEG2 (1 chunk => 0) -> NRR = 1/2 = 0.5
        assert metrics.negative_rejection_rate == pytest.approx(0.5, abs=1e-4)
        assert metrics.false_positive_rate == pytest.approx(0.5, abs=1e-4)

    def test_positive_negative_separation(self):
        """Negative distractor queries never count toward or distort positive recall/precision calculations."""
        responses = {
            "POS": [make_chunk("doc.pdf", [1], "true text")],
            "NEG": [],
        }
        dataset = [
            {"query": "POS", "is_negative": False, "expected_document": "doc.pdf", "expected_pages": [1]},
            {"query": "NEG", "is_negative": True, "expected_document": None},
        ]
        mock = MockRetriever(responses)
        evaluator = RetrievalEvaluator()
        metrics = evaluator.evaluate(mock, dataset, top_k=1)

        assert metrics.positive_queries == 1
        assert metrics.negative_queries == 1
        assert metrics.recall_at_k == 1.0
        assert metrics.precision_at_k == 1.0
        assert metrics.mrr == 1.0
        assert metrics.negative_rejection_rate == 1.0

    def test_latency_percentiles_calculation(self):
        """Verifies mean, median, and P95 latency statistics."""
        responses = {"Q": [make_chunk("d.pdf", [1])]}
        dataset = [{"query": "Q", "is_negative": False, "expected_document": "d.pdf", "expected_pages": [1]}]
        mock = MockRetriever(responses)
        evaluator = RetrievalEvaluator()
        metrics = evaluator.evaluate(mock, dataset, top_k=1)

        assert metrics.embed_latency_mean > 0.0
        assert metrics.embed_latency_median > 0.0
        assert metrics.embed_latency_p95 >= metrics.embed_latency_median
        assert metrics.total_latency_p95 > 0.0


class TestPublicPDFRetrievalEvaluationIntegration:
    """Integration test evaluating retrieval on real ingested PDF and ground-truth dataset."""

    PDF_PATH = Path("research/technical_references/BASWE-15-RAG-Concepts-Guide.pdf")
    GT_PATH = Path("data/evaluation/retrieval_ground_truth.json")

    @pytest.mark.skipif(
        not (Path("research/technical_references/BASWE-15-RAG-Concepts-Guide.pdf")).exists(),
        reason="Research PDF not found in local workspace",
    )
    def test_evaluator_against_ground_truth_calibration_split(self):
        """Ingests BASWE guide and evaluates Split A (calibration), verifying accuracy without brittle hardware gates."""
        with open(self.GT_PATH, "r", encoding="utf-8") as f:
            all_gt = json.load(f)

        calib_items = [item for item in all_gt if item.get("split") == "calibration"]
        assert len(calib_items) == 10

        # Ingest PDF into clean index
        embedder = LocalEmbedder()
        store = FaissVectorStore(dimension=embedder.dimension)
        index = DocumentIndex(embedder=embedder, vector_store=store)
        pipeline = IngestionPipeline(chunk_size=1000, chunk_overlap=150)

        chunks, report = pipeline.ingest_file(self.PDF_PATH)
        assert report.success is True
        index.add_chunks(chunks)

        retriever = SemanticRetriever(index)
        evaluator = RetrievalEvaluator()

        # 1. Evaluate baseline (no threshold)
        baseline_metrics: RetrievalMetrics = evaluator.evaluate(
            retriever, calib_items, top_k=5, similarity_threshold=None
        )
        assert baseline_metrics.total_queries == 10
        assert baseline_metrics.positive_queries == 6
        assert baseline_metrics.negative_queries == 4
        assert baseline_metrics.recall_at_k >= 0.75
        assert baseline_metrics.mrr >= 0.70
        assert baseline_metrics.attribution_accuracy == 1.0

        # 2. Evaluate with initial candidate threshold 0.35
        filtered_metrics: RetrievalMetrics = evaluator.evaluate(
            retriever, calib_items, top_k=5, similarity_threshold=0.35
        )
        # Verify negative queries are rejected when threshold is active
        assert filtered_metrics.negative_rejection_rate >= 0.75
        assert filtered_metrics.attribution_accuracy == 1.0
