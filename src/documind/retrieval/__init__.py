"""Semantic retrieval subsystem for DocuMind."""

from documind.retrieval.schemas import (
    ContextReference,
    FormattedContext,
    RetrievalMetrics,
    RetrievalQuery,
    RetrievalResult,
    RetrievedChunk,
)
from documind.retrieval.retriever import BaseRetriever, SemanticRetriever
from documind.retrieval.context_builder import ContextBuilder
from documind.retrieval.evaluator import RetrievalEvaluator

__all__ = [
    "BaseRetriever",
    "SemanticRetriever",
    "ContextBuilder",
    "RetrievalEvaluator",
    "RetrievalQuery",
    "RetrievedChunk",
    "RetrievalResult",
    "ContextReference",
    "FormattedContext",
    "RetrievalMetrics",
]
