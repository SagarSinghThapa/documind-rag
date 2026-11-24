"""Retrieval data models, serialization schemas, and telemetry records."""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from documind.storage.schemas import SearchResult


@dataclass(frozen=True)
class RetrievalQuery:
    """Encapsulates a user query, retrieval parameters, and optional filters."""
    text: str
    top_k: int = 5
    similarity_threshold: Optional[float] = None  # None means use config default; supports dynamic override
    document_id_filter: Optional[str] = None


@dataclass(frozen=True)
class RetrievedChunk:
    """Ranked chunk retrieved from vector store coupling score with 100% of Phase 1 metadata."""
    chunk_id: str
    document_id: str
    document_name: str
    source_path: str
    chunk_index: int
    text: str
    page_numbers: List[int]
    start_page: int
    end_page: int
    char_start: int
    char_end: int
    score: float                                # Effective/prioritized score for ranking and thresholding
    raw_score: Optional[float] = None           # Actual raw cosine similarity from dense vector search
    retrieval_source: str = "dense"             # 'dense' or 'metadata_prioritized'
    chunking_strategy: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_search_result(cls, result: SearchResult) -> "RetrievedChunk":
        """Instantiate RetrievedChunk from an underlying SearchResult."""
        rec = result.record
        score_val = float(result.score)
        return cls(
            chunk_id=rec.chunk_id,
            document_id=rec.document_id,
            document_name=rec.document_name,
            source_path=rec.source_path,
            chunk_index=rec.chunk_index,
            text=rec.text,
            page_numbers=list(rec.page_numbers),
            start_page=rec.start_page,
            end_page=rec.end_page,
            char_start=rec.char_start,
            char_end=rec.char_end,
            score=score_val,
            raw_score=score_val,
            retrieval_source="dense",
            chunking_strategy=rec.chunking_strategy,
            chunk_size=rec.chunk_size,
            chunk_overlap=rec.chunk_overlap,
            metadata=dict(rec.metadata),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to a serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class RetrievalResult:
    """Structured telemetry and candidate chunks returned by SemanticRetriever."""
    query: str
    chunks: List[RetrievedChunk]
    total_retrieved: int
    total_below_threshold: int
    embed_latency_ms: float                     # Pure query vectorization time
    search_latency_ms: float                    # Pure FAISS index lookup time
    total_latency_ms: float                     # End-to-end retrieval latency


@dataclass(frozen=True)
class ContextReference:
    """Verifiable citation reference linking context to exact physical source pages."""
    document_name: str
    document_id: str
    source_path: str
    start_page: int
    end_page: int
    page_numbers: List[int]
    chunk_id: str
    chunk_index: int
    score: float
    char_start: int
    char_end: int
    source_id: str = ""                         # Deterministic source identifier (e.g. "S1", "S2")


@dataclass(frozen=True)
class FormattedContext:
    """Structured context block ready for downstream LLM prompt injection."""
    text: str
    references: List[ContextReference]
    total_chars: int
    estimated_tokens: int
    chunks_included: int
    chunks_discarded_budget: int
    context_build_latency_ms: float


@dataclass(frozen=True)
class RetrievalMetrics:
    """Formal information retrieval and latency benchmarks across query splits."""
    total_queries: int
    positive_queries: int
    negative_queries: int
    recall_at_k: float                          # Computed solely on positive queries where |Relevant| > 0
    precision_at_k: float                       # Computed on positive queries
    mrr: float                                  # Mean Reciprocal Rank on positive queries (0.0 if not in top K)
    negative_rejection_rate: float              # Fraction of negative queries returning 0 chunks above threshold
    false_positive_rate: float                  # 1.0 - negative_rejection_rate
    attribution_accuracy: float                 # Fraction of retrieved chunks matching verified document/page
    # Disaggregated Latency Profiles (ms)
    embed_latency_mean: float
    embed_latency_median: float
    embed_latency_p95: float
    search_latency_mean: float
    search_latency_median: float
    search_latency_p95: float
    context_latency_mean: float
    context_latency_median: float
    context_latency_p95: float
    total_latency_mean: float
    total_latency_median: float
    total_latency_p95: float
