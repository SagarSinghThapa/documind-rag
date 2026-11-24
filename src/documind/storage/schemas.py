"""Storage data models, serialization schemas, and telemetry records."""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import datetime

from documind.ingestion.schemas import TextChunk


@dataclass
class VectorRecord:
    """Storage record coupling chunk text, all Phase 1 metadata fields, and embedding.
    
    Preserves 100% of WP-02 chunk attributes for full source attribution and citation.
    """
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
    chunking_strategy: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    @classmethod
    def from_text_chunk(
        cls, chunk: TextChunk, embedding: Optional[List[float]] = None
    ) -> "VectorRecord":
        """Factory creating a VectorRecord from an existing WP-02 TextChunk."""
        return cls(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            document_name=chunk.document_name,
            source_path=chunk.source_path,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            page_numbers=list(chunk.page_numbers),
            start_page=chunk.start_page,
            end_page=chunk.end_page,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            chunking_strategy=chunk.chunking_strategy,
            chunk_size=chunk.chunk_size,
            chunk_overlap=chunk.chunk_overlap,
            metadata=dict(chunk.metadata),
            embedding=embedding,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorRecord":
        """Instantiate a VectorRecord from a dictionary."""
        return cls(
            chunk_id=data["chunk_id"],
            document_id=data["document_id"],
            document_name=data["document_name"],
            source_path=data["source_path"],
            chunk_index=data["chunk_index"],
            text=data["text"],
            page_numbers=list(data["page_numbers"]),
            start_page=data["start_page"],
            end_page=data["end_page"],
            char_start=data["char_start"],
            char_end=data["char_end"],
            chunking_strategy=data.get("chunking_strategy", "recursive"),
            chunk_size=data.get("chunk_size", 1000),
            chunk_overlap=data.get("chunk_overlap", 150),
            metadata=dict(data.get("metadata", {})),
            embedding=data.get("embedding"),
        )


@dataclass
class IndexManifest:
    """Metadata manifest describing vector index identity, configuration, and commit state.
    
    Includes commit status sentinel ('staging' vs 'committed') and SHA-256 checksums
    to guarantee crash consistency and detect partial writes.
    """
    model_name: str
    dimension: int
    similarity_metric: str
    normalize_embeddings: bool
    index_type: str
    total_vectors: int
    total_documents: int
    document_ids: List[str]
    created_at: str
    updated_at: str
    status: str = "committed"  # Commit sentinel: 'staging' or 'committed'
    snapshot_version: str = "1.0"
    checksums: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexManifest":
        """Instantiate IndexManifest from a dictionary."""
        return cls(
            model_name=data["model_name"],
            dimension=data["dimension"],
            similarity_metric=data["similarity_metric"],
            normalize_embeddings=data["normalize_embeddings"],
            index_type=data["index_type"],
            total_vectors=data["total_vectors"],
            total_documents=data["total_documents"],
            document_ids=list(data.get("document_ids", [])),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            status=data.get("status", "committed"),
            snapshot_version=data.get("snapshot_version", "1.0"),
            checksums=data.get("checksums"),
        )


@dataclass(frozen=True)
class SearchResult:
    """Ranked search result combining cosine score and full VectorRecord metadata."""
    record: VectorRecord
    score: float


@dataclass(frozen=True)
class IndexingResult:
    """Structured telemetry returned by document and chunk indexing operations."""
    total_received: int
    newly_indexed: int
    intra_batch_duplicates_skipped: int
    existing_duplicates_skipped: int
    empty_chunks_skipped: int
