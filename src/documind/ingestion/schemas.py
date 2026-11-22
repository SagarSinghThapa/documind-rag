"""Data schemas and transfer models for the document ingestion pipeline.

Uses standard Python dataclasses for lightweight execution without heavy external dependencies.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class DocumentPage:
    """Represents a single extracted page from a source document.
    
    Attributes:
        document_id: Deterministic SHA-256 hash of the source document content.
        document_name: Base filename of the source PDF.
        source_path: Full or relative filesystem path to the file.
        page_number: 1-indexed integer indicating the physical page.
        text: Normalized plain text extracted from the page.
        metadata: Additional diagnostic or structural metadata.
    """
    document_id: str
    document_name: str
    source_path: str
    page_number: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert page record to a dictionary."""
        return asdict(self)


@dataclass
class TextChunk:
    """Represents a discrete text chunk prepared for embedding and retrieval.
    
    Attributes:
        chunk_id: Deterministic identifier unique to this document, chunk index, and settings.
        document_id: Source document SHA-256 identifier.
        document_name: Source document filename.
        source_path: Path to the source file.
        chunk_index: 0-indexed position of this chunk within the document sequence.
        text: Plain text content of the chunk.
        page_numbers: List of all 1-indexed page numbers contributing text to this chunk.
        start_page: Earliest contributing page number.
        end_page: Latest contributing page number.
        chunking_strategy: Name of algorithm ('recursive' or 'fixed').
        chunk_size: Target chunk size (in characters).
        chunk_overlap: Target chunk overlap (in characters).
        char_start: Character offset start in the aggregated document text.
        char_end: Character offset end in the aggregated document text.
        metadata: Extensible metadata dictionary for downstream stages.
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
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int
    char_start: int = 0
    char_end: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk record to a dictionary."""
        return asdict(self)


@dataclass
class DocumentReport:
    """Per-document processing status and metrics."""
    document_id: str
    document_name: str
    source_path: str
    success: bool
    total_pages: int = 0
    pages_extracted: int = 0
    empty_pages: int = 0
    chunks_generated: int = 0
    error_message: Optional[str] = None
    extraction_time_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to a dictionary."""
        return asdict(self)


@dataclass
class IngestionReport:
    """Comprehensive summary of an ingestion pipeline run across one or more documents."""
    documents_submitted: int = 0
    documents_processed: int = 0
    documents_failed: int = 0
    pages_extracted: int = 0
    empty_pages: int = 0
    chunks_generated: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    processing_time_sec: float = 0.0
    document_reports: List[DocumentReport] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert ingestion report to a dictionary."""
        return asdict(self)
