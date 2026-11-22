"""DocuMind Ingestion Package: PDF extraction, normalization, and chunking."""

from documind.ingestion.schemas import (
    DocumentPage,
    TextChunk,
    DocumentReport,
    IngestionReport,
)
from documind.ingestion.text_normalizer import TextNormalizer
from documind.ingestion.pdf_loader import PDFLoader
from documind.ingestion.chunker import (
    BaseChunker,
    FixedSizeChunker,
    RecursiveCharacterChunker,
    create_chunker,
)
from documind.ingestion.pipeline import IngestionPipeline

__all__ = [
    "DocumentPage",
    "TextChunk",
    "DocumentReport",
    "IngestionReport",
    "TextNormalizer",
    "PDFLoader",
    "BaseChunker",
    "FixedSizeChunker",
    "RecursiveCharacterChunker",
    "create_chunker",
    "IngestionPipeline",
]
