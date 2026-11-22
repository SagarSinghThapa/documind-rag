"""Ingestion pipeline orchestrator coordinating PDF loading, normalization, and chunking.

Provides a unified, reusable entry point returning structured chunks and comprehensive
execution reports.
"""

import time
from pathlib import Path
from typing import List, Tuple, Union, Optional

from documind.config import IngestionConfig
from documind.ingestion.schemas import (
    DocumentPage,
    TextChunk,
    DocumentReport,
    IngestionReport,
)
from documind.ingestion.pdf_loader import PDFLoader
from documind.ingestion.text_normalizer import TextNormalizer
from documind.ingestion.chunker import (
    BaseChunker,
    create_chunker,
)


class IngestionPipeline:
    """End-to-end document ingestion pipeline."""

    def __init__(
        self,
        config: Optional[IngestionConfig] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        chunking_strategy: Optional[str] = None,
        pdf_loader: Optional[PDFLoader] = None,
        normalizer: Optional[TextNormalizer] = None,
    ):
        """Initialize the ingestion pipeline.
        
        Args:
            config: Optional IngestionConfig; defaults to standard IngestionConfig().
            chunk_size: Optional override for chunk size in characters.
            chunk_overlap: Optional override for chunk overlap in characters.
            chunking_strategy: Optional override for chunking strategy ('recursive' or 'fixed').
            pdf_loader: Optional custom PDFLoader instance.
            normalizer: Optional custom TextNormalizer instance.
        """
        self.config = config or IngestionConfig()

        self.chunk_size = chunk_size if chunk_size is not None else self.config.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else self.config.chunk_overlap
        self.chunking_strategy = chunking_strategy if chunking_strategy is not None else self.config.chunking_strategy

        self.normalizer = normalizer or TextNormalizer()
        self.loader = pdf_loader or PDFLoader(
            normalizer=self.normalizer,
            max_file_size_bytes=self.config.max_file_size_bytes,
            extract_empty_pages=self.config.extract_empty_pages,
        )
        self.chunker: BaseChunker = create_chunker(
            strategy=self.chunking_strategy,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.config.separators,
        )

    def ingest_file(self, file_path: Union[str, Path]) -> Tuple[List[TextChunk], DocumentReport]:
        """Ingest and chunk a single PDF file.
        
        Args:
            file_path: Path to target PDF file.
            
        Returns:
            Tuple of (chunks: List[TextChunk], report: DocumentReport).
        """
        path = Path(file_path)
        pages, doc_report = self.loader.load_pdf(path)

        if not doc_report.success:
            return [], doc_report

        if not pages:
            # File had 0 valid extractable pages
            doc_report.chunks_generated = 0
            return [], doc_report

        try:
            chunks = self.chunker.chunk_document(pages)
            doc_report.chunks_generated = len(chunks)
            return chunks, doc_report
        except Exception as e:
            doc_report.success = False
            doc_report.error_message = f"Chunking error: {e}"
            doc_report.chunks_generated = 0
            return [], doc_report

    def ingest_files(self, file_paths: List[Union[str, Path]]) -> Tuple[List[TextChunk], IngestionReport]:
        """Ingest, normalize, and chunk multiple PDF files.
        
        Args:
            file_paths: List of file paths.
            
        Returns:
            Tuple of (all_chunks: List[TextChunk], report: IngestionReport).
        """
        t0 = time.perf_counter()
        all_chunks: List[TextChunk] = []
        doc_reports: List[DocumentReport] = []
        errors: List[dict] = []

        pages_extracted_total = 0
        empty_pages_total = 0
        docs_processed = 0
        docs_failed = 0

        for file_path in file_paths:
            path = Path(file_path)
            chunks, doc_report = self.ingest_file(path)
            doc_reports.append(doc_report)

            if doc_report.success:
                docs_processed += 1
                all_chunks.extend(chunks)
                pages_extracted_total += doc_report.pages_extracted
                empty_pages_total += doc_report.empty_pages
                if doc_report.error_message:
                    # Non-fatal warning (e.g. OCR needed for scanned PDF)
                    errors.append({
                        "file": str(path),
                        "type": "warning",
                        "message": doc_report.error_message,
                    })
            else:
                docs_failed += 1
                errors.append({
                    "file": str(path),
                    "type": "error",
                    "message": doc_report.error_message or "Unknown processing failure",
                })

        duration = time.perf_counter() - t0

        ingestion_report = IngestionReport(
            documents_submitted=len(file_paths),
            documents_processed=docs_processed,
            documents_failed=docs_failed,
            pages_extracted=pages_extracted_total,
            empty_pages=empty_pages_total,
            chunks_generated=len(all_chunks),
            errors=errors,
            processing_time_sec=duration,
            document_reports=doc_reports,
        )

        return all_chunks, ingestion_report
