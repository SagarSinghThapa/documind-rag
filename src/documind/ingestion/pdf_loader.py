"""PDF text extraction and validation engine using PyMuPDF.

Provides robust page-by-page extraction, deterministic document hashing,
invariant 1-indexed page preservation, empty page detection, and graceful
handling of encrypted or malformed files.
"""

import hashlib
import os
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import pymupdf

from documind.ingestion.schemas import DocumentPage, DocumentReport
from documind.ingestion.text_normalizer import TextNormalizer


class PDFExtractionError(Exception):
    """Custom exception raised when PDF extraction fails irrecoverably."""
    pass


class PDFLoader:
    """Extracts text and metadata page-by-page from PDF files using PyMuPDF."""

    PDF_MAGIC_BYTES = b"%PDF-"

    def __init__(
        self,
        normalizer: Optional[TextNormalizer] = None,
        max_file_size_bytes: int = 50 * 1024 * 1024,  # 50 MB
        extract_empty_pages: bool = False,
    ):
        """Initialize PDFLoader.
        
        Args:
            normalizer: TextNormalizer instance; defaults to new instance.
            max_file_size_bytes: Maximum allowed file size in bytes.
            extract_empty_pages: Whether to emit DocumentPage objects for blank pages.
        """
        self.normalizer = normalizer or TextNormalizer()
        self.max_file_size_bytes = max_file_size_bytes
        self.extract_empty_pages = extract_empty_pages

    @staticmethod
    def compute_document_id(file_path: Path) -> str:
        """Compute a deterministic SHA-256 hash of the PDF file content.
        
        Args:
            file_path: Path to the target PDF file.
            
        Returns:
            Hexadecimal SHA-256 string.
        """
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def validate_pdf(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """Validate that a file exists, is non-empty, and has a valid PDF signature.
        
        Args:
            file_path: Path to target file.
            
        Returns:
            Tuple of (is_valid: bool, error_message: Optional[str]).
        """
        if not file_path.exists():
            return False, f"File does not exist: {file_path}"

        if not file_path.is_file():
            return False, f"Path is not a regular file: {file_path}"

        if file_path.suffix.lower() != ".pdf":
            return False, f"Unsupported file extension '{file_path.suffix}'; expected '.pdf'"

        try:
            file_size = file_path.stat().st_size
        except OSError as e:
            return False, f"Unable to read file size: {e}"

        if file_size == 0:
            return False, "File is empty (0 bytes)"

        if file_size > self.max_file_size_bytes:
            limit_mb = self.max_file_size_bytes / (1024 * 1024)
            actual_mb = file_size / (1024 * 1024)
            return False, f"File size ({actual_mb:.2f} MB) exceeds maximum allowed limit ({limit_mb:.2f} MB)"

        # Check magic bytes
        try:
            with open(file_path, "rb") as f:
                header = f.read(1024)
                if self.PDF_MAGIC_BYTES not in header:
                    return False, "Invalid PDF header signature (missing '%PDF-')"
        except OSError as e:
            return False, f"Error reading file header: {e}"

        return True, None

    def load_pdf(self, file_path: Path) -> Tuple[List[DocumentPage], DocumentReport]:
        """Extract all pages and metadata from a single PDF document.
        
        Args:
            file_path: Pathlib Path to target PDF.
            
        Returns:
            Tuple of (pages: List[DocumentPage], report: DocumentReport).
        """
        t0 = time.perf_counter()
        doc_name = file_path.name
        source_path_str = str(file_path.resolve())

        # 1. Validation
        is_valid, validation_error = self.validate_pdf(file_path)
        if not is_valid:
            duration = time.perf_counter() - t0
            report = DocumentReport(
                document_id="invalid",
                document_name=doc_name,
                source_path=source_path_str,
                success=False,
                error_message=validation_error,
                extraction_time_sec=duration,
            )
            return [], report

        # 2. Compute deterministic document ID
        try:
            doc_id = self.compute_document_id(file_path)
        except Exception as e:
            duration = time.perf_counter() - t0
            report = DocumentReport(
                document_id="error",
                document_name=doc_name,
                source_path=source_path_str,
                success=False,
                error_message=f"Failed to compute document hash: {e}",
                extraction_time_sec=duration,
            )
            return [], report

        # 3. Open and Extract with PyMuPDF
        doc = None
        pages: List[DocumentPage] = []
        empty_page_count = 0
        total_extracted_chars = 0

        try:
            doc = pymupdf.open(file_path)

            # Check encryption
            if doc.is_encrypted or doc.needs_pass:
                duration = time.perf_counter() - t0
                report = DocumentReport(
                    document_id=doc_id,
                    document_name=doc_name,
                    source_path=source_path_str,
                    success=False,
                    total_pages=len(doc) if doc else 0,
                    error_message="Document is encrypted / password-protected and cannot be opened without credentials.",
                    extraction_time_sec=duration,
                )
                return [], report

            total_pages = len(doc)
            if total_pages == 0:
                duration = time.perf_counter() - t0
                report = DocumentReport(
                    document_id=doc_id,
                    document_name=doc_name,
                    source_path=source_path_str,
                    success=False,
                    total_pages=0,
                    error_message="Document contains 0 pages.",
                    extraction_time_sec=duration,
                )
                return [], report

            # Extract page-by-page preserving 1-indexed page numbers
            for page_index in range(total_pages):
                page_number = page_index + 1  # 1-indexed
                page = doc[page_index]

                raw_text = page.get_text("text") or ""
                normalized_text = self.normalizer.normalize(raw_text)

                char_count = len(normalized_text)
                total_extracted_chars += char_count

                if char_count == 0:
                    empty_page_count += 1
                    if not self.extract_empty_pages:
                        # Skip adding to pages list, but page numbering of other pages remains invariant
                        continue

                page_record = DocumentPage(
                    document_id=doc_id,
                    document_name=doc_name,
                    source_path=source_path_str,
                    page_number=page_number,
                    text=normalized_text,
                    metadata={
                        "raw_char_count": len(raw_text),
                        "char_count": char_count,
                        "is_empty": (char_count == 0),
                        "width": page.rect.width,
                        "height": page.rect.height,
                    },
                )
                pages.append(page_record)

            duration = time.perf_counter() - t0

            # Detect scanned / image-only PDFs with 0 extractable text
            error_message = None
            if total_extracted_chars == 0 and total_pages > 0:
                error_message = (
                    "Document contains no extractable text across all pages. "
                    "Scanned image pages require Optical Character Recognition (OCR), "
                    "which is outside the scope of this ingestion pipeline."
                )

            report = DocumentReport(
                document_id=doc_id,
                document_name=doc_name,
                source_path=source_path_str,
                success=True,
                total_pages=total_pages,
                pages_extracted=len(pages),
                empty_pages=empty_page_count,
                error_message=error_message,
                extraction_time_sec=duration,
            )
            return pages, report

        except Exception as e:
            duration = time.perf_counter() - t0
            report = DocumentReport(
                document_id=doc_id,
                document_name=doc_name,
                source_path=source_path_str,
                success=False,
                error_message=f"PyMuPDF extraction error: {e}",
                extraction_time_sec=duration,
            )
            return [], report

        finally:
            if doc is not None:
                doc.close()
