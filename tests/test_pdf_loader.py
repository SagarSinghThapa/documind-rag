"""Automated unit and integration tests for PDFLoader using synthetic test PDFs."""

import pytest
from pathlib import Path
import pymupdf

from documind.ingestion.pdf_loader import PDFLoader
from documind.ingestion.schemas import DocumentPage, DocumentReport


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary workspace directory for synthetic PDFs."""
    return tmp_path


@pytest.fixture
def loader():
    """Default PDFLoader instance."""
    return PDFLoader()


def create_synthetic_pdf(
    path: Path,
    pages_text: list[str],
    password: str = None,
) -> Path:
    """Helper to generate synthetic PDF files on the fly without external dependencies."""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            # Insert text at position (50, 72)
            page.insert_text(pymupdf.Point(50, 72), text)

    if password:
        # Encrypt the PDF
        perm = int(
            pymupdf.PDF_PERM_ACCESSIBILITY
            | pymupdf.PDF_PERM_PRINT
            | pymupdf.PDF_PERM_COPY
            | pymupdf.PDF_PERM_ANNOTATE
        )
        doc.save(
            str(path),
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw=password,
            user_pw=password,
            permissions=perm,
        )
    else:
        doc.save(str(path))
    doc.close()
    return path


class TestPDFLoader:
    """Test suite for PDFLoader extraction and validation."""

    def test_valid_single_page_pdf(self, temp_dir, loader):
        """Extract text from a valid single-page PDF."""
        pdf_path = temp_dir / "single_page.pdf"
        create_synthetic_pdf(pdf_path, ["This is page one text content."])

        pages, report = loader.load_pdf(pdf_path)

        assert report.success is True
        assert report.total_pages == 1
        assert report.pages_extracted == 1
        assert report.empty_pages == 0
        assert len(pages) == 1
        assert pages[0].page_number == 1
        assert "This is page one text content." in pages[0].text
        assert pages[0].document_name == "single_page.pdf"
        assert len(pages[0].document_id) == 64  # SHA-256

    def test_multi_page_pdf_with_page_ordering(self, temp_dir, loader):
        """Extract multi-page PDF and verify 1-indexed page ordering and content mapping."""
        pdf_path = temp_dir / "multipage.pdf"
        create_synthetic_pdf(
            pdf_path,
            [
                "Content of first page.",
                "Content of second page.",
                "Content of third page.",
            ],
        )

        pages, report = loader.load_pdf(pdf_path)

        assert report.success is True
        assert report.total_pages == 3
        assert report.pages_extracted == 3
        assert [p.page_number for p in pages] == [1, 2, 3]
        assert "first page" in pages[0].text
        assert "second page" in pages[1].text
        assert "third page" in pages[2].text

    def test_empty_page_handling_preserves_page_numbers(self, temp_dir, loader):
        """When an empty page is skipped, subsequent pages must retain their true 1-indexed numbers."""
        pdf_path = temp_dir / "with_blank_page.pdf"
        # Page 1 has text, Page 2 is empty, Page 3 has text
        create_synthetic_pdf(
            pdf_path,
            [
                "Page 1 has valid text.",
                "",  # Empty
                "Page 3 has valid text.",
            ],
        )

        pages, report = loader.load_pdf(pdf_path)

        assert report.success is True
        assert report.total_pages == 3
        assert report.empty_pages == 1
        assert report.pages_extracted == 2
        assert len(pages) == 2
        # Crucial check: page 3 must be numbered 3, not 2
        assert pages[0].page_number == 1
        assert pages[1].page_number == 3

    def test_extract_empty_pages_flag(self, temp_dir):
        """When extract_empty_pages=True, empty pages are returned as DocumentPage with text=''."""
        pdf_path = temp_dir / "blank_page_allowed.pdf"
        create_synthetic_pdf(pdf_path, ["Page 1", "", "Page 3"])

        loader = PDFLoader(extract_empty_pages=True)
        pages, report = loader.load_pdf(pdf_path)

        assert report.success is True
        assert report.pages_extracted == 3
        assert len(pages) == 3
        assert [p.page_number for p in pages] == [1, 2, 3]
        assert pages[1].text == ""
        assert pages[1].metadata["is_empty"] is True

    def test_pdf_with_no_extractable_text_flags_ocr_requirement(self, temp_dir, loader):
        """PDF with pages but zero text should notify clearly that OCR is required."""
        pdf_path = temp_dir / "scanned_or_empty.pdf"
        create_synthetic_pdf(pdf_path, ["", "", ""])

        pages, report = loader.load_pdf(pdf_path)

        assert report.success is True
        assert report.total_pages == 3
        assert report.pages_extracted == 0
        assert report.empty_pages == 3
        assert len(pages) == 0
        assert report.error_message is not None
        assert "OCR" in report.error_message

    def test_encrypted_pdf_handling(self, temp_dir, loader):
        """Encrypted PDF should fail gracefully with informative report rather than crashing."""
        pdf_path = temp_dir / "encrypted.pdf"
        create_synthetic_pdf(pdf_path, ["Secret content"], password="SecurePassword123!")

        pages, report = loader.load_pdf(pdf_path)

        assert report.success is False
        assert len(pages) == 0
        assert "encrypted" in report.error_message.lower()

    def test_malformed_header_pdf_rejected(self, temp_dir, loader):
        """Files missing the '%PDF-' magic bytes should be rejected during validation."""
        corrupt_path = temp_dir / "corrupted.pdf"
        corrupt_path.write_bytes(b"NOT A VALID PDF FILE AT ALL 123456789")

        pages, report = loader.load_pdf(corrupt_path)

        assert report.success is False
        assert len(pages) == 0
        assert "header signature" in report.error_message

    def test_empty_zero_byte_file_rejected(self, temp_dir, loader):
        """Zero-byte files should be rejected cleanly."""
        empty_path = temp_dir / "zero_bytes.pdf"
        empty_path.write_bytes(b"")

        pages, report = loader.load_pdf(empty_path)

        assert report.success is False
        assert "empty (0 bytes)" in report.error_message

    def test_non_existent_file_handling(self, temp_dir, loader):
        """Non-existent file should return a clean failure report."""
        ghost_path = temp_dir / "ghost.pdf"
        pages, report = loader.load_pdf(ghost_path)

        assert report.success is False
        assert "does not exist" in report.error_message

    def test_unsupported_file_extension(self, temp_dir, loader):
        """Non-pdf extension should be rejected cleanly."""
        txt_path = temp_dir / "notes.txt"
        txt_path.write_text("Hello world")

        pages, report = loader.load_pdf(txt_path)

        assert report.success is False
        assert "Unsupported file extension" in report.error_message

    def test_deterministic_document_id(self, temp_dir, loader):
        """Document ID must be a deterministic SHA-256 hash invariant across multiple runs."""
        pdf_path = temp_dir / "deterministic.pdf"
        create_synthetic_pdf(pdf_path, ["Deterministic content test."])

        id_1 = loader.compute_document_id(pdf_path)
        id_2 = loader.compute_document_id(pdf_path)

        assert id_1 == id_2
        assert len(id_1) == 64
