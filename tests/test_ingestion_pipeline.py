"""Automated integration tests for IngestionPipeline using synthetic and public research PDFs."""

import pytest
from pathlib import Path
import pymupdf

from documind.config import IngestionConfig
from documind.ingestion.pipeline import IngestionPipeline
from documind.ingestion.schemas import TextChunk, IngestionReport


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary workspace directory for synthetic PDFs."""
    return tmp_path


def create_synthetic_pdf(path: Path, pages_text: list[str]) -> Path:
    """Helper to generate synthetic PDF files on the fly."""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_text(pymupdf.Point(50, 72), text)
    doc.save(str(path))
    doc.close()
    return path


class TestIngestionPipelineUnit:
    """Unit tests for pipeline orchestration with synthetic files."""

    def test_single_file_ingestion(self, temp_dir):
        """Pipeline ingests a valid single PDF and returns chunks and document report."""
        pdf_path = temp_dir / "doc1.pdf"
        create_synthetic_pdf(
            pdf_path,
            [
                "Page one introduces retrieval-augmented generation.",
                "Page two explains chunking and embedding strategies.",
            ],
        )

        pipeline = IngestionPipeline(chunk_size=100, chunk_overlap=20, chunking_strategy="recursive")
        chunks, doc_report = pipeline.ingest_file(pdf_path)

        assert doc_report.success is True
        assert doc_report.total_pages == 2
        assert doc_report.pages_extracted == 2
        assert doc_report.chunks_generated == len(chunks)
        assert len(chunks) >= 2
        assert all(isinstance(c, TextChunk) for c in chunks)
        assert chunks[0].document_name == "doc1.pdf"

    def test_batch_ingestion_with_mixed_results(self, temp_dir):
        """Batch ingestion processes valid and handles invalid files without stopping."""
        valid_pdf = temp_dir / "valid.pdf"
        create_synthetic_pdf(valid_pdf, ["Valid document text content here."])

        corrupt_pdf = temp_dir / "corrupt.pdf"
        corrupt_pdf.write_bytes(b"NOT_A_PDF_STREAM")

        ghost_pdf = temp_dir / "non_existent.pdf"

        pipeline = IngestionPipeline(chunk_size=200, chunk_overlap=30)
        chunks, report = pipeline.ingest_files([valid_pdf, corrupt_pdf, ghost_pdf])

        assert isinstance(report, IngestionReport)
        assert report.documents_submitted == 3
        assert report.documents_processed == 1
        assert report.documents_failed == 2
        assert report.pages_extracted == 1
        assert report.chunks_generated == len(chunks)
        assert len(report.errors) == 2
        assert report.processing_time_sec > 0.0

    def test_custom_config_override(self, temp_dir):
        """Pipeline respects IngestionConfig settings."""
        pdf_path = temp_dir / "custom_config.pdf"
        create_synthetic_pdf(pdf_path, ["Custom configuration test."])

        custom_cfg = IngestionConfig(
            chunk_size=50,
            chunk_overlap=10,
            chunking_strategy="fixed",
        )
        pipeline = IngestionPipeline(config=custom_cfg)

        assert pipeline.chunk_size == 50
        assert pipeline.chunk_overlap == 10
        assert pipeline.chunking_strategy == "fixed"

        chunks, report = pipeline.ingest_file(pdf_path)
        assert report.success is True
        assert chunks[0].chunking_strategy == "fixed"


class TestPublicResearchPDFsIntegration:
    """Integration test suite executing against public research PDFs in the archive."""

    RESEARCH_DIR = Path(__file__).resolve().parent.parent / "research" / "technical_references"

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parent.parent / "research" / "technical_references" / "BASWE-15-RAG-Concepts-Guide.pdf").exists(),
        reason="Research PDF not found in local workspace",
    )
    def test_ingest_baswe_15_rag_guide(self):
        """Integration test on BASWE-15-RAG-Concepts-Guide.pdf (14 pages)."""
        pdf_path = self.RESEARCH_DIR / "BASWE-15-RAG-Concepts-Guide.pdf"
        assert pdf_path.exists(), f"Missing expected research file: {pdf_path}"

        pipeline = IngestionPipeline(chunk_size=1000, chunk_overlap=150, chunking_strategy="recursive")
        chunks, doc_report = pipeline.ingest_file(pdf_path)

        assert doc_report.success is True
        assert doc_report.total_pages == 14
        assert doc_report.pages_extracted == 14
        assert doc_report.empty_pages == 0
        assert len(chunks) > 0
        assert doc_report.chunks_generated == len(chunks)

        # Verify chunk structure and metadata traceability
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.document_name == "BASWE-15-RAG-Concepts-Guide.pdf"
            assert len(chunk.document_id) == 64
            assert len(chunk.page_numbers) >= 1
            assert chunk.start_page >= 1
            assert chunk.end_page <= 14
            assert chunk.start_page <= chunk.end_page
            assert chunk.chunking_strategy == "recursive"
            assert len(chunk.text) > 0

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parent.parent / "research" / "technical_references" / "Everything-to-know-about-RAG.pdf").exists(),
        reason="Research PDF not found in local workspace",
    )
    def test_ingest_everything_to_know_about_rag(self):
        """Integration test on Everything-to-know-about-RAG.pdf (92 pages)."""
        pdf_path = self.RESEARCH_DIR / "Everything-to-know-about-RAG.pdf"
        assert pdf_path.exists(), f"Missing expected research file: {pdf_path}"

        pipeline = IngestionPipeline(chunk_size=1200, chunk_overlap=200, chunking_strategy="recursive")
        chunks, doc_report = pipeline.ingest_file(pdf_path)

        assert doc_report.success is True
        assert doc_report.total_pages == 92
        assert doc_report.pages_extracted > 80  # Most pages have text
        assert len(chunks) > 0
        assert doc_report.chunks_generated == len(chunks)

        # Check multi-page chunks existence and validity
        multipage_chunks = [c for c in chunks if c.metadata.get("is_multipage", False)]
        assert len(multipage_chunks) > 0
        for mc in multipage_chunks:
            assert len(mc.page_numbers) > 1
            assert mc.start_page < mc.end_page

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parent.parent / "research" / "technical_references").exists(),
        reason="Research directory not found",
    )
    def test_batch_ingest_multiple_public_pdfs(self):
        """Integration test executing batch ingestion across multiple public reference PDFs."""
        pdf1 = self.RESEARCH_DIR / "BASWE-15-RAG-Concepts-Guide.pdf"
        pdf2 = self.RESEARCH_DIR / "Everything-to-know-about-RAG.pdf"

        files = [p for p in [pdf1, pdf2] if p.exists()]
        if len(files) < 2:
            pytest.skip("Not enough public research files found for batch test")

        pipeline = IngestionPipeline(chunk_size=1000, chunk_overlap=150)
        chunks, report = pipeline.ingest_files(files)

        assert report.documents_submitted == len(files)
        assert report.documents_processed == len(files)
        assert report.documents_failed == 0
        assert report.chunks_generated == len(chunks)
        assert report.pages_extracted > 0
        assert len(report.document_reports) == len(files)
