"""Unit tests for UI state initialization, IngestionManager, and safe index lifecycle."""

import io
from pathlib import Path
import pytest
import streamlit as st

from documind.storage import DocumentIndex, FaissVectorStore, LocalEmbedder
from documind.ui.ingestion_manager import IngestionManager
from documind.ui.state import (
    init_session_state,
    load_persistent_registry,
    safe_reset_index,
    save_persistent_registry,
)


class MockUploadedFile:
    """Mock Streamlit UploadedFile buffer."""

    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data

    def getbuffer(self) -> bytes:
        return self._data

    def read(self) -> bytes:
        return self._data


@pytest.fixture
def test_dirs(tmp_path):
    upload_dir = tmp_path / "uploads"
    index_dir = tmp_path / "index"
    upload_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir, index_dir


class TestIngestionManagerUnit:
    """Unit tests for IngestionManager file handling, deduplication, and persistence."""

    def test_save_uploaded_file_sanitization(self, test_dirs):
        upload_dir, index_dir = test_dirs
        mgr = IngestionManager(upload_dir=str(upload_dir), index_dir=str(index_dir))

        file_data = b"%PDF-1.4 test mock content"
        file_hash = mgr.compute_file_hash(file_data)
        mock_file = MockUploadedFile("../../unsafe_path/test report.pdf", file_data)

        saved = mgr.save_uploaded_file(mock_file, file_hash)
        assert saved.exists()
        # Filename must be sanitized against path traversal and spaces
        assert ".." not in saved.name
        assert "unsafe_path" not in saved.name
        assert saved.name.endswith(".pdf")
        assert saved.read_bytes() == file_data

    def test_duplicate_document_skipping(self, test_dirs):
        upload_dir, index_dir = test_dirs
        mgr = IngestionManager(upload_dir=str(upload_dir), index_dir=str(index_dir))

        file_data = b"%PDF-1.4 test duplicate"
        file_hash = mgr.compute_file_hash(file_data)

        # Pre-seed registry with this hash
        registry = {
            file_hash: {
                "document_id": file_hash,
                "filename": "existing.pdf",
                "chunks": 5,
            }
        }
        embedder = LocalEmbedder()
        store = FaissVectorStore(dimension=embedder.dimension)
        doc_index = DocumentIndex(embedder=embedder, vector_store=store)

        mock_file = MockUploadedFile("existing.pdf", file_data)
        summary = mgr.ingest_and_index([mock_file], doc_index, registry)

        assert summary.total_files == 1
        assert summary.skipped_files == 1
        assert summary.indexed_files == 0
        assert any("already indexed" in msg for msg in summary.messages)

    def test_corrupt_file_handling(self, test_dirs):
        upload_dir, index_dir = test_dirs
        mgr = IngestionManager(upload_dir=str(upload_dir), index_dir=str(index_dir))

        # 0-byte file
        corrupt_file = MockUploadedFile("empty.pdf", b"")
        registry = {}
        embedder = LocalEmbedder()
        store = FaissVectorStore(dimension=embedder.dimension)
        doc_index = DocumentIndex(embedder=embedder, vector_store=store)

        summary = mgr.ingest_and_index([corrupt_file], doc_index, registry)
        assert len(summary.errors) == 1
        assert "Validation failed" in summary.errors[0]

    def test_scanned_pdf_zero_text_handling(self, test_dirs, monkeypatch):
        upload_dir, index_dir = test_dirs
        mgr = IngestionManager(upload_dir=str(upload_dir), index_dir=str(index_dir))

        # Mock IngestionPipeline to simulate image-only PDF with 0 extractable chars
        from documind.ingestion.schemas import DocumentReport
        def mock_ingest(file_path):
            return [], DocumentReport(
                document_id="doc_scanned",
                document_name="scanned.pdf",
                source_path=str(file_path),
                success=True,
                total_pages=3,
                pages_extracted=0,
                empty_pages=3,
                chunks_generated=0,
            )
        monkeypatch.setattr(mgr.pipeline, "ingest_file", mock_ingest)
        monkeypatch.setattr(mgr.pipeline.loader, "validate_pdf", lambda p: (True, None))

        mock_file = MockUploadedFile("scanned.pdf", b"%PDF-1.4 scanned content")
        registry = {}
        embedder = LocalEmbedder()
        store = FaissVectorStore(dimension=embedder.dimension)
        doc_index = DocumentIndex(embedder=embedder, vector_store=store)

        summary = mgr.ingest_and_index([mock_file], doc_index, registry)
        assert summary.skipped_files == 1
        assert any("OCR" in msg for msg in summary.messages)

    def test_real_pdf_ingestion_and_snapshot_persistence(self, test_dirs):
        upload_dir, index_dir = test_dirs
        mgr = IngestionManager(upload_dir=str(upload_dir), index_dir=str(index_dir))

        real_pdf_path = Path("research/technical_references/BASWE-15-RAG-Concepts-Guide.pdf")
        if not real_pdf_path.exists():
            pytest.skip("Technical reference PDF not available for test")

        embedder = LocalEmbedder()
        store = FaissVectorStore(dimension=embedder.dimension)
        doc_index = DocumentIndex(embedder=embedder, vector_store=store)
        registry = {}

        summary = mgr.ingest_and_index([real_pdf_path], doc_index, registry)
        assert summary.indexed_files == 1
        assert summary.total_chunks > 0
        assert doc_index.total_vectors == summary.total_chunks

        # Verify disk persistence
        active_snapshot_ptr = index_dir / "active_snapshot.json"
        assert active_snapshot_ptr.exists()
        persisted_reg = load_persistent_registry(str(index_dir))
        assert len(persisted_reg) == 1


class TestSafeResetUnit:
    """Unit tests for safe index reset deletion boundaries."""

    def test_safe_reset_index_scope(self, tmp_path):
        data_dir = tmp_path / "data"
        upload_dir = data_dir / "uploads"
        index_dir = data_dir / "index"
        snapshots_dir = index_dir / "snapshots"
        models_dir = tmp_path / "models"  # Critical: must not be touched

        upload_dir.mkdir(parents=True, exist_ok=True)
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)

        # Seed files
        dummy_upload = upload_dir / "test_doc.pdf"
        dummy_upload.write_text("sample content")
        dummy_snap = snapshots_dir / "snapshot_01"
        dummy_snap.mkdir()
        (dummy_snap / "index.faiss").write_text("index data")
        active_ptr = index_dir / "active_snapshot.json"
        active_ptr.write_text('{"snapshot": "snapshot_01"}')
        save_persistent_registry({"doc1": {}}, str(index_dir))

        # Model weight dummy
        dummy_model_weight = models_dir / "model.safetensors"
        dummy_model_weight.write_text("weights")

        # Initialize minimal st.session_state for reset
        st.session_state["indexed_docs"] = {"doc1": {}}
        st.session_state["messages"] = [{"role": "user", "content": "hi"}]

        safe_reset_index(
            data_dir=str(data_dir),
            index_dir=str(index_dir),
            upload_dir=str(upload_dir),
        )

        # Asserts on session state
        assert st.session_state["indexed_docs"] == {}
        assert st.session_state["messages"] == []

        # Asserts on index and uploads
        assert not dummy_upload.exists()
        assert not active_ptr.exists()
        assert load_persistent_registry(str(index_dir)) == {}

        # CRITICAL SAFETY ASSERTION: Model weights MUST be untouched
        assert dummy_model_weight.exists()
        assert dummy_model_weight.read_text() == "weights"
