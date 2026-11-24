"""Multi-PDF upload management, SHA-256 deduplication, quarantine, and index persistence."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import logging
import os
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from documind.ingestion.pdf_loader import PDFExtractionError, PDFLoader
from documind.ingestion.pipeline import IngestionPipeline
from documind.storage import DocumentIndex
from documind.ui.state import save_persistent_registry

logger = logging.getLogger(__name__)

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB limit per file


@dataclass
class IngestionSummary:
    """Summary of a batch ingestion operation across uploaded files."""
    total_files: int = 0
    indexed_files: int = 0
    skipped_files: int = 0
    total_chunks: int = 0
    errors: List[str] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)


class IngestionManager:
    """Orchestrates safe file upload quarantine, deduplication, chunking, and FAISS persistence."""

    def __init__(
        self,
        pipeline: Optional[IngestionPipeline] = None,
        upload_dir: Optional[str] = None,
        index_dir: Optional[str] = None,
        max_file_size_bytes: int = MAX_FILE_SIZE_BYTES,
    ) -> None:
        self.pipeline = pipeline or IngestionPipeline(chunk_size=1000, chunk_overlap=150)
        self.upload_dir = Path(upload_dir or os.environ.get("DOCUMIND_UPLOAD_DIR", "data/uploads"))
        self.index_dir = Path(index_dir or os.environ.get("DOCUMIND_INDEX_DIR", "data/index"))
        self.max_file_size_bytes = max_file_size_bytes

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_file_hash(file_bytes: bytes) -> str:
        """Compute SHA-256 hex digest of file bytes before any disk I/O."""
        hasher = hashlib.sha256()
        hasher.update(file_bytes)
        return hasher.hexdigest()

    @staticmethod
    def sanitize_filename(name: str) -> str:
        """Sanitize uploaded filename against directory traversal and control characters."""
        base_name = Path(name).name
        # Keep letters, numbers, hyphens, underscores, dots
        sanitized = re.sub(r"[^a-zA-Z0-9_\.-]", "_", base_name)
        if not sanitized.lower().endswith(".pdf"):
            sanitized += ".pdf"
        return sanitized

    def save_uploaded_file(self, uploaded_file: Any, file_hash: str) -> Path:
        """Save uploaded file buffer to quarantine directory under deterministic hash prefix."""
        # Handle Path or str
        if isinstance(uploaded_file, (str, Path)):
            path_obj = Path(uploaded_file)
            data = path_obj.read_bytes()
            safe_name = self.sanitize_filename(path_obj.name)
        elif hasattr(uploaded_file, "getbuffer"):
            safe_name = self.sanitize_filename(getattr(uploaded_file, "name", "uploaded.pdf"))
            data = uploaded_file.getbuffer()
        elif hasattr(uploaded_file, "read"):
            safe_name = self.sanitize_filename(getattr(uploaded_file, "name", "uploaded.pdf"))
            data = uploaded_file.read()
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
        elif isinstance(uploaded_file, (bytes, bytearray)):
            safe_name = "uploaded.pdf"
            data = uploaded_file
        else:
            raise ValueError(f"Unsupported uploaded file type: {type(uploaded_file)}")

        target_path = self.upload_dir / f"{file_hash[:12]}_{safe_name}"

        if len(data) > self.max_file_size_bytes:
            limit_mb = self.max_file_size_bytes / (1024 * 1024)
            actual_mb = len(data) / (1024 * 1024)
            raise ValueError(
                f"File '{safe_name}' ({actual_mb:.2f} MB) exceeds maximum allowed size ({limit_mb:.2f} MB)."
            )

        with open(target_path, "wb") as f:
            f.write(data)

        return target_path

    def ingest_and_index(
        self,
        uploaded_files: List[Any],
        document_index: DocumentIndex,
        registry: Dict[str, Any],
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> IngestionSummary:
        """Ingest multiple uploaded PDF files, updating DocumentIndex and registry with persistence.
        
        Args:
            uploaded_files: List of Streamlit UploadedFile objects or file paths.
            document_index: Target DocumentIndex instance.
            registry: Session document registry dict to update.
            progress_callback: Optional callable(status_text, fraction) for UI progress bars.
            
        Returns:
            IngestionSummary containing counts, chunk totals, and warning/error messages.
        """
        summary = IngestionSummary(total_files=len(uploaded_files))
        if not uploaded_files:
            return summary

        total_files = len(uploaded_files)

        for i, file_obj in enumerate(uploaded_files):
            file_name = getattr(file_obj, "name", str(file_obj))
            fraction = (i + 1) / total_files
            if progress_callback:
                progress_callback(f"Processing {file_name}...", fraction)

            try:
                # 1. Read bytes & compute cryptographic hash
                if isinstance(file_obj, (str, Path)):
                    with open(file_obj, "rb") as f:
                        file_bytes = f.read()
                    file_name = Path(file_obj).name
                elif hasattr(file_obj, "getvalue"):
                    file_bytes = file_obj.getvalue()
                elif hasattr(file_obj, "read"):
                    file_bytes = file_obj.read()
                    if hasattr(file_obj, "seek"):
                        file_obj.seek(0)
                else:
                    file_bytes = bytes(file_obj)

                file_hash = self.compute_file_hash(file_bytes)

                # 2. Duplicate Detection Check (Document-level)
                if file_hash in registry:
                    summary.skipped_files += 1
                    msg = f"Document '{file_name}' already indexed (Hash: {file_hash[:8]}). Skipping."
                    summary.messages.append(msg)
                    logger.info(msg)
                    continue

                # 3. Save to quarantine
                saved_path = self.save_uploaded_file(file_obj, file_hash)

                # 4. Ingestion Pre-validation & Extraction
                is_valid, err_msg = self.pipeline.loader.validate_pdf(saved_path)
                if not is_valid:
                    summary.errors.append(f"Validation failed for '{file_name}': {err_msg}")
                    # Remove invalid quarantine file
                    if saved_path.exists():
                        saved_path.unlink()
                    continue

                # Ingest through full pipeline (extract, normalize, chunk)
                chunks, doc_report = self.pipeline.ingest_file(saved_path)

                # Check for scanned image-only PDFs
                if doc_report.chunks_generated == 0 or len(chunks) == 0:
                    notice = (
                        f"PDF '{file_name}' contains no extractable text (likely a scanned image). "
                        f"Scanned image processing requires OCR (scheduled for Milestone 2). Skipping vector indexing."
                    )
                    summary.messages.append(notice)
                    summary.skipped_files += 1
                    continue

                # 5. Vector Store Insertion (with chunk deduplication)
                indexing_res = document_index.add_chunks(chunks)

                # 6. Update Registry Metadata
                registry[file_hash] = {
                    "document_id": file_hash,
                    "filename": file_name,
                    "pages": doc_report.total_pages,
                    "chunks": indexing_res.newly_indexed,
                    "size_bytes": len(file_bytes),
                    "file_path": str(saved_path),
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                }

                summary.indexed_files += 1
                summary.total_chunks += indexing_res.newly_indexed
                summary.messages.append(
                    f"Successfully indexed '{file_name}': {indexing_res.newly_indexed} chunks ({doc_report.total_pages} pages)."
                )

            except PDFExtractionError as e:
                err = f"Extraction error on '{file_name}': {e}"
                summary.errors.append(err)
                logger.error(err)
            except Exception as e:
                err = f"Unexpected error processing '{file_name}': {e}"
                summary.errors.append(err)
                logger.error(err, exc_info=True)

        # 7. Flush Persistent Snapshot and Registry if new documents were indexed
        if summary.indexed_files > 0:
            try:
                document_index.vector_store.save(self.index_dir)
                save_persistent_registry(registry, index_dir=str(self.index_dir))
                logger.info(
                    "Persisted active vector store snapshot and registry to %s", self.index_dir
                )
            except Exception as e:
                logger.error("Failed to persist vector store snapshot: %s", e)

        if progress_callback:
            progress_callback("Ingestion complete", 1.0)

        return summary
