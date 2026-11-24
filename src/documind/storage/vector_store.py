"""Persistent FAISS vector store with immutable versioned snapshots and crash-consistency."""

import hashlib
import json
import os
import shutil
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Set KMP duplicate lib policy before importing OpenMP-linked libraries on Windows
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import faiss
import numpy as np

from documind.storage.embedder import validate_vectors
from documind.storage.schemas import IndexManifest, SearchResult, VectorRecord


class VectorStoreError(Exception):
    """Base exception for vector store operations."""
    pass


class IncompatibleIndexError(VectorStoreError):
    """Raised when loaded index does not match required model, dimension, or metric."""
    pass


class IndexCorruptionError(VectorStoreError):
    """Raised when index files, metadata, or checksums are missing or corrupted."""
    pass


def _compute_sha256(file_path: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class BaseVectorStore(ABC):
    """Abstract interface for persistent vector store implementations."""

    @property
    @abstractmethod
    def total_vectors(self) -> int:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @abstractmethod
    def add(self, vectors: np.ndarray, records: List[VectorRecord]) -> int:
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        pass

    @abstractmethod
    def save(self, directory: Union[str, Path]) -> Path:
        pass

    @abstractmethod
    def load(
        self,
        directory: Union[str, Path],
        expected_model_name: Optional[str] = None,
    ) -> Path:
        pass

    @abstractmethod
    def contains_chunk(self, chunk_id: str) -> bool:
        pass


class FaissVectorStore(BaseVectorStore):
    """Persistent vector store powered by faiss-cpu IndexFlatIP.
    
    Persistence Guarantee:
    Uses immutable, versioned snapshot directories. Never overwrites files in place.
    Publishes active snapshot atomically via active_snapshot.json pointer.
    Keeps previous valid snapshot for fallback recovery in case of interrupted writes.
    """

    def __init__(
        self,
        dimension: int = 384,
        metric: str = "cosine",
        normalize_embeddings: bool = True,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        snapshot_retention: int = 2,
    ) -> None:
        self._dimension = dimension
        self._metric = metric
        self._normalize = normalize_embeddings
        self._model_name = model_name
        self._snapshot_retention = max(snapshot_retention, 2)

        self._index = faiss.IndexFlatIP(dimension)
        self._records: List[VectorRecord] = []
        self._chunk_id_set: Set[str] = set()

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def records(self) -> List[VectorRecord]:
        return list(self._records)

    def contains_chunk(self, chunk_id: str) -> bool:
        return chunk_id in self._chunk_id_set

    def add(self, vectors: np.ndarray, records: List[VectorRecord]) -> int:
        """Add dense vectors and aligned metadata records to the index.
        
        Performs strict pre-insertion validation on vector geometry, contiguity,
        finiteness, and normalization.
        """
        if len(records) == 0:
            return 0

        vectors = validate_vectors(
            vectors,
            expected_dim=self._dimension,
            must_be_normalized=self._normalize,
        )

        if vectors.shape[0] != len(records):
            raise ValueError(
                f"Vector count ({vectors.shape[0]}) does not match record count ({len(records)})"
            )

        self._index.add(vectors)
        self._records.extend(records)
        for r in records:
            self._chunk_id_set.add(r.chunk_id)

        return len(records)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        """Perform top-K vector search using inner product / cosine similarity.
        
        Handles empty index safely and validates query vector.
        """
        if self.total_vectors == 0:
            return []

        if top_k <= 0:
            return []

        query_vector = validate_vectors(
            query_vector,
            expected_dim=self._dimension,
            must_be_normalized=self._normalize,
        )
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        k = min(top_k, self.total_vectors)
        distances, indices = self._index.search(query_vector, k)

        results: List[SearchResult] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._records):
                continue
            results.append(SearchResult(record=self._records[idx], score=float(dist)))

        return results

    def save(self, directory: Union[str, Path]) -> Path:
        """Save vector store to an immutable, versioned snapshot directory.
        
        Steps:
        1. Create a new staging snapshot directory.
        2. Write index.faiss and metadata.json.
        3. Compute SHA-256 checksums.
        4. Write manifest.json with status='committed'.
        5. Atomically rename staging directory to permanent snapshot directory.
        6. Atomically update active_snapshot.json pointer using os.replace.
        7. Prune older snapshots exceeding snapshot_retention.
        """
        base_dir = Path(directory)
        base_dir.mkdir(parents=True, exist_ok=True)
        snapshots_dir = base_dir / "snapshots"
        snapshots_dir.mkdir(exist_ok=True)

        token = uuid.uuid4().hex[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_id = f"snapshot_{timestamp}_{token}"
        staging_dir = snapshots_dir / f".staging_{snapshot_id}"
        staging_dir.mkdir(parents=True, exist_ok=True)

        faiss_file = staging_dir / "index.faiss"
        metadata_file = staging_dir / "metadata.json"
        manifest_file = staging_dir / "manifest.json"

        try:
            # 1. Write FAISS index
            faiss.write_index(self._index, str(faiss_file))

            # 2. Write metadata sidecar
            records_data = [r.to_dict() for r in self._records]
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(records_data, f, indent=2, ensure_ascii=False)

            # 3. Calculate checksums
            checksums = {
                "index.faiss": _compute_sha256(faiss_file),
                "metadata.json": _compute_sha256(metadata_file),
            }

            # 4. Write manifest with committed status
            doc_ids = sorted(list({r.document_id for r in self._records}))
            now_iso = datetime.now().isoformat()
            manifest = IndexManifest(
                model_name=self._model_name,
                dimension=self._dimension,
                similarity_metric=self._metric,
                normalize_embeddings=self._normalize,
                index_type="IndexFlatIP",
                total_vectors=self._index.ntotal,
                total_documents=len(doc_ids),
                document_ids=doc_ids,
                created_at=now_iso,
                updated_at=now_iso,
                status="committed",
                snapshot_version="1.0",
                checksums=checksums,
            )
            with open(manifest_file, "w", encoding="utf-8") as f:
                json.dump(manifest.to_dict(), f, indent=2)

            # 5. Promote staging directory to permanent snapshot
            final_snapshot_dir = snapshots_dir / snapshot_id
            os.replace(str(staging_dir), str(final_snapshot_dir))

            # 6. Atomically publish pointer via temporary file + os.replace
            pointer_file = base_dir / "active_snapshot.json"
            tmp_pointer = base_dir / f".tmp_pointer_{token}.json"
            pointer_data = {
                "active_snapshot_id": snapshot_id,
                "relative_path": f"snapshots/{snapshot_id}",
                "updated_at": now_iso,
            }
            with open(tmp_pointer, "w", encoding="utf-8") as f:
                json.dump(pointer_data, f, indent=2)
            os.replace(str(tmp_pointer), str(pointer_file))

            # 7. Prune older snapshots
            self._prune_snapshots(snapshots_dir, active_id=snapshot_id)

            return final_snapshot_dir

        except Exception as exc:
            # Clean up staging directory on failure
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
            raise VectorStoreError(f"Failed to save snapshot: {exc}") from exc

    def _prune_snapshots(self, snapshots_dir: Path, active_id: str) -> None:
        """Keep the active snapshot and previous valid snapshots up to retention limit."""
        all_snapshots = sorted(
            [d for d in snapshots_dir.iterdir() if d.is_dir() and d.name.startswith("snapshot_")],
            key=lambda d: d.name,
            reverse=True,
        )
        if len(all_snapshots) > self._snapshot_retention:
            for old_snap in all_snapshots[self._snapshot_retention:]:
                if old_snap.name != active_id:
                    shutil.rmtree(old_snap, ignore_errors=True)

    def load(
        self,
        directory: Union[str, Path],
        expected_model_name: Optional[str] = None,
    ) -> Path:
        """Load vector store from persistent storage with fallback recovery.
        
        Validates:
        - Existence and format of manifest.json, index.faiss, metadata.json.
        - Manifest status == 'committed'.
        - File checksums match manifest.
        - Model identity, dimension, similarity metric, and normalization policy.
        - Vector count matches metadata record count.
        
        If active snapshot is corrupted, attempts fallback to the previous valid snapshot.
        """
        base_dir = Path(directory)
        if not base_dir.exists():
            raise IndexCorruptionError(f"Index directory '{base_dir}' does not exist")

        target_model = expected_model_name or self._model_name

        # Determine snapshot candidates
        candidates = self._find_snapshot_candidates(base_dir)
        if not candidates:
            raise IndexCorruptionError(f"No valid snapshot candidates found in '{base_dir}'")

        last_error = None
        for candidate_dir in candidates:
            try:
                self._load_from_snapshot(candidate_dir, target_model)
                # If loaded successfully, ensure pointer points to this valid snapshot
                self._update_pointer(base_dir, candidate_dir.name)
                return candidate_dir
            except (IndexCorruptionError, IncompatibleIndexError) as err:
                last_error = err
                continue

        raise last_error or IndexCorruptionError(f"Failed to load any valid snapshot from '{base_dir}'")

    def _find_snapshot_candidates(self, base_dir: Path) -> List[Path]:
        """Identify candidate snapshot directories, prioritizing active pointer then latest valid."""
        candidates: List[Path] = []
        pointer_file = base_dir / "active_snapshot.json"

        # Check if direct snapshot directory was passed
        if (base_dir / "manifest.json").exists() and (base_dir / "index.faiss").exists():
            return [base_dir]

        # 1. Active pointer candidate
        if pointer_file.exists():
            try:
                with open(pointer_file, "r", encoding="utf-8") as f:
                    ptr = json.load(f)
                active_path = base_dir / ptr.get("relative_path", "")
                if active_path.exists():
                    candidates.append(active_path)
            except Exception:
                pass

        # 2. Add all snapshots in reverse chronological order
        snapshots_dir = base_dir / "snapshots"
        if snapshots_dir.exists():
            all_snaps = sorted(
                [d for d in snapshots_dir.iterdir() if d.is_dir() and d.name.startswith("snapshot_")],
                key=lambda d: d.name,
                reverse=True,
            )
            for s in all_snaps:
                if s not in candidates:
                    candidates.append(s)

        return candidates

    def _update_pointer(self, base_dir: Path, snapshot_id: str) -> None:
        """Atomically update active pointer if pointing to a recovered fallback."""
        pointer_file = base_dir / "active_snapshot.json"
        tmp_pointer = base_dir / f".tmp_recovery_ptr_{uuid.uuid4().hex[:8]}.json"
        data = {
            "active_snapshot_id": snapshot_id,
            "relative_path": f"snapshots/{snapshot_id}",
            "updated_at": datetime.now().isoformat(),
        }
        try:
            with open(tmp_pointer, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(str(tmp_pointer), str(pointer_file))
        except Exception:
            if tmp_pointer.exists():
                tmp_pointer.unlink()

    def _load_from_snapshot(self, snapshot_dir: Path, target_model: str) -> None:
        """Load single snapshot with full verification."""
        manifest_file = snapshot_dir / "manifest.json"
        faiss_file = snapshot_dir / "index.faiss"
        metadata_file = snapshot_dir / "metadata.json"

        if not manifest_file.exists():
            raise IndexCorruptionError(f"Missing manifest.json in '{snapshot_dir}'")
        if not faiss_file.exists():
            raise IndexCorruptionError(f"Missing index.faiss in '{snapshot_dir}'")
        if not metadata_file.exists():
            raise IndexCorruptionError(f"Missing metadata.json in '{snapshot_dir}'")

        # 1. Parse and validate manifest
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            manifest = IndexManifest.from_dict(manifest_data)
        except Exception as exc:
            raise IndexCorruptionError(f"Corrupted manifest.json in '{snapshot_dir}': {exc}") from exc

        if manifest.status != "committed":
            raise IndexCorruptionError(
                f"Snapshot in '{snapshot_dir}' is uncommitted (status='{manifest.status}')"
            )

        # 2. Verify checksums
        if manifest.checksums:
            if "index.faiss" in manifest.checksums:
                actual_hash = _compute_sha256(faiss_file)
                if actual_hash != manifest.checksums["index.faiss"]:
                    raise IndexCorruptionError(f"Checksum mismatch for index.faiss in '{snapshot_dir}'")
            if "metadata.json" in manifest.checksums:
                actual_hash = _compute_sha256(metadata_file)
                if actual_hash != manifest.checksums["metadata.json"]:
                    raise IndexCorruptionError(f"Checksum mismatch for metadata.json in '{snapshot_dir}'")

        # 3. Verify compatibility gates
        if manifest.dimension != self._dimension:
            raise IncompatibleIndexError(
                f"Dimension mismatch: expected {self._dimension}, index has {manifest.dimension}"
            )
        if manifest.model_name != target_model:
            raise IncompatibleIndexError(
                f"Model mismatch: expected '{target_model}', index was built with '{manifest.model_name}'"
            )
        if manifest.similarity_metric != self._metric:
            raise IncompatibleIndexError(
                f"Similarity metric mismatch: expected '{self._metric}', index has '{manifest.similarity_metric}'"
            )
        if manifest.normalize_embeddings != self._normalize:
            raise IncompatibleIndexError(
                f"Normalization mismatch: expected {self._normalize}, index has {manifest.normalize_embeddings}"
            )

        # 4. Load FAISS index and metadata
        try:
            loaded_index = faiss.read_index(str(faiss_file))
        except Exception as exc:
            raise IndexCorruptionError(f"Failed to read FAISS index: {exc}") from exc

        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                records_raw = json.load(f)
            loaded_records = [VectorRecord.from_dict(r) for r in records_raw]
        except Exception as exc:
            raise IndexCorruptionError(f"Failed to read metadata sidecar: {exc}") from exc

        # 5. Check count alignment
        if loaded_index.ntotal != len(loaded_records):
            raise IndexCorruptionError(
                f"Vector count mismatch: FAISS has {loaded_index.ntotal} vectors, "
                f"metadata has {len(loaded_records)} records"
            )

        # Commit state to instance
        self._index = loaded_index
        self._records = loaded_records
        self._chunk_id_set = {r.chunk_id for r in loaded_records}
