"""High-level DocumentIndex orchestrator integrating IngestionPipeline, Embedder, and VectorStore."""

from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from documind.config import VectorStoreConfig
from documind.ingestion.schemas import TextChunk
from documind.storage.embedder import BaseEmbedder, LocalEmbedder
from documind.storage.schemas import IndexingResult, SearchResult, VectorRecord
from documind.storage.vector_store import (
    BaseVectorStore,
    FaissVectorStore,
    IncompatibleIndexError,
    IndexCorruptionError,
)


class DocumentIndex:
    """Orchestrates embedding generation, two-level deduplication, vector indexing, and search.
    
    Acts as the primary entry point for vector search in DocuMind, bridging
    Phase 1 TextChunk objects with persistent local retrieval.
    """

    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        vector_store: Optional[BaseVectorStore] = None,
        config: Optional[VectorStoreConfig] = None,
    ) -> None:
        self._config = config or VectorStoreConfig()
        self._embedder = embedder or LocalEmbedder()
        self._vector_store = vector_store or FaissVectorStore(
            dimension=self._embedder.dimension,
            metric=self._config.metric,
            normalize_embeddings=self._embedder.normalize_embeddings,
            model_name=self._embedder.model_name,
            snapshot_retention=self._config.snapshot_retention,
        )

    @property
    def embedder(self) -> BaseEmbedder:
        return self._embedder

    @property
    def vector_store(self) -> BaseVectorStore:
        return self._vector_store

    @property
    def total_vectors(self) -> int:
        return self._vector_store.total_vectors

    def add_chunks(self, chunks: List[TextChunk]) -> IndexingResult:
        """Index a list of chunks with two-level deduplication and empty chunk filtering.
        
        Safeguards:
        1. Filters empty or whitespace-only chunks (never embeds or indexes zero/NaN vectors).
        2. Filters intra-batch duplicate chunk IDs.
        3. Filters chunk IDs that already exist in the vector store.
        """
        seen_in_batch: Set[str] = set()
        valid_chunks: List[TextChunk] = []
        empty_count = 0
        intra_dup_count = 0
        exist_dup_count = 0

        for chunk in chunks:
            # 1. Filter empty or whitespace-only chunks
            if not chunk.text or not chunk.text.strip():
                empty_count += 1
                continue

            # 2. Filter intra-batch duplicates
            if chunk.chunk_id in seen_in_batch:
                intra_dup_count += 1
                continue
            seen_in_batch.add(chunk.chunk_id)

            # 3. Filter existing index duplicates
            if self._vector_store.contains_chunk(chunk.chunk_id):
                exist_dup_count += 1
                continue

            valid_chunks.append(chunk)

        if not valid_chunks:
            return IndexingResult(
                total_received=len(chunks),
                newly_indexed=0,
                intra_batch_duplicates_skipped=intra_dup_count,
                existing_duplicates_skipped=exist_dup_count,
                empty_chunks_skipped=empty_count,
            )

        # Vectorize valid unique chunks
        texts = [c.text for c in valid_chunks]
        vectors = self._embedder.embed_texts(texts)

        # Build VectorRecord objects with embeddings
        records: List[VectorRecord] = []
        for i, chunk in enumerate(valid_chunks):
            record = VectorRecord.from_text_chunk(
                chunk=chunk,
                embedding=vectors[i].tolist(),
            )
            records.append(record)

        # Add to vector store
        self._vector_store.add(vectors, records)

        return IndexingResult(
            total_received=len(chunks),
            newly_indexed=len(valid_chunks),
            intra_batch_duplicates_skipped=intra_dup_count,
            existing_duplicates_skipped=exist_dup_count,
            empty_chunks_skipped=empty_count,
        )

    def add_documents(
        self, documents_chunks: Dict[str, List[TextChunk]]
    ) -> Dict[str, IndexingResult]:
        """Ingest multiple documents sequentially, returning per-document telemetry."""
        results: Dict[str, IndexingResult] = {}
        for doc_id, chunks in documents_chunks.items():
            results[doc_id] = self.add_chunks(chunks)
        return results

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Perform semantic search against indexed chunks.
        
        Safeguards:
        - Rejects empty or whitespace-only queries.
        - Safely returns empty list if index contains no vectors.
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty or whitespace-only")

        if self._vector_store.total_vectors == 0:
            return []

        query_vec = self._embedder.embed_text(query)
        return self._vector_store.search(query_vec, top_k=top_k)

    def save(self, directory: Optional[Union[str, Path]] = None) -> Path:
        """Save vector store to disk using crash-consistent snapshotting."""
        target_dir = Path(directory or self._config.index_dir)
        return self._vector_store.save(target_dir)

    def load(self, directory: Optional[Union[str, Path]] = None) -> Path:
        """Load vector store from disk with full compatibility and recovery validation."""
        target_dir = Path(directory or self._config.index_dir)
        return self._vector_store.load(
            target_dir, expected_model_name=self._embedder.model_name
        )
