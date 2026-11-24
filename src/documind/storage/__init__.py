"""DocuMind Storage Subsystem: Schemas, Embeddings, and Persistent Vector Indexing."""

from documind.storage.schemas import (
    VectorRecord,
    IndexManifest,
    SearchResult,
    IndexingResult,
)
from documind.storage.embedder import (
    BaseEmbedder,
    LocalEmbedder,
    EmbedderError,
    ModelNotFoundError,
    validate_vectors,
)
from documind.storage.vector_store import (
    BaseVectorStore,
    FaissVectorStore,
    VectorStoreError,
    IncompatibleIndexError,
    IndexCorruptionError,
)
from documind.storage.index import DocumentIndex

__all__ = [
    "VectorRecord",
    "IndexManifest",
    "SearchResult",
    "IndexingResult",
    "BaseEmbedder",
    "LocalEmbedder",
    "EmbedderError",
    "ModelNotFoundError",
    "validate_vectors",
    "BaseVectorStore",
    "FaissVectorStore",
    "VectorStoreError",
    "IncompatibleIndexError",
    "IndexCorruptionError",
    "DocumentIndex",
]
