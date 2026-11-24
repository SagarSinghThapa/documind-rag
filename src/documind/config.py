"""Configuration constants and default parameters for DocuMind."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class IngestionConfig:
    """Configuration parameters for document ingestion and chunking.
    
    Measurement unit for chunking is characters. At typical English densities,
    1,000 characters corresponds to approximately 200-250 tokens.
    """
    chunk_size: int = 1000
    chunk_overlap: int = 150
    chunking_strategy: str = "recursive"  # 'recursive' or 'fixed'
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])
    max_file_size_bytes: int = 50 * 1024 * 1024  # 50 MB
    supported_extensions: List[str] = field(default_factory=lambda: [".pdf"])
    extract_empty_pages: bool = False  # Whether to emit empty PageDocument objects


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration parameters for local embedding generation.
    
    Offline-first by default: `local_files_only` prevents silent network requests
    during runtime execution. Explicit provisioning via scripts/download_model.py
    or `allow_download=True` is required for initial model retrieval.
    """
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32
    normalize_embeddings: bool = True
    device: str = "cpu"
    cache_folder: Optional[str] = None
    local_files_only: bool = True       # Strict offline default: runtime never downloads
    allow_download: bool = False        # Explicit permission flag required for model download


@dataclass(frozen=True)
class VectorStoreConfig:
    """Configuration parameters for FAISS vector storage and crash-consistent persistence."""
    index_type: str = "IndexFlatIP"     # Exact inner product for L2-normalized vectors (cosine similarity)
    metric: str = "cosine"
    index_dir: str = "data/processed/vector_index"
    snapshot_retention: int = 2         # Minimum number of valid snapshots preserved for recovery


@dataclass(frozen=True)
class RetrievalConfig:
    """Configuration parameters for semantic retrieval, threshold filtering, and context assembly."""
    top_k: int = 5
    similarity_threshold: Optional[float] = None  # None = unthresholded baseline; float (e.g. 0.35) = candidate filter
    max_context_chars: int = 4000               # ~1000 tokens
    deduplicate_overlap: bool = True
    min_overlap_chars: int = 20                 # Minimum verified overlap length to trigger deduplication
    document_id_filter: Optional[str] = None   # Optional single-document constraint


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration parameters for local grounded LLM generation.
    
    Offline-first by default: `local_files_only=True` prevents silent network requests
    during runtime execution. Explicit provisioning via scripts/download_llm.py or
    `allow_download=True` is required for initial model weight retrieval.
    """
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"  # Initial lightweight CPU baseline
    max_new_tokens: int = 256
    temperature: float = 0.0                        # Deterministic generation
    top_p: float = 1.0
    device: str = "cpu"
    num_threads: Optional[int] = 8                  # CPU execution thread count
    local_files_only: bool = True                   # Strict offline runtime default
    allow_download: bool = False                    # Explicit permission flag required for download
    cache_folder: Optional[str] = "models/llm"
    max_context_chars: int = 1800                   # Context window right-sized for fast CPU prefill
    include_exemplar: bool = True                   # Few-shot exemplar for reliable [S1] citation
    abstention_phrase: str = "I cannot answer this question based on the provided document context."
