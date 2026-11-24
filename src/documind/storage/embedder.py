"""Local embedding engine supporting CPU inference, offline safety, and vector validation."""

from abc import ABC, abstractmethod
from typing import List, Optional, Union
import numpy as np

from documind.config import EmbeddingConfig


class EmbedderError(Exception):
    """Base exception for embedder failures."""
    pass


class ModelNotFoundError(EmbedderError):
    """Raised when model weights are missing locally and offline runtime is enforced."""
    pass


def validate_vectors(
    vectors: np.ndarray,
    expected_dim: int,
    must_be_normalized: bool = True,
    norm_tolerance: float = 1e-3,
) -> np.ndarray:
    """Validate vector shape, dtype, memory contiguity, finiteness, and normalization.
    
    Protects downstream FAISS C++ indexing and similarity search from invalid,
    unnormalized, non-contiguous, or NaN/Inf vector data.
    """
    if not isinstance(vectors, np.ndarray):
        vectors = np.asarray(vectors, dtype=np.float32)
    
    # Enforce float32 and C-contiguity required by FAISS
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    vectors = np.ascontiguousarray(vectors)

    # Validate shape
    if vectors.ndim == 1:
        if vectors.shape[0] != expected_dim:
            raise ValueError(f"Vector dimension mismatch: expected {expected_dim}, got {vectors.shape[0]}")
    elif vectors.ndim == 2:
        if vectors.shape[1] != expected_dim:
            raise ValueError(f"Vector dimension mismatch: expected {expected_dim}, got {vectors.shape[1]}")
    else:
        raise ValueError(f"Invalid vector dimensionality: expected 1D or 2D array, got {vectors.ndim}D")

    # Assert finiteness (no NaN, +Inf, -Inf)
    if not np.all(np.isfinite(vectors)):
        raise ValueError("Vectors contain NaN or Inf values")

    # Validate normalization if required
    if must_be_normalized:
        if vectors.ndim == 1:
            norm = float(np.linalg.norm(vectors))
            if abs(norm - 1.0) > norm_tolerance:
                raise ValueError(f"Vector is not L2-normalized: norm={norm:.6f} (expected ~1.0)")
        elif vectors.ndim == 2 and vectors.shape[0] > 0:
            norms = np.linalg.norm(vectors, axis=1)
            deviations = np.abs(norms - 1.0)
            if np.any(deviations > norm_tolerance):
                max_dev = float(np.max(deviations))
                raise ValueError(f"Vectors are not L2-normalized: max deviation from 1.0 is {max_dev:.6f}")

    return vectors


class BaseEmbedder(ABC):
    """Abstract base class for all DocuMind embedding engines."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        pass

    @property
    @abstractmethod
    def normalize_embeddings(self) -> bool:
        """Return whether embeddings are L2-normalized."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Generate a 1D normalized embedding vector for a single text."""
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Generate a 2D normalized embedding matrix for a list of texts."""
        pass


class LocalEmbedder(BaseEmbedder):
    """Local, CPU-compatible embedding engine powered by sentence-transformers.
    
    Enforces offline execution by default: if weights are missing locally and
    `allow_download` is False, raises `ModelNotFoundError` with explicit instructions.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None) -> None:
        self._config = config or EmbeddingConfig()
        self._dimension = self._config.dimension
        self._model_name = self._config.model_name
        self._normalize = self._config.normalize_embeddings
        self._batch_size = self._config.batch_size
        self._device = "cpu"  # Strictly enforce CPU execution
        self._model = self._load_model()

    def _load_model(self):
        """Load model with offline protection."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbedderError(
                "sentence-transformers is not installed. Install requirements first."
            ) from exc

        try:
            # If local_files_only is True and allow_download is False, enforce offline loading
            local_only = self._config.local_files_only and not self._config.allow_download
            model = SentenceTransformer(
                self._model_name,
                device=self._device,
                cache_folder=self._config.cache_folder,
                local_files_only=local_only,
            )
            return model
        except Exception as exc:
            # Detect missing model when offline
            err_msg = str(exc).lower()
            if "offline" in err_msg or "not found" in err_msg or "local_files_only" in err_msg or "entry not found" in err_msg:
                raise ModelNotFoundError(
                    f"Model '{self._model_name}' was not found in the local cache, and offline mode is enforced. "
                    f"Please run the explicit provisioning script first: 'python scripts/download_model.py --model-name {self._model_name}'"
                ) from exc
            raise EmbedderError(f"Failed to initialize embedding model '{self._model_name}': {exc}") from exc

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def normalize_embeddings(self) -> bool:
        return self._normalize

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string on CPU with validation.
        
        Rejects empty or whitespace-only inputs to prevent zero-vector or NaN pollution.
        """
        if not text or not text.strip():
            raise ValueError("Text to embed cannot be empty or whitespace-only")

        vector = self._model.encode(
            text,
            device=self._device,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        vector = np.asarray(vector, dtype=np.float32)
        return validate_vectors(vector, self._dimension, self._normalize)

    def embed_texts(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Embed multiple text strings in batches on CPU with validation.
        
        Rejects any batch containing empty or whitespace-only strings.
        """
        if len(texts) == 0:
            return np.empty((0, self._dimension), dtype=np.float32)

        for i, t in enumerate(texts):
            if not t or not t.strip():
                raise ValueError(f"Text at index {i} is empty or whitespace-only. Filter empty chunks before embedding.")

        batch_sz = batch_size or self._batch_size
        vectors = self._model.encode(
            texts,
            batch_size=batch_sz,
            device=self._device,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        return validate_vectors(vectors, self._dimension, self._normalize)
