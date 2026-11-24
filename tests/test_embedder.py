"""Unit tests for DocuMind LocalEmbedder, vector validation, and offline safeguards."""

import pytest
import numpy as np

from documind.config import EmbeddingConfig
from documind.storage.embedder import (
    LocalEmbedder,
    ModelNotFoundError,
    validate_vectors,
)


@pytest.fixture(scope="module")
def embedder():
    """Shared LocalEmbedder instance for tests.
    
    Uses local_files_only=True to ensure strictly offline execution.
    Requires model to be provisioned prior to test run.
    """
    config = EmbeddingConfig(local_files_only=True, allow_download=False)
    return LocalEmbedder(config)


def test_embedder_dimension_and_dtype(embedder):
    """Test output vector has exact expected dimension, float32 dtype, and C-contiguity."""
    text = "Retrieval-Augmented Generation processes document chunks into dense representations."
    vector = embedder.embed_text(text)

    assert isinstance(vector, np.ndarray)
    assert vector.shape == (384,)
    assert vector.dtype == np.float32
    assert vector.flags.c_contiguous


def test_embedder_l2_normalization(embedder):
    """Test output vector is L2-normalized within mathematical tolerance."""
    texts = [
        "Vector search relies on normalized inner products for cosine similarity.",
        "Short sentence.",
        "A somewhat longer paragraph detailing how recursive chunking splits text on sentence boundaries.",
    ]
    for text in texts:
        vector = embedder.embed_text(text)
        norm = float(np.linalg.norm(vector))
        assert norm == pytest.approx(1.0, abs=1e-3)


def test_embedder_deterministic_output(embedder):
    """Test identical text produces deterministic embeddings using numerical tolerance checks (Safeguard 7)."""
    text = "Deterministic testing guarantees reproducible vector indexes."
    v1 = embedder.embed_text(text)
    v2 = embedder.embed_text(text)

    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-6)


def test_embedder_batch_equality(embedder):
    """Test batch vectorization matches individual encoding within numerical tolerance."""
    texts = [
        "First document chunk regarding natural language processing.",
        "Second chunk concerning dense vector representations.",
        "Third chunk describing persistent vector databases.",
    ]
    batch_vectors = embedder.embed_texts(texts)
    indiv_vectors = np.vstack([embedder.embed_text(t) for t in texts])

    assert batch_vectors.shape == (3, 384)
    assert np.allclose(batch_vectors, indiv_vectors, rtol=1e-5, atol=1e-6)


def test_embedder_semantic_ordering(embedder):
    """Test semantically related sentences have strictly higher cosine similarity than unrelated ones."""
    query = embedder.embed_text("deep neural network architecture and backpropagation")
    related = embedder.embed_text("convolutional models and gradient descent optimization")
    unrelated = embedder.embed_text("authentic Italian recipe for handmade carbonara pasta")

    sim_related = float(np.dot(query, related))
    sim_unrelated = float(np.dot(query, unrelated))

    assert sim_related > sim_unrelated
    assert sim_related > 0.4


def test_embedder_empty_input_rejection(embedder):
    """Test empty string and whitespace-only inputs are rejected with ValueError (Safeguards 5, 6)."""
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        embedder.embed_text("")

    with pytest.raises(ValueError, match="empty or whitespace-only"):
        embedder.embed_text("   \n\t  ")


def test_embedder_empty_batch_handling(embedder):
    """Test passing an empty list returns empty 2D array of shape (0, 384) with float32 dtype."""
    vectors = embedder.embed_texts([])
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (0, 384)
    assert vectors.dtype == np.float32


def test_embedder_batch_rejects_empty_chunk(embedder):
    """Test passing a batch containing an empty chunk raises ValueError with index info."""
    texts = ["Valid text span.", "", "Another valid span."]
    with pytest.raises(ValueError, match="index 1 is empty or whitespace-only"):
        embedder.embed_texts(texts)


def test_embedder_vector_finiteness(embedder):
    """Test generated vectors contain no NaN or Infinite values."""
    vector = embedder.embed_text("Testing vector finiteness under CPU inference.")
    assert np.all(np.isfinite(vector))


def test_embedder_offline_mode_protection():
    """Test that requesting a non-cached model in offline mode raises ModelNotFoundError (Safeguard 3)."""
    non_existent_config = EmbeddingConfig(
        model_name="non-existent-user/totally-fake-model-xyz",
        local_files_only=True,
        allow_download=False,
    )
    with pytest.raises(ModelNotFoundError) as exc_info:
        LocalEmbedder(non_existent_config)

    assert "scripts/download_model.py" in str(exc_info.value)


def test_validate_vectors_function():
    """Directly test the vector validation helper against edge cases."""
    dim = 384
    valid_vec = np.random.randn(dim).astype(np.float32)
    valid_vec /= np.linalg.norm(valid_vec)

    # Valid 1D
    out = validate_vectors(valid_vec, dim, must_be_normalized=True)
    assert out.shape == (dim,)

    # Dimension mismatch
    with pytest.raises(ValueError, match="dimension mismatch"):
        validate_vectors(valid_vec[:100], dim)

    # NaN detection
    nan_vec = valid_vec.copy()
    nan_vec[10] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_vectors(nan_vec, dim, must_be_normalized=False)

    # Inf detection
    inf_vec = valid_vec.copy()
    inf_vec[10] = np.inf
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_vectors(inf_vec, dim, must_be_normalized=False)

    # Unnormalized rejection
    unnorm_vec = valid_vec * 5.0
    with pytest.raises(ValueError, match="not L2-normalized"):
        validate_vectors(unnorm_vec, dim, must_be_normalized=True)
