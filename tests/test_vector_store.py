"""Comprehensive test suite for DocuMind persistent FAISS vector store, deduplication, and snapshots."""

import json
import shutil
import pytest
import numpy as np
from pathlib import Path

from documind.config import EmbeddingConfig, VectorStoreConfig
from documind.ingestion.pipeline import IngestionPipeline
from documind.ingestion.schemas import TextChunk
from documind.storage.embedder import LocalEmbedder
from documind.storage.index import DocumentIndex
from documind.storage.schemas import VectorRecord
from documind.storage.vector_store import (
    FaissVectorStore,
    IncompatibleIndexError,
    IndexCorruptionError,
)


def create_sample_chunk(
    chunk_id: str,
    doc_id: str = "doc_hash_123",
    text: str = "Sample chunk text for testing.",
    chunk_index: int = 0,
) -> TextChunk:
    """Helper to create a valid TextChunk with all 12 metadata attributes."""
    return TextChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        document_name="test_sample.pdf",
        source_path="tests/fixtures/test_sample.pdf",
        chunk_index=chunk_index,
        text=text,
        page_numbers=[1],
        start_page=1,
        end_page=1,
        chunking_strategy="recursive",
        chunk_size=1000,
        chunk_overlap=150,
        char_start=0,
        char_end=len(text),
        metadata={"created_by": "test_suite", "ordinal": chunk_index},
    )


def create_normalized_vectors(num_vectors: int, dim: int = 384) -> np.ndarray:
    """Helper to generate random L2-normalized float32 vectors."""
    np.random.seed(42)
    vecs = np.random.randn(num_vectors, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def test_vector_store_initialization():
    """Verify initial vector store state, empty counts, and dimension."""
    vs = FaissVectorStore(dimension=384, metric="cosine", normalize_embeddings=True)
    assert vs.total_vectors == 0
    assert vs.dimension == 384
    assert len(vs.records) == 0


def test_vector_store_add_and_search():
    """Verify adding vectors and retrieving top-K results with cosine scores."""
    vs = FaissVectorStore(dimension=384)
    vectors = create_normalized_vectors(5, 384)
    chunks = [create_sample_chunk(f"c_{i}", text=f"Chunk text {i}", chunk_index=i) for i in range(5)]
    records = [VectorRecord.from_text_chunk(c, embedding=vectors[i].tolist()) for i, c in enumerate(chunks)]

    count = vs.add(vectors, records)
    assert count == 5
    assert vs.total_vectors == 5

    # Query with exact first vector
    results = vs.search(vectors[0], top_k=3)
    assert len(results) == 3
    assert results[0].record.chunk_id == "c_0"
    assert results[0].score == pytest.approx(1.0, abs=1e-3)


def test_vector_store_vector_validation():
    """Verify that NaN, Inf, or bad dimensions are rejected with ValueError."""
    vs = FaissVectorStore(dimension=384)
    chunk = create_sample_chunk("c_bad")
    record = VectorRecord.from_text_chunk(chunk)

    # Wrong dimension
    with pytest.raises(ValueError, match="dimension mismatch"):
        bad_dim = np.ones((1, 100), dtype=np.float32)
        vs.add(bad_dim, [record])

    # NaN vector
    nan_vec = np.zeros((1, 384), dtype=np.float32)
    nan_vec[0, 5] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        vs.add(nan_vec, [record])


def test_vector_store_metadata_integrity_all_12_fields(tmp_path):
    """Exhaustively verify all 12 metadata attributes survive persistence roundtrip."""
    vs = FaissVectorStore(dimension=384)
    vector = create_normalized_vectors(1, 384)
    original_chunk = TextChunk(
        chunk_id="chk_full_meta_001",
        document_id="sha256_full_doc_abc",
        document_name="architecture_deep_dive.pdf",
        source_path="docs/architecture_deep_dive.pdf",
        chunk_index=42,
        text="A detailed explanation of retrieval-augmented generation architectures.",
        page_numbers=[3, 4],
        start_page=3,
        end_page=4,
        chunking_strategy="recursive",
        chunk_size=1000,
        chunk_overlap=150,
        char_start=1520,
        char_end=2480,
        metadata={"domain": "ai", "reviewed": True, "token_count": 210},
    )
    record = VectorRecord.from_text_chunk(original_chunk, embedding=vector[0].tolist())
    vs.add(vector, [record])

    # Save and reload
    save_dir = vs.save(tmp_path / "metadata_test_index")
    vs_reloaded = FaissVectorStore(dimension=384)
    vs_reloaded.load(tmp_path / "metadata_test_index")

    reloaded_record = vs_reloaded.records[0]
    assert reloaded_record.chunk_id == original_chunk.chunk_id
    assert reloaded_record.document_id == original_chunk.document_id
    assert reloaded_record.document_name == original_chunk.document_name
    assert reloaded_record.source_path == original_chunk.source_path
    assert reloaded_record.chunk_index == original_chunk.chunk_index
    assert reloaded_record.page_numbers == original_chunk.page_numbers
    assert reloaded_record.start_page == original_chunk.start_page
    assert reloaded_record.end_page == original_chunk.end_page
    assert reloaded_record.char_start == original_chunk.char_start
    assert reloaded_record.char_end == original_chunk.char_end
    assert reloaded_record.chunking_strategy == original_chunk.chunking_strategy
    assert reloaded_record.chunk_size == original_chunk.chunk_size
    assert reloaded_record.chunk_overlap == original_chunk.chunk_overlap
    assert reloaded_record.text == original_chunk.text
    assert reloaded_record.metadata == original_chunk.metadata


def test_deduplication_intra_and_inter_batch(tmp_path):
    """Verify two-level deduplication (intra-batch and inter-batch) with IndexingResult telemetry."""
    embedder = LocalEmbedder(EmbeddingConfig(local_files_only=True, allow_download=False))
    doc_index = DocumentIndex(embedder=embedder)

    c1 = create_sample_chunk("dup_c1", text="Chunk one text for deduplication test.")
    c2 = create_sample_chunk("dup_c2", text="Chunk two text for deduplication test.")
    c3 = create_sample_chunk("dup_c3", text="Chunk three text for deduplication test.")

    # 1. Batch with intra-batch duplicate: [c1, c2, c1]
    batch_1 = [c1, c2, c1]
    res_1 = doc_index.add_chunks(batch_1)
    assert res_1.total_received == 3
    assert res_1.newly_indexed == 2
    assert res_1.intra_batch_duplicates_skipped == 1
    assert res_1.existing_duplicates_skipped == 0
    assert doc_index.total_vectors == 2

    # 2. Subsequent batch with existing duplicates: [c1, c2, c3]
    batch_2 = [c1, c2, c3]
    res_2 = doc_index.add_chunks(batch_2)
    assert res_2.total_received == 3
    assert res_2.newly_indexed == 1
    assert res_2.intra_batch_duplicates_skipped == 0
    assert res_2.existing_duplicates_skipped == 2
    assert doc_index.total_vectors == 3


def test_empty_chunks_skipped():
    """Verify empty or whitespace-only chunks are skipped without error or indexing zero vectors."""
    embedder = LocalEmbedder(EmbeddingConfig(local_files_only=True, allow_download=False))
    doc_index = DocumentIndex(embedder=embedder)

    empty_c1 = create_sample_chunk("empty_1", text="")
    empty_c2 = create_sample_chunk("empty_2", text="   \n\t   ")
    valid_c = create_sample_chunk("valid_1", text="Valid text content.")

    res = doc_index.add_chunks([empty_c1, empty_c2, valid_c])
    assert res.total_received == 3
    assert res.newly_indexed == 1
    assert res.empty_chunks_skipped == 2
    assert doc_index.total_vectors == 1


def test_empty_index_search():
    """Verify searching an empty index returns empty list [] without throwing exceptions."""
    vs = FaissVectorStore(dimension=384)
    dummy_query = np.ones(384, dtype=np.float32) / np.sqrt(384)
    results = vs.search(dummy_query, top_k=5)
    assert results == []


def test_empty_query_raises_value_error():
    """Verify empty or whitespace-only search queries raise ValueError."""
    embedder = LocalEmbedder(EmbeddingConfig(local_files_only=True, allow_download=False))
    doc_index = DocumentIndex(embedder=embedder)
    with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
        doc_index.search("")
    with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
        doc_index.search("   ")


def test_persistence_immutable_snapshots_and_pointer(tmp_path):
    """Verify crash-consistent versioned snapshot directories and atomic pointer publication."""
    index_dir = tmp_path / "snapshot_repo"
    vs = FaissVectorStore(dimension=384)
    vectors = create_normalized_vectors(2, 384)
    chunks = [create_sample_chunk(f"c_{i}", text=f"Chunk {i}") for i in range(2)]
    records = [VectorRecord.from_text_chunk(c, embedding=vectors[i].tolist()) for i, c in enumerate(chunks)]
    vs.add(vectors, records)

    # First save
    snap_1_dir = vs.save(index_dir)
    assert snap_1_dir.exists()
    assert (snap_1_dir / "index.faiss").exists()
    assert (snap_1_dir / "metadata.json").exists()
    assert (snap_1_dir / "manifest.json").exists()

    pointer_file = index_dir / "active_snapshot.json"
    assert pointer_file.exists()
    with open(pointer_file, "r", encoding="utf-8") as f:
        ptr_1 = json.load(f)
    assert ptr_1["active_snapshot_id"] == snap_1_dir.name

    # Second save (creates new snapshot without overwriting snap_1 in place)
    new_vector = create_normalized_vectors(1, 384)
    new_chunk = create_sample_chunk("c_new", text="New chunk")
    vs.add(new_vector, [VectorRecord.from_text_chunk(new_chunk, embedding=new_vector[0].tolist())])

    snap_2_dir = vs.save(index_dir)
    assert snap_2_dir.exists()
    assert snap_2_dir != snap_1_dir
    assert snap_1_dir.exists()  # Previous snapshot remains available!

    with open(pointer_file, "r", encoding="utf-8") as f:
        ptr_2 = json.load(f)
    assert ptr_2["active_snapshot_id"] == snap_2_dir.name


def test_persistence_rejects_uncommitted_snapshot(tmp_path):
    """Verify loading rejects uncommitted snapshots where status != 'committed'."""
    index_dir = tmp_path / "uncommitted_repo"
    vs = FaissVectorStore(dimension=384)
    vs.save(index_dir)

    # Read manifest and corrupt status to 'staging'
    pointer_file = index_dir / "active_snapshot.json"
    with open(pointer_file, "r") as f:
        ptr = json.load(f)
    manifest_file = index_dir / ptr["relative_path"] / "manifest.json"
    with open(manifest_file, "r") as f:
        m = json.load(f)
    m["status"] = "staging"
    with open(manifest_file, "w") as f:
        json.dump(m, f)

    vs_reload = FaissVectorStore(dimension=384)
    with pytest.raises(IndexCorruptionError, match="uncommitted"):
        vs_reload.load(index_dir)


def test_load_missing_or_corrupt_files(tmp_path):
    """Verify missing index.faiss or corrupt metadata.json raises IndexCorruptionError."""
    index_dir = tmp_path / "corrupt_repo"
    vs = FaissVectorStore(dimension=384)
    vs.save(index_dir)

    pointer_file = index_dir / "active_snapshot.json"
    with open(pointer_file, "r") as f:
        ptr = json.load(f)
    snap_dir = index_dir / ptr["relative_path"]

    # Delete index.faiss
    (snap_dir / "index.faiss").unlink()

    vs_reload = FaissVectorStore(dimension=384)
    with pytest.raises(IndexCorruptionError, match="Missing index.faiss"):
        vs_reload.load(index_dir)


def test_load_count_mismatch_error(tmp_path):
    """Verify vector count mismatch between FAISS and metadata sidecar raises IndexCorruptionError."""
    index_dir = tmp_path / "mismatch_repo"
    vs = FaissVectorStore(dimension=384)
    vec = create_normalized_vectors(2, 384)
    c1 = create_sample_chunk("c1")
    c2 = create_sample_chunk("c2")
    vs.add(vec, [VectorRecord.from_text_chunk(c1), VectorRecord.from_text_chunk(c2)])
    vs.save(index_dir)

    # Modify metadata.json to drop one record (updating checksum in manifest so count mismatch is triggered)
    pointer_file = index_dir / "active_snapshot.json"
    with open(pointer_file, "r") as f:
        ptr = json.load(f)
    meta_file = index_dir / ptr["relative_path"] / "metadata.json"
    with open(meta_file, "r") as f:
        records = json.load(f)
    records.pop()  # Drop one record
    with open(meta_file, "w") as f:
        json.dump(records, f)

    vs_reload = FaissVectorStore(dimension=384)
    with pytest.raises(IndexCorruptionError, match="Vector count mismatch|Checksum mismatch"):
        vs_reload.load(index_dir)


def test_load_incompatible_model_or_dimension(tmp_path):
    """Verify loading index created with different model or dimension raises IncompatibleIndexError."""
    index_dir = tmp_path / "compat_repo"
    vs = FaissVectorStore(dimension=384, model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs.save(index_dir)

    # 1. Dimension mismatch
    vs_dim = FaissVectorStore(dimension=768)
    with pytest.raises(IncompatibleIndexError, match="Dimension mismatch"):
        vs_dim.load(index_dir)

    # 2. Model name mismatch
    vs_model = FaissVectorStore(dimension=384, model_name="BAAI/bge-small-en-v1.5")
    with pytest.raises(IncompatibleIndexError, match="Model mismatch"):
        vs_model.load(index_dir, expected_model_name="BAAI/bge-small-en-v1.5")


def test_fallback_recovery_when_active_corrupted(tmp_path):
    """Verify that when active snapshot is corrupt, load automatically falls back to previous valid snapshot."""
    index_dir = tmp_path / "fallback_repo"
    vs = FaissVectorStore(dimension=384)

    # Save Snapshot 1 (valid)
    v1 = create_normalized_vectors(1, 384)
    c1 = create_sample_chunk("chunk_1", text="First snapshot content")
    vs.add(v1, [VectorRecord.from_text_chunk(c1)])
    snap_1 = vs.save(index_dir)

    # Save Snapshot 2 (valid)
    v2 = create_normalized_vectors(1, 384)
    c2 = create_sample_chunk("chunk_2", text="Second snapshot content")
    vs.add(v2, [VectorRecord.from_text_chunk(c2)])
    snap_2 = vs.save(index_dir)

    # Corrupt Snapshot 2's FAISS file
    with open(snap_2 / "index.faiss", "wb") as f:
        f.write(b"corrupted binary data")

    # Load should detect snap_2 corruption and recover snap_1!
    vs_recovered = FaissVectorStore(dimension=384)
    recovered_path = vs_recovered.load(index_dir)

    assert recovered_path == snap_1
    assert vs_recovered.total_vectors == 1
    assert vs_recovered.records[0].chunk_id == "chunk_1"


def test_document_index_end_to_end_real_pdf(tmp_path):
    """End-to-end integration: Ingest real PDF from Phase 1, index, save, reload, and semantic search."""
    pdf_path = Path("research/technical_references/BASWE-15-RAG-Concepts-Guide.pdf")
    if not pdf_path.exists():
        pytest.skip("Reference PDF not present in environment.")

    # 1. Phase 1 Ingestion
    pipeline = IngestionPipeline()
    chunks, doc_report = pipeline.ingest_file(pdf_path)
    assert doc_report.success
    assert len(chunks) > 0

    # 2. DocumentIndex Ingestion
    embedder = LocalEmbedder(EmbeddingConfig(local_files_only=True, allow_download=False))
    doc_index = DocumentIndex(embedder=embedder)
    res = doc_index.add_chunks(chunks)
    assert res.newly_indexed == len(chunks)

    # 3. Persistent Snapshot Save
    index_dir = tmp_path / "real_pdf_index"
    saved_path = doc_index.save(index_dir)
    assert saved_path.exists()

    # 4. Reload into fresh instance
    doc_index_reloaded = DocumentIndex(embedder=embedder)
    doc_index_reloaded.load(index_dir)
    assert doc_index_reloaded.total_vectors == len(chunks)

    # 5. Semantic Query
    results = doc_index_reloaded.search("What is chunking strategy?", top_k=3)
    assert len(results) > 0
    top_hit = results[0]
    assert top_hit.score > 0.3
    assert top_hit.record.document_name == "BASWE-15-RAG-Concepts-Guide.pdf"
    assert top_hit.record.start_page >= 1
    assert len(top_hit.record.text) > 0
