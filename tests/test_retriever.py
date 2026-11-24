"""Automated unit and integration tests for SemanticRetriever, query validation, and filtering."""

import hashlib
import re
from pathlib import Path
from typing import List, Optional
import numpy as np
import pytest

from documind.config import IngestionConfig, RetrievalConfig
from documind.ingestion.schemas import TextChunk
from documind.retrieval import RetrievalQuery, RetrievalResult, RetrievedChunk, SemanticRetriever
from documind.storage.embedder import BaseEmbedder, LocalEmbedder
from documind.storage.index import DocumentIndex
from documind.storage.vector_store import FaissVectorStore


class DeterministicMockEmbedder(BaseEmbedder):
    """Deterministic, lightweight pseudo-semantic embedder for fast unit tests.
    
    Guarantees:
    - Zero external model downloads or PyTorch/SentenceTransformer memory allocations.
    - Deterministic cosine similarities matching domain topics (RAG vs FAISS vs OOD).
    - Prevents MemoryError and process hangs during fast unit-test execution.
    """

    def __init__(self, dimension: int = 384) -> None:
        self._dim = dimension
        self._model_name = "deterministic-mock-embedder"

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def normalize_embeddings(self) -> bool:
        return True

    def embed_text(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            raise ValueError("Text to embed cannot be empty or whitespace-only")
        t = text.lower()
        v = np.zeros(self._dim, dtype=np.float32)
        # RAG semantic partition (dims 0:50)
        if any(w in t for w in ["rag", "retrieval", "augmented", "chunking", "overlap"]):
            v[0:50] += 1.0
        # FAISS semantic partition (dims 50:100)
        if any(w in t for w in ["faiss", "similarity", "dense", "vectors", "index", "inner product", "cpu", "clustering"]):
            v[50:100] += 1.0
        # Sourdough / OOD partition (dims 200:250)
        if any(w in t for w in ["sourdough", "oven", "temperature", "hydration", "bread"]):
            v[200:250] += 1.0
        # General background deterministic hash for other tokens
        tokens = re.findall(r"\b\w+\b", t)
        for tok in tokens:
            idx = int(hashlib.md5(tok.encode()).hexdigest()[:4], 16) % 100 + 100
            v[idx] += 0.05
        norm = float(np.linalg.norm(v))
        return (v / norm).astype(np.float32) if norm > 0 else v

    def embed_texts(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        if len(texts) == 0:
            return np.empty((0, self._dim), dtype=np.float32)
        return np.vstack([self.embed_text(x) for x in texts])


def create_sample_chunk(
    chunk_id: str,
    text: str,
    doc_id: str = "doc_1",
    doc_name: str = "test.pdf",
    start_page: int = 1,
    end_page: int = 1,
    chunk_index: int = 0,
) -> TextChunk:
    """Helper to generate a well-formed TextChunk with all 12 metadata attributes."""
    return TextChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        document_name=doc_name,
        source_path=f"data/{doc_name}",
        chunk_index=chunk_index,
        text=text,
        page_numbers=[start_page] if start_page == end_page else [start_page, end_page],
        start_page=start_page,
        end_page=end_page,
        char_start=0,
        char_end=len(text),
        chunking_strategy="recursive",
        chunk_size=1000,
        chunk_overlap=150,
        metadata={"domain": "ai_research"},
    )


@pytest.fixture
def populated_index():
    """Fixture initializing an in-memory DocumentIndex populated with test knowledge chunks."""
    embedder = DeterministicMockEmbedder()
    store = FaissVectorStore(dimension=embedder.dimension)
    index = DocumentIndex(embedder=embedder, vector_store=store)

    chunks = [
        create_sample_chunk(
            chunk_id="chunk_rag_01",
            text="Retrieval-Augmented Generation combines an external information retriever with an LLM generator.",
            doc_id="doc_rag",
            doc_name="rag_overview.pdf",
            start_page=1,
            end_page=1,
            chunk_index=0,
        ),
        create_sample_chunk(
            chunk_id="chunk_rag_02",
            text="Chunking strategies partition long PDFs into manageable semantic windows like 1000 characters with 150 overlap.",
            doc_id="doc_rag",
            doc_name="rag_overview.pdf",
            start_page=2,
            end_page=2,
            chunk_index=1,
        ),
        create_sample_chunk(
            chunk_id="chunk_faiss_01",
            text="FAISS provides efficient similarity search and clustering of dense vectors on local CPUs using AVX2 SIMD.",
            doc_id="doc_faiss",
            doc_name="faiss_manual.pdf",
            start_page=1,
            end_page=2,
            chunk_index=0,
        ),
        create_sample_chunk(
            chunk_id="chunk_faiss_02",
            text="IndexFlatIP performs exact inner product computations matching cosine similarity on normalized vectors.",
            doc_id="doc_faiss",
            doc_name="faiss_manual.pdf",
            start_page=3,
            end_page=3,
            chunk_index=1,
        ),
    ]
    index.add_chunks(chunks)
    return index


class TestSemanticRetrieverUnit:
    """Unit tests for query validation, baseline retrieval, relevance filtering, and telemetry."""

    def test_query_validation(self, populated_index):
        """Query validation rejects empty strings, whitespace, None, invalid types, and non-positive top_k."""
        retriever = SemanticRetriever(populated_index)

        with pytest.raises(ValueError, match="cannot be empty"):
            retriever.retrieve("")

        with pytest.raises(ValueError, match="cannot be empty"):
            retriever.retrieve("   \n\t  ")

        with pytest.raises(ValueError, match="cannot be empty"):
            retriever.retrieve(RetrievalQuery(text="   "))

        with pytest.raises(ValueError, match="must be a string or RetrievalQuery"):
            retriever.retrieve(None)  # type: ignore

        with pytest.raises(ValueError, match="must be a string or RetrievalQuery"):
            retriever.retrieve(12345)  # type: ignore

        with pytest.raises(ValueError, match="top_k must be positive"):
            retriever.retrieve("Valid query", top_k=0)

        with pytest.raises(ValueError, match="top_k must be positive"):
            retriever.retrieve("Valid query", top_k=-5)

    def test_empty_index_retrieval(self):
        """Retrieving against an empty index gracefully returns an empty result without crashing."""
        embedder = DeterministicMockEmbedder()
        store = FaissVectorStore(dimension=embedder.dimension)
        empty_index = DocumentIndex(embedder=embedder, vector_store=store)

        retriever = SemanticRetriever(empty_index)
        result = retriever.retrieve("What is retrieval augmented generation?", top_k=5)

        assert isinstance(result, RetrievalResult)
        assert result.total_retrieved == 0
        assert len(result.chunks) == 0
        assert result.total_below_threshold == 0
        assert result.embed_latency_ms == 0.0
        assert result.search_latency_ms == 0.0

    def test_top_k_ordering(self, populated_index):
        """Retrieved chunks must be strictly ordered in descending cosine similarity score."""
        retriever = SemanticRetriever(populated_index)
        result = retriever.retrieve("What is RAG and how does it work with language models?", top_k=4)

        assert result.total_retrieved > 0
        scores = [c.score for c in result.chunks]
        # Verify scores are sorted descending
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

        # Top result should be the RAG overview chunk
        assert result.chunks[0].chunk_id.startswith("chunk_rag")

    def test_no_threshold_baseline(self, populated_index):
        """When similarity_threshold is None, pure Top-K chunks are returned with total_below_threshold=0."""
        cfg = RetrievalConfig(top_k=3, similarity_threshold=None)
        retriever = SemanticRetriever(populated_index, config=cfg)

        result = retriever.retrieve("Any query about computing systems")
        assert result.total_retrieved == 3
        assert result.total_below_threshold == 0
        assert len(result.chunks) == 3

    def test_relevance_threshold_filtering(self, populated_index):
        """Candidate chunks below similarity_threshold are filtered out, and telemetry is recorded."""
        # Query specifically about FAISS
        query = "How does FAISS achieve fast vector indexing on CPU?"
        
        # Test with moderate threshold
        retriever = SemanticRetriever(populated_index)
        result = retriever.retrieve(query, top_k=4, similarity_threshold=0.30)

        # Chunks above 0.30 are kept
        assert all(c.score >= 0.30 for c in result.chunks)
        assert result.total_below_threshold >= 0
        assert result.total_retrieved + result.total_below_threshold == 4

    def test_all_below_threshold_returns_empty(self, populated_index):
        """Out-of-domain query with high threshold returns 0 chunks and counts all candidates as filtered."""
        retriever = SemanticRetriever(populated_index)
        # Sourdough baking query against an index containing RAG and FAISS documentation
        out_of_domain = "What is the optimal oven temperature and hydration for baking sourdough bread?"
        result = retriever.retrieve(out_of_domain, top_k=4, similarity_threshold=0.50)

        assert result.total_retrieved == 0
        assert len(result.chunks) == 0
        assert result.total_below_threshold == 4

    def test_metadata_fidelity(self, populated_index):
        """All 12 metadata attributes are preserved intact on RetrievedChunk."""
        retriever = SemanticRetriever(populated_index)
        result = retriever.retrieve("exact inner product vector indexing", top_k=1)

        assert len(result.chunks) == 1
        chunk: RetrievedChunk = result.chunks[0]

        assert isinstance(chunk.chunk_id, str)
        assert chunk.document_id in ["doc_rag", "doc_faiss"]
        assert chunk.document_name in ["rag_overview.pdf", "faiss_manual.pdf"]
        assert chunk.source_path.startswith("data/")
        assert chunk.chunk_index >= 0
        assert len(chunk.text) > 0
        assert isinstance(chunk.page_numbers, list)
        assert chunk.start_page >= 1
        assert chunk.end_page >= chunk.start_page
        assert chunk.char_start == 0
        assert chunk.char_end == len(chunk.text)
        assert chunk.chunking_strategy == "recursive"
        assert chunk.chunk_size == 1000
        assert chunk.chunk_overlap == 150
        assert chunk.metadata == {"domain": "ai_research"}
        assert -1.0 <= chunk.score <= 1.0

    def test_multi_document_retrieval(self, populated_index):
        """Retrieval across multiple documents returns results from distinct sources."""
        retriever = SemanticRetriever(populated_index)
        result = retriever.retrieve("Explain RAG retrieval systems and FAISS vector indices", top_k=4)

        doc_names = {c.document_name for c in result.chunks}
        # Query touches both concepts, should retrieve from both documents
        assert "rag_overview.pdf" in doc_names
        assert "faiss_manual.pdf" in doc_names

    def test_document_id_filter(self, populated_index):
        """Specifying document_id_filter restricts retrieval strictly to the target document."""
        retriever = SemanticRetriever(populated_index)
        query = RetrievalQuery(
            text="Explain vector search algorithms",
            top_k=3,
            document_id_filter="doc_faiss",
        )
        result = retriever.retrieve(query)

        assert result.total_retrieved > 0
        assert all(c.document_id == "doc_faiss" for c in result.chunks)
        assert all(c.document_name == "faiss_manual.pdf" for c in result.chunks)

    def test_disaggregated_latency_recording(self, populated_index):
        """Verifies that embed, search, and total latencies are measured separately as positive values."""
        retriever = SemanticRetriever(populated_index)
        result = retriever.retrieve("Dense vector retrieval latency benchmark", top_k=2)

        assert result.embed_latency_ms > 0.0
        assert result.search_latency_ms >= 0.0
        assert result.total_latency_ms >= (result.embed_latency_ms + result.search_latency_ms)

    def test_retrieval_offline_guarantee(self, populated_index, monkeypatch):
        """Confirms that retrieval executes purely in local CPU memory with zero socket operations."""
        import socket

        def mock_socket_connect(*args, **kwargs):
            raise AssertionError("Network socket connection attempted during retrieval!")

        monkeypatch.setattr(socket.socket, "connect", mock_socket_connect)

        retriever = SemanticRetriever(populated_index)
        result = retriever.retrieve("Testing strict offline retrieval without internet access", top_k=2)
        assert result.total_retrieved > 0

    def test_document_scoping_from_natural_language_query(self, populated_index):
        """Natural language queries mentioning indexed document filenames automatically scope retrieval."""
        retriever = SemanticRetriever(populated_index)
        query = "Summarize the key architecture points in rag_overview.pdf"
        result = retriever.retrieve(query, top_k=3)

        assert result.total_retrieved > 0
        assert all(c.document_id == "doc_rag" for c in result.chunks)
        assert all(c.document_name == "rag_overview.pdf" for c in result.chunks)

    def test_document_scoping_direct_parameter(self, populated_index):
        """Passing document_id parameter directly to retrieve() strictly filters candidate pool."""
        retriever = SemanticRetriever(populated_index)
        result = retriever.retrieve("Explain indexing", top_k=3, document_id="doc_faiss")

        assert result.total_retrieved > 0
        assert all(c.document_id == "doc_faiss" for c in result.chunks)

    def test_author_metadata_query_retrieves_title_chunk(self, populated_index):
        """Author queries for a specific document retrieve the title-page chunk (chunk_index=0) with elevated score."""
        retriever = SemanticRetriever(populated_index)
        query = "Who are the authors of rag_overview.pdf? Please check the first page and cite the source."
        result = retriever.retrieve(query, top_k=4, similarity_threshold=0.35)

        assert result.total_retrieved > 0
        # Title page chunk must be present and at index 0
        top_chunk = result.chunks[0]
        assert top_chunk.chunk_index == 0
        assert top_chunk.document_id == "doc_rag"
        assert top_chunk.score == 0.90
        assert top_chunk.retrieval_source == "metadata_prioritized"
        assert top_chunk.raw_score is not None
        assert top_chunk.metadata.get("retrieval_mode") == "metadata_prioritized"
        assert top_chunk.start_page == 1

    def test_multi_pdf_metadata_isolation(self, populated_index):
        """Metadata query naming faiss_manual.pdf retrieves chunk 0 of FAISS and not RAG overview."""
        retriever = SemanticRetriever(populated_index)
        query = "Who wrote faiss_manual.pdf?"
        result = retriever.retrieve(query, top_k=3, similarity_threshold=0.35)

        assert result.total_retrieved > 0
        top_chunk = result.chunks[0]
        assert top_chunk.document_id == "doc_faiss"
        assert top_chunk.chunk_index == 0
        assert top_chunk.document_name == "faiss_manual.pdf"
        assert not any(c.document_id == "doc_rag" for c in result.chunks)

    def test_threshold_false_negative_prevention_on_metadata(self, populated_index):
        """Metadata path prevents dense similarity threshold from false-negative rejecting title chunks."""
        retriever = SemanticRetriever(populated_index)
        # Even with high similarity threshold of 0.45
        query = "Who are the authors of rag_overview.pdf?"
        result = retriever.retrieve(query, top_k=2, similarity_threshold=0.45)

        assert result.total_retrieved >= 1
        assert result.chunks[0].chunk_index == 0
        assert result.chunks[0].score >= 0.45

    def test_out_of_domain_query_does_not_trigger_metadata_path(self, populated_index):
        """General non-metadata out-of-domain queries do not inject title chunks."""
        retriever = SemanticRetriever(populated_index)
        ood_query = "What is the optimal oven temperature and hydration for baking sourdough bread?"
        result = retriever.retrieve(ood_query, top_k=4, similarity_threshold=0.35)

        assert result.total_retrieved == 0
        assert len(result.chunks) == 0

