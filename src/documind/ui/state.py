"""Session state initialization, cached resource management, and safe index lifecycle."""

import json
import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from documind.config import GenerationConfig, RetrievalConfig
from documind.generation.citation import CitationParser
from documind.generation.generator import GroundedGenerator
from documind.generation.llm import BaseLLM, MockLLM, TransformersLLM
from documind.retrieval.context_builder import ContextBuilder
from documind.retrieval.retriever import SemanticRetriever
from documind.storage import DocumentIndex, FaissVectorStore, LocalEmbedder
from documind.storage.embedder import BaseEmbedder
import numpy as np

logger = logging.getLogger(__name__)

# Process-level inference lock to serialize CPU forward passes
_INFERENCE_LOCK = threading.Lock()


def get_inference_lock() -> threading.Lock:
    """Return the global lock for thread-safe CPU inference."""
    return _INFERENCE_LOCK


class MockEmbedder(BaseEmbedder):
    """Fast, deterministic mock embedder for headless testing and zero-overhead UI tests."""

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension
        self._model_name = "mock-embedder-384"
        self._normalize = True

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
        vec = np.ones(self._dimension, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        return (vec / norm).astype(np.float32)

    def embed_texts(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)
        v = self.embed_text("sample")
        return np.repeat(v[np.newaxis, :], len(texts), axis=0).astype(np.float32)


@st.cache_resource(show_spinner=False)
def get_embedder(use_mock: bool = False) -> BaseEmbedder:
    """Load and cache the singleton LocalEmbedder or MockEmbedder instance globally."""
    if use_mock:
        return MockEmbedder()
    return LocalEmbedder()


@st.cache_resource(show_spinner=False)
def get_llm(use_mock: bool = False, model_name: Optional[str] = None) -> BaseLLM:
    """Load and cache the singleton language model engine globally.
    
    Args:
        use_mock: If True, returns a deterministic MockLLM for unit testing.
        model_name: Optional model identifier override.
    """
    if use_mock:
        return MockLLM(default_response="Mock answer [S1].")

    cfg = GenerationConfig(
        model_name=model_name or "Qwen/Qwen2.5-0.5B-Instruct",
        cache_folder="models/llm",
        device="cpu",
        local_files_only=True,
        temperature=0.0,
        max_new_tokens=256,
        max_context_chars=1800,
        num_threads=8,
        include_exemplar=True,
    )
    llm = TransformersLLM(config=cfg)
    return llm


def load_persistent_registry(index_dir: str = "data/index") -> Dict[str, Any]:
    """Load document registry metadata from disk if available."""
    reg_path = Path(index_dir) / "document_registry.json"
    if reg_path.exists():
        try:
            with open(reg_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load document_registry.json: %s", e)
    return {}


def save_persistent_registry(registry: Dict[str, Any], index_dir: str = "data/index") -> None:
    """Flush document registry metadata to disk."""
    idx_path = Path(index_dir)
    idx_path.mkdir(parents=True, exist_ok=True)
    reg_path = idx_path / "document_registry.json"
    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def init_session_state(
    use_mock_llm: bool = False,
    data_dir: Optional[str] = None,
    index_dir: Optional[str] = None,
) -> None:
    """Initialize per-session isolated state variables in Streamlit.
    
    Guarantees:
    1. Read-only model weights are shared via @st.cache_resource.
    2. Mutable vector stores, documents, and chat history are isolated in st.session_state.
    3. Existing on-disk index snapshots and document registries are hydrated on startup.
    """
    data_dir = data_dir or os.environ.get("DOCUMIND_DATA_DIR", "data")
    index_dir = index_dir or os.environ.get("DOCUMIND_INDEX_DIR", "data/index")
    # 1. Cached singleton models
    use_mock = (
        use_mock_llm
        or (os.environ.get("DOCUMIND_USE_MOCK_LLM", "0") == "1")
        or st.session_state.get("use_mock_llm", False)
    )
    st.session_state.use_mock_llm = use_mock
    embedder = get_embedder(use_mock=use_mock)
    llm = get_llm(use_mock=use_mock)

    # 2. Document Index & Snapshot Hydration
    if "document_index" not in st.session_state:
        vector_store = FaissVectorStore(dimension=embedder.dimension)
        active_snap = Path(index_dir) / "active_snapshot.json"
        if active_snap.exists():
            try:
                vector_store.load(index_dir)
                logger.info("Loaded active vector store snapshot from %s", index_dir)
            except Exception as e:
                logger.warning("Could not restore snapshot from %s: %s", index_dir, e)
                vector_store = FaissVectorStore(dimension=embedder.dimension)

        doc_index = DocumentIndex(embedder=embedder, vector_store=vector_store)
        st.session_state.document_index = doc_index

    # 3. Document Registry
    if "indexed_docs" not in st.session_state:
        st.session_state.indexed_docs = load_persistent_registry(index_dir)

    # 4. Retrieval & Generation Pipeline
    if "retriever" not in st.session_state:
        retrieval_cfg = RetrievalConfig(top_k=5, similarity_threshold=0.35)
        st.session_state.retriever = SemanticRetriever(
            document_index=st.session_state.document_index,
            config=retrieval_cfg,
        )

    if "context_builder" not in st.session_state:
        st.session_state.context_builder = ContextBuilder(max_chars=1800)

    if "citation_parser" not in st.session_state:
        st.session_state.citation_parser = CitationParser()

    if "generator" not in st.session_state:
        gen_cfg = GenerationConfig(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            max_context_chars=1800,
            include_exemplar=True,
        )
        st.session_state.generator = GroundedGenerator(
            retriever=st.session_state.retriever,
            llm=llm,
            context_builder=st.session_state.context_builder,
            citation_parser=st.session_state.citation_parser,
            config=gen_cfg,
        )

    # 5. Conversational History & UI Flags
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "error_notice" not in st.session_state:
        st.session_state.error_notice = None

    if "enable_streaming" not in st.session_state:
        st.session_state.enable_streaming = False


def safe_reset_index(
    data_dir: Optional[str] = None,
    index_dir: Optional[str] = None,
    upload_dir: Optional[str] = None,
) -> None:
    """Perform a safe, confirmed reset of in-memory and on-disk indexed documents.
    
    Scope of Deletion:
    - Clears in-memory vectors and chunk tracking in FaissVectorStore.
    - Clears st.session_state.indexed_docs and st.session_state.messages.
    - Removes uploaded PDF files in data/uploads/.
    - Removes index snapshots in data/index/snapshots/ and clears document_registry.json.
    - Re-initializes a clean, empty DocumentIndex.
    
    SAFETY INVARIANT:
    - NEVER touches models/ (model weights) or research/ or source code.
    """
    data_dir = data_dir or os.environ.get("DOCUMIND_DATA_DIR", "data")
    index_dir = index_dir or os.environ.get("DOCUMIND_INDEX_DIR", "data/index")
    upload_dir = upload_dir or os.environ.get("DOCUMIND_UPLOAD_DIR", "data/uploads")
    # 1. Reset in-memory structures
    use_mock = st.session_state.get("use_mock_llm", False) or (os.environ.get("DOCUMIND_USE_MOCK_LLM", "0") == "1")
    embedder = get_embedder(use_mock=use_mock)
    new_store = FaissVectorStore(dimension=embedder.dimension)
    new_index = DocumentIndex(embedder=embedder, vector_store=new_store)
    st.session_state.document_index = new_index

    if "retriever" in st.session_state:
        st.session_state.retriever._index = new_index
    if "generator" in st.session_state:
        st.session_state.generator._retriever = st.session_state.retriever

    st.session_state.indexed_docs = {}
    st.session_state.messages = []
    st.session_state.error_notice = None

    # 2. Clean uploads directory
    uploads_path = Path(upload_dir)
    if uploads_path.exists():
        for item in uploads_path.iterdir():
            if item.is_file():
                try:
                    item.unlink()
                except OSError as e:
                    logger.warning("Could not delete uploaded file %s: %s", item, e)

    # 3. Clean index snapshots directory
    idx_path = Path(index_dir)
    snapshots_dir = idx_path / "snapshots"
    if snapshots_dir.exists():
        try:
            shutil.rmtree(snapshots_dir)
            snapshots_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning("Could not delete snapshots directory: %s", e)

    # 4. Remove active snapshot pointer and clear registry
    active_ptr = idx_path / "active_snapshot.json"
    if active_ptr.exists():
        try:
            active_ptr.unlink()
        except OSError as e:
            logger.warning("Could not delete active_snapshot.json: %s", e)

    save_persistent_registry({}, index_dir=index_dir)
    logger.info("Safe index reset completed successfully.")
