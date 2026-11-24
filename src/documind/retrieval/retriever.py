"""Modular semantic retrieval engine supporting unthresholded baselines and candidate filtering."""

from pathlib import Path
import re
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np

from documind.config import RetrievalConfig
from documind.retrieval.schemas import RetrievalQuery, RetrievalResult, RetrievedChunk
from documind.storage.index import DocumentIndex
from documind.storage.schemas import SearchResult

# Patterns identifying metadata/authorship/title-page queries
METADATA_QUERY_PATTERNS = re.compile(
    r"\b(author|authors|who\s+wrote|writer|written\s+by|creator|submitted\s+by|"
    r"first\s+page|title\s+page|cover\s+page|front\s+page|metadata)\b",
    re.IGNORECASE,
)


class BaseRetriever(ABC):
    """Abstract interface for document retrieval engines in DocuMind."""

    @abstractmethod
    def retrieve(
        self,
        query: Union[str, RetrievalQuery],
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        document_id: Optional[str] = None,
    ) -> RetrievalResult:
        """Retrieve relevant chunks matching the query."""
        pass


class SemanticRetriever(BaseRetriever):
    """Semantic dense retriever powered by DocumentIndex, LocalEmbedder, and FAISS.
    
    Supports:
    1. Unthresholded baseline retrieval (similarity_threshold=None) returning pure Top-K chunks.
    2. Calibrated relevance threshold filtering discarding low-similarity distractors.
    3. Document-scoped retrieval via parameter or natural language query resolution.
    4. Grounded title-page / front-matter retrieval path for document metadata & author queries.
    5. Fine-grained latency disaggregation (embed, search, total).
    6. Strict input validation rejecting empty/whitespace inputs.
    7. Graceful handling of empty indexes and out-of-domain queries.
    """

    def __init__(
        self,
        document_index: DocumentIndex,
        config: Optional[RetrievalConfig] = None,
    ) -> None:
        self._index = document_index
        self._config = config or RetrievalConfig()

    @property
    def document_index(self) -> DocumentIndex:
        return self._index

    @property
    def config(self) -> RetrievalConfig:
        return self._config

    @staticmethod
    def _clean_doc_name(name: str) -> str:
        """Strip 12-hex prefix like '95cca1e4774a_' if present on indexed filenames."""
        base = Path(name.strip()).name
        if len(base) > 13 and base[12] == "_" and all(c in "0123456789abcdefABCDEF" for c in base[:12]):
            return base[13:]
        return base

    def _resolve_document_from_query(self, query_text: str) -> Optional[str]:
        """Inspect query text for mentions of indexed document filenames or stems."""
        if not hasattr(self._index, "vector_store") or not hasattr(self._index.vector_store, "records"):
            return None

        records = self._index.vector_store.records
        if not records:
            return None

        query_lower = query_text.lower()
        doc_candidates: Dict[str, Tuple[str, str]] = {}
        for r in records:
            if r.document_id not in doc_candidates:
                clean = self._clean_doc_name(r.document_name).lower()
                stem = Path(clean).stem.lower()
                doc_candidates[r.document_id] = (clean, stem)

        for doc_id, (clean, stem) in doc_candidates.items():
            if clean in query_lower:
                return doc_id
            if len(stem) >= 3:
                pattern = rf"\b{re.escape(stem)}\b"
                if re.search(pattern, query_lower):
                    return doc_id

        return None

    def _get_title_page_chunk(
        self,
        doc_id: str,
        query_vec: Optional[np.ndarray] = None,
        raw_results: Optional[List[SearchResult]] = None,
    ) -> Optional[RetrievedChunk]:
        """Retrieve the title-page / front-matter chunk for a document, recording actual cosine similarity."""
        if not hasattr(self._index, "vector_store") or not hasattr(self._index.vector_store, "records"):
            return None

        matching_records = [
            r for r in self._index.vector_store.records if r.document_id == doc_id
        ]
        if not matching_records:
            return None

        # Select chunk with lowest chunk_index (and lowest start_page)
        first_record = min(matching_records, key=lambda r: (r.chunk_index, r.start_page))

        # Determine actual cosine similarity from raw search results or direct reconstruction
        actual_cosine: Optional[float] = None
        if raw_results is not None:
            for r in raw_results:
                if r.record.chunk_id == first_record.chunk_id:
                    actual_cosine = float(r.score)
                    break

        if actual_cosine is None and query_vec is not None:
            try:
                records = self._index.vector_store.records
                idx = next(i for i, r in enumerate(records) if r.chunk_id == first_record.chunk_id)
                if hasattr(self._index.vector_store, "_index") and hasattr(self._index.vector_store._index, "reconstruct"):
                    rec_vec = self._index.vector_store._index.reconstruct(idx)
                    actual_cosine = float(np.dot(query_vec.flatten(), rec_vec.flatten()))
            except Exception:
                actual_cosine = None

        metadata = dict(first_record.metadata)
        if actual_cosine is not None:
            metadata["actual_cosine_similarity"] = actual_cosine
        metadata["retrieval_mode"] = "metadata_prioritized"

        return RetrievedChunk(
            chunk_id=first_record.chunk_id,
            document_id=first_record.document_id,
            document_name=first_record.document_name,
            source_path=first_record.source_path,
            chunk_index=first_record.chunk_index,
            text=first_record.text,
            page_numbers=list(first_record.page_numbers),
            start_page=first_record.start_page,
            end_page=first_record.end_page,
            char_start=first_record.char_start,
            char_end=first_record.char_end,
            score=0.90,  # Prioritized relevance score for ranking and thresholding
            raw_score=actual_cosine,  # Separately labeled actual cosine similarity
            retrieval_source="metadata_prioritized",
            chunking_strategy=first_record.chunking_strategy,
            chunk_size=first_record.chunk_size,
            chunk_overlap=first_record.chunk_overlap,
            metadata=metadata,
        )

    def retrieve(
        self,
        query: Union[str, RetrievalQuery],
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        document_id: Optional[str] = None,
    ) -> RetrievalResult:
        """Retrieve relevant chunks for a natural language query or RetrievalQuery object."""
        t0 = time.perf_counter()

        # 1. Parse and validate query input
        explicit_query_doc_filter = None
        if isinstance(query, str):
            query_text = query.strip()
            if not query_text:
                raise ValueError("Search query cannot be empty or whitespace-only")
            effective_top_k = top_k if top_k is not None else self._config.top_k
            effective_threshold = (
                similarity_threshold
                if similarity_threshold is not None
                else self._config.similarity_threshold
            )
        elif isinstance(query, RetrievalQuery):
            query_text = query.text.strip()
            if not query_text:
                raise ValueError("Search query cannot be empty or whitespace-only")
            effective_top_k = (
                top_k
                if top_k is not None
                else (query.top_k if query.top_k is not None else self._config.top_k)
            )
            if similarity_threshold is not None:
                effective_threshold = similarity_threshold
            elif query.similarity_threshold is not None:
                effective_threshold = query.similarity_threshold
            else:
                effective_threshold = self._config.similarity_threshold
            explicit_query_doc_filter = query.document_id_filter
        else:
            raise ValueError(
                f"Query must be a string or RetrievalQuery instance, got {type(query)}"
            )

        if effective_top_k <= 0:
            raise ValueError(f"top_k must be positive, got {effective_top_k}")

        # Document scoping resolution precedence:
        # 1. Explicit document_id parameter
        # 2. Query object document_id_filter
        # 3. Mention of document name/stem in query text
        # 4. Config default document_id_filter
        if document_id is not None:
            doc_filter = document_id
        elif explicit_query_doc_filter is not None:
            doc_filter = explicit_query_doc_filter
        else:
            resolved_doc = self._resolve_document_from_query(query_text)
            doc_filter = (
                resolved_doc
                if resolved_doc is not None
                else self._config.document_id_filter
            )

        # 2. Check if index contains vectors
        if self._index.total_vectors == 0:
            total_time = (time.perf_counter() - t0) * 1000.0
            return RetrievalResult(
                query=query_text,
                chunks=[],
                total_retrieved=0,
                total_below_threshold=0,
                embed_latency_ms=0.0,
                search_latency_ms=0.0,
                total_latency_ms=total_time,
            )

        # 3. Disaggregated Timing: Query Embedding (LocalEmbedder on CPU)
        t_embed_start = time.perf_counter()
        query_vec = self._index.embedder.embed_text(query_text)
        embed_latency_ms = (time.perf_counter() - t_embed_start) * 1000.0

        # 4. Disaggregated Timing: FAISS Inner Product Search
        t_search_start = time.perf_counter()
        if doc_filter is not None:
            # Full candidate pool search when document scope is active
            fetch_k = self._index.total_vectors
        else:
            fetch_k = (
                max(effective_top_k * 4, 20)
                if effective_threshold is not None
                else effective_top_k
            )
            fetch_k = min(fetch_k, self._index.total_vectors)

        raw_results = self._index.vector_store.search(query_vec, top_k=fetch_k)
        search_latency_ms = (time.perf_counter() - t_search_start) * 1000.0

        # 5. Apply Document Filter if requested
        if doc_filter is not None:
            raw_results = [
                r for r in raw_results if r.record.document_id == doc_filter
            ]

        # 6. Convert to RetrievedChunk records
        converted_chunks = [
            RetrievedChunk.from_search_result(r) for r in raw_results
        ]

        # 7. Grounded Title-Page / Metadata Retrieval Path
        # Dense embeddings of author questions ("who is the author") have low cosine
        # similarity to raw name blocks on title pages ("By Anjali Singh").
        # If the query asks for author, writer, or title page metadata, include the
        # target document's real title-page chunk (chunk_index=0).
        is_metadata_query = bool(METADATA_QUERY_PATTERNS.search(query_text))
        if is_metadata_query:
            target_docs: List[str] = []
            if doc_filter is not None:
                target_docs = [doc_filter]
            elif hasattr(self._index, "vector_store") and hasattr(self._index.vector_store, "records"):
                distinct_doc_ids = list(dict.fromkeys(r.document_id for r in self._index.vector_store.records))
                if len(distinct_doc_ids) == 1:
                    target_docs = distinct_doc_ids

            for t_doc in target_docs:
                title_chunk = self._get_title_page_chunk(
                    t_doc,
                    query_vec=query_vec,
                    raw_results=raw_results,
                )
                if title_chunk is not None:
                    # Deduplicate and prepend to candidates
                    converted_chunks = [
                        c for c in converted_chunks if c.chunk_id != title_chunk.chunk_id
                    ]
                    converted_chunks.insert(0, title_chunk)

        # 8. Apply Relevance Threshold Filtering
        total_below_threshold = 0
        if effective_threshold is not None:
            kept_chunks = []
            for chunk in converted_chunks:
                if chunk.score >= effective_threshold:
                    kept_chunks.append(chunk)
                else:
                    total_below_threshold += 1
            final_chunks = kept_chunks[:effective_top_k]
        else:
            # Unthresholded baseline: keep all top_k candidate chunks
            final_chunks = converted_chunks[:effective_top_k]
            total_below_threshold = 0

        total_latency_ms = (time.perf_counter() - t0) * 1000.0

        return RetrievalResult(
            query=query_text,
            chunks=final_chunks,
            total_retrieved=len(final_chunks),
            total_below_threshold=total_below_threshold,
            embed_latency_ms=embed_latency_ms,
            search_latency_ms=search_latency_ms,
            total_latency_ms=total_latency_ms,
        )
