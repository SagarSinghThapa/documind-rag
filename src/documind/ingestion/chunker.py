"""Configurable text chunking implementations with cross-page metadata tracking.

Supports:
1. Fixed-size chunking (character-based sliding window).
2. Recursive character chunking (separator-hierarchy splitting).

All chunk sizes and overlaps are measured in characters.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import hashlib

from documind.ingestion.schemas import DocumentPage, TextChunk


class BaseChunker(ABC):
    """Abstract base class for document chunkers."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        """Initialize chunker with size and overlap parameters in characters.
        
        Args:
            chunk_size: Maximum chunk size in characters (must be > 0).
            chunk_overlap: Overlap between consecutive chunks in characters (0 <= overlap < chunk_size).
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Name of the chunking strategy."""
        pass

    @abstractmethod
    def split_text(self, text: str) -> List[Tuple[str, int, int]]:
        """Split a unified text string into (chunk_text, char_start, char_end) tuples."""
        pass

    def chunk_document(self, pages: List[DocumentPage]) -> List[TextChunk]:
        """Chunk an ordered list of document pages, tracking contributing page spans.
        
        Args:
            pages: List of DocumentPage records belonging to a single document.
            
        Returns:
            List of TextChunk records with exact page attribution.
        """
        if not pages:
            return []

        doc_id = pages[0].document_id
        doc_name = pages[0].document_name
        source_path = pages[0].source_path

        # 1. Build unified document text and page character-offset map
        unified_text_parts: List[str] = []
        # List of (page_num, start_offset, end_offset)
        page_spans: List[Tuple[int, int, int]] = []
        current_offset = 0

        for page in pages:
            page_text = page.text
            if not page_text:
                continue

            if unified_text_parts:
                # Page separator
                separator = "\n\n"
                unified_text_parts.append(separator)
                current_offset += len(separator)

            start = current_offset
            unified_text_parts.append(page_text)
            current_offset += len(page_text)
            end = current_offset
            page_spans.append((page.page_number, start, end))

        unified_text = "".join(unified_text_parts)
        if not unified_text.strip():
            return []

        # 2. Split unified text using concrete chunker
        raw_chunks = self.split_text(unified_text)

        # 3. Construct TextChunk records with exact page attribution
        chunks: List[TextChunk] = []
        for raw_chunk_text, c_start, c_end in raw_chunks:
            chunk_text = raw_chunk_text.strip()
            if not chunk_text:
                continue

            # Align character offsets to stripped content boundaries
            leading_ws = len(raw_chunk_text) - len(raw_chunk_text.lstrip())
            trailing_ws = len(raw_chunk_text) - len(raw_chunk_text.rstrip())
            adj_start = c_start + leading_ws
            adj_end = c_end - trailing_ws

            chunk_index = len(chunks)

            # Identify all contributing pages using aligned offsets
            contributing_pages = []
            for p_num, p_start, p_end in page_spans:
                # Overlap test: max(start1, start2) < min(end1, end2)
                if max(adj_start, p_start) < min(adj_end, p_end):
                    contributing_pages.append(p_num)

            # Fallback if chunk lies strictly on boundary whitespace
            if not contributing_pages:
                closest_page = page_spans[0][0]
                min_dist = float("inf")
                for p_num, p_start, p_end in page_spans:
                    dist = min(abs(adj_start - p_start), abs(adj_start - p_end))
                    if dist < min_dist:
                        min_dist = dist
                        closest_page = p_num
                contributing_pages = [closest_page]

            contributing_pages = sorted(list(set(contributing_pages)))
            start_page = contributing_pages[0]
            end_page = contributing_pages[-1]

            # Deterministic chunk ID
            # Combines doc_id, chunk_index, strategy, size, overlap
            chunk_id = f"{doc_id[:12]}_c{chunk_index:04d}_{self.strategy_name[:3]}"

            chunk_record = TextChunk(
                chunk_id=chunk_id,
                document_id=doc_id,
                document_name=doc_name,
                source_path=source_path,
                chunk_index=chunk_index,
                text=chunk_text,
                page_numbers=contributing_pages,
                start_page=start_page,
                end_page=end_page,
                chunking_strategy=self.strategy_name,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                char_start=adj_start,
                char_end=adj_end,
                metadata={
                    "char_count": len(chunk_text),
                    "is_multipage": len(contributing_pages) > 1,
                },
            )
            chunks.append(chunk_record)

        return chunks


class FixedSizeChunker(BaseChunker):
    """Chunks text into strict fixed-size character windows with overlap."""

    @property
    def strategy_name(self) -> str:
        return "fixed"

    def split_text(self, text: str) -> List[Tuple[str, int, int]]:
        """Split text by sliding character window.
        
        Args:
            text: Input text string.
            
        Returns:
            List of (chunk_text, char_start, char_end).
        """
        if not text:
            return []

        chunks: List[Tuple[str, int, int]] = []
        text_len = len(text)
        step = self.chunk_size - self.chunk_overlap

        start = 0
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append((chunk_text, start, end))
            if end >= text_len:
                break
            start += step

        return chunks


class RecursiveCharacterChunker(BaseChunker):
    """Recursively splits text using a hierarchy of structural delimiters.
    
    Default delimiters: ["\n\n", "\n", ". ", " ", ""]
    Attempts to preserve paragraphs first, then lines, sentences, words, and characters.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        separators: Optional[List[str]] = None,
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    @property
    def strategy_name(self) -> str:
        return "recursive"

    def split_text(self, text: str) -> List[Tuple[str, int, int]]:
        """Split text recursively by delimiter hierarchy.
        
        Args:
            text: Input text string.
            
        Returns:
            List of (chunk_text, char_start, char_end).
        """
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [(text, 0, len(text))]

        splits = self._split_recursive(text, self.separators)
        return self._merge_splits(splits, text)

    def _split_recursive(self, text: str, separators: List[str]) -> List[Tuple[str, int]]:
        """Recursively split text, tracking original text offsets.
        
        Returns:
            List of (segment_text, start_offset)
        """
        final_chunks: List[Tuple[str, int]] = []
        separator = separators[-1]
        new_separators: List[str] = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        # Track offsets
        current_offset = 0
        for i, s in enumerate(splits):
            if separator and i > 0:
                current_offset += len(separator)

            seg_start = current_offset
            seg_len = len(s)
            current_offset += seg_len

            if not s:
                continue

            if len(s) <= self.chunk_size:
                final_chunks.append((s, seg_start))
            else:
                if new_separators:
                    sub_splits = self._split_recursive(s, new_separators)
                    for sub_text, sub_off in sub_splits:
                        final_chunks.append((sub_text, seg_start + sub_off))
                else:
                    # Character level fallback
                    step = self.chunk_size - self.chunk_overlap
                    for c_start in range(0, len(s), step):
                        c_end = min(c_start + self.chunk_size, len(s))
                        final_chunks.append((s[c_start:c_end], seg_start + c_start))
                        if c_end >= len(s):
                            break

        return final_chunks

    def _merge_splits(self, splits: List[Tuple[str, int]], original_text: str) -> List[Tuple[str, int, int]]:
        """Merge split segments into overlapping chunks up to chunk_size preserving exact text."""
        if not splits:
            return []

        merged: List[Tuple[str, int, int]] = []
        current_doc: List[Tuple[str, int]] = []

        for s, seg_start in splits:
            seg_len = len(s)
            seg_end = seg_start + seg_len

            if current_doc:
                potential_start = current_doc[0][1]
                potential_len = seg_end - potential_start
                if potential_len > self.chunk_size:
                    c_start = current_doc[0][1]
                    c_end = current_doc[-1][1] + len(current_doc[-1][0])
                    chunk_text = original_text[c_start:c_end]
                    if chunk_text.strip():
                        merged.append((chunk_text, c_start, c_end))

                    while current_doc and (
                        (c_end - current_doc[0][1] > self.chunk_overlap)
                        or (seg_end - current_doc[0][1] > self.chunk_size)
                    ):
                        current_doc.pop(0)

            current_doc.append((s, seg_start))

        # Emit remaining document segment
        if current_doc:
            c_start = current_doc[0][1]
            c_end = current_doc[-1][1] + len(current_doc[-1][0])
            chunk_text = original_text[c_start:c_end]
            if chunk_text.strip():
                merged.append((chunk_text, c_start, c_end))

        return merged


def create_chunker(
    strategy: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    separators: Optional[List[str]] = None,
) -> BaseChunker:
    """Factory function for instantiating document chunkers.
    
    Args:
        strategy: 'recursive' or 'fixed'.
        chunk_size: Target size in characters.
        chunk_overlap: Overlap in characters.
        separators: Optional custom delimiters for recursive chunker.
        
    Returns:
        Configured BaseChunker instance.
    """
    strategy_clean = strategy.lower().strip()
    if strategy_clean in ("recursive", "recursive_character", "separator"):
        return RecursiveCharacterChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )
    elif strategy_clean in ("fixed", "fixed_size", "sliding_window"):
        return FixedSizeChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        raise ValueError(f"Unknown chunking strategy '{strategy}'. Supported strategies: 'recursive', 'fixed'")
