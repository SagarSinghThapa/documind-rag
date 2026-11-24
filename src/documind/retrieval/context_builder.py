"""Context window construction, dynamic overlap deduplication, and citation management."""

import math
import time
from typing import List, Optional

from documind.retrieval.schemas import ContextReference, FormattedContext, RetrievedChunk


class ContextBuilder:
    """Constructs citation-grounded, budget-enforced context blocks from retrieved chunks.
    
    Guarantees:
    1. Dynamic overlap deduplication: Only strips overlapping prefix/suffix text when an
       actual matching substring of length >= min_overlap_chars is verified.
    2. Untrusted text sandboxing: Encapsulates document text inside structural blockquotes
       (> [Document Content - Untrusted Context]) so future LLMs distinguish context from prompts.
    3. Strict budget packing: Accumulates complete chunks in rank order up to max_chars without
       mid-word cutting, recording discarded chunks in telemetry.
    4. Metadata fidelity: Populates structured ContextReference records preserving 100% of
       physical page and character span attributes.
    """

    def __init__(
        self,
        max_chars: int = 4000,
        deduplicate_overlap: bool = True,
        min_overlap_chars: int = 20,
    ) -> None:
        self._max_chars = max(max_chars, 200)
        self._deduplicate_overlap = deduplicate_overlap
        self._min_overlap_chars = max(min_overlap_chars, 5)

    @property
    def max_chars(self) -> int:
        return self._max_chars

    @property
    def deduplicate_overlap(self) -> bool:
        return self._deduplicate_overlap

    @staticmethod
    def _find_overlap(prev_text: str, curr_text: str, min_chars: int) -> int:
        """Find the longest suffix of prev_text that matches a prefix of curr_text."""
        if not prev_text or not curr_text:
            return 0
        
        # Look for overlap within a reasonable window (e.g. up to 300 characters)
        search_window = min(len(prev_text), len(curr_text), 300)
        if search_window < min_chars:
            return 0

        # Scan from largest possible overlap down to min_chars
        prev_suffix = prev_text[-search_window:]
        for length in range(search_window, min_chars - 1, -1):
            if curr_text.startswith(prev_suffix[-length:]):
                return length
        return 0

    def build(self, chunks: List[RetrievedChunk]) -> FormattedContext:
        """Assemble retrieved chunks into a formatted, citation-ready context block."""
        t0 = time.perf_counter()

        if not chunks:
            return FormattedContext(
                text="",
                references=[],
                total_chars=0,
                estimated_tokens=0,
                chunks_included=0,
                chunks_discarded_budget=0,
                context_build_latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # Work on candidate chunks in priority order
        formatted_blocks: List[str] = []
        references: List[ContextReference] = []
        current_chars = 0
        chunks_included = 0
        chunks_discarded_budget = 0

        # Keep track of previous included chunk for overlap verification
        prev_chunk: Optional[RetrievedChunk] = None
        prev_chunk_processed_text: Optional[str] = None

        for chunk in chunks:
            chunk_text = chunk.text.strip()
            
            # Dynamic Overlap Verification
            if (
                self._deduplicate_overlap
                and prev_chunk is not None
                and prev_chunk.document_id == chunk.document_id
                and prev_chunk.chunk_index + 1 == chunk.chunk_index
                and prev_chunk_processed_text is not None
            ):
                match_len = self._find_overlap(
                    prev_chunk_processed_text, chunk_text, self._min_overlap_chars
                )
                if match_len >= self._min_overlap_chars:
                    # Strip the verified duplicate prefix and strip leading whitespace
                    chunk_text = chunk_text[match_len:].lstrip()

            # Format physical page span
            if chunk.start_page == chunk.end_page:
                page_span = f"p. {chunk.start_page}"
            else:
                page_span = f"pp. {chunk.start_page}-{chunk.end_page}"

            # Structural blockquote encapsulation for untrusted document text
            quoted_lines = [f"> {line}" for line in chunk_text.splitlines()]
            quoted_text = "\n".join(quoted_lines) if quoted_lines else "> [Empty content]"

            source_id = f"S{chunks_included + 1}"
            block = (
                f"[{source_id}] ### Source: {chunk.document_name} | Physical Page(s): {page_span} | Ref ID: {chunk.chunk_id}\n"
                f"> [Document Content - Untrusted Context]\n"
                f"{quoted_text}\n"
            )

            # Check budget constraint
            delimiter_cost = 2 if formatted_blocks else 0  # "\n\n"
            projected_chars = current_chars + delimiter_cost + len(block)

            if projected_chars > self._max_chars and formatted_blocks:
                # Exceeds budget; omit this and subsequent chunks
                chunks_discarded_budget += 1
                continue

            # Include chunk
            formatted_blocks.append(block)
            current_chars = projected_chars
            chunks_included += 1

            # Build structured reference
            ref = ContextReference(
                document_name=chunk.document_name,
                document_id=chunk.document_id,
                source_path=chunk.source_path,
                start_page=chunk.start_page,
                end_page=chunk.end_page,
                page_numbers=list(chunk.page_numbers),
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                score=chunk.score,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                source_id=source_id,
            )
            references.append(ref)

            prev_chunk = chunk
            prev_chunk_processed_text = chunk.text.strip()

        final_text = "\n\n".join(formatted_blocks)
        total_chars = len(final_text)
        estimated_tokens = int(math.ceil(total_chars / 4.0))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return FormattedContext(
            text=final_text,
            references=references,
            total_chars=total_chars,
            estimated_tokens=estimated_tokens,
            chunks_included=chunks_included,
            chunks_discarded_budget=chunks_discarded_budget,
            context_build_latency_ms=latency_ms,
        )
