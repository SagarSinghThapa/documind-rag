"""Automated unit tests for document chunking strategies and metadata tracking."""

import pytest
from documind.ingestion.schemas import DocumentPage, TextChunk
from documind.ingestion.chunker import (
    FixedSizeChunker,
    RecursiveCharacterChunker,
    create_chunker,
)


def make_test_pages(
    texts: list[str],
    doc_id: str = "d41d8cd98f00b204e9800998ecf8427e0123456789abcdef0123456789abcdef",
    doc_name: str = "sample_document.pdf",
) -> list[DocumentPage]:
    """Helper to create dummy DocumentPage records for testing."""
    pages = []
    for i, t in enumerate(texts):
        pages.append(
            DocumentPage(
                document_id=doc_id,
                document_name=doc_name,
                source_path=f"/dummy/{doc_name}",
                page_number=i + 1,
                text=t,
            )
        )
    return pages


class TestChunkerValidation:
    """Tests for chunking parameter validation across all chunker classes."""

    def test_invalid_chunk_size_non_positive(self):
        """Zero or negative chunk size must raise ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            FixedSizeChunker(chunk_size=0, chunk_overlap=0)

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            RecursiveCharacterChunker(chunk_size=-100, chunk_overlap=0)

    def test_invalid_chunk_overlap_negative(self):
        """Negative overlap must raise ValueError."""
        with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
            FixedSizeChunker(chunk_size=500, chunk_overlap=-10)

        with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
            RecursiveCharacterChunker(chunk_size=500, chunk_overlap=-1)

    def test_invalid_chunk_overlap_greater_or_equal_to_size(self):
        """Overlap >= size must raise ValueError."""
        with pytest.raises(ValueError, match="strictly less than chunk_size"):
            FixedSizeChunker(chunk_size=500, chunk_overlap=500)

        with pytest.raises(ValueError, match="strictly less than chunk_size"):
            RecursiveCharacterChunker(chunk_size=500, chunk_overlap=600)

    def test_factory_invalid_strategy(self):
        """Invalid strategy name in factory must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            create_chunker(strategy="unsupported_quantum_chunker")


class TestFixedSizeChunker:
    """Unit tests for fixed-size character chunking."""

    def test_text_shorter_than_chunk_size(self):
        """Short text produces exactly one chunk."""
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        pages = make_test_pages(["Short text content."])
        chunks = chunker.chunk_document(pages)

        assert len(chunks) == 1
        assert chunks[0].text == "Short text content."
        assert chunks[0].chunk_index == 0
        assert chunks[0].page_numbers == [1]
        assert chunks[0].start_page == 1
        assert chunks[0].end_page == 1
        assert chunks[0].chunking_strategy == "fixed"

    def test_text_exactly_at_chunk_size(self):
        """Text of exact chunk length produces one chunk."""
        text = "A" * 100
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        pages = make_test_pages([text])
        chunks = chunker.chunk_document(pages)

        assert len(chunks) == 1
        assert chunks[0].text == text
        assert len(chunks[0].text) == 100

    def test_text_longer_than_chunk_size_with_overlap(self):
        """Long text splits into multiple chunks with overlapping content."""
        # 250 characters, chunk_size=100, overlap=25 -> step=75
        # chunk 0: [0:100]
        # chunk 1: [75:175]
        # chunk 2: [150:250]
        text = "".join(f"Block{i:03d} " for i in range(28))  # 9 chars * 28 = 252 chars
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=25)
        pages = make_test_pages([text])
        chunks = chunker.chunk_document(pages)

        assert len(chunks) >= 3
        assert chunks[0].chunk_index == 0
        assert chunks[1].chunk_index == 1
        assert chunks[2].chunk_index == 2
        # Verify overlap between chunk 0 and 1
        overlap_tail = chunks[0].text[-25:]
        assert overlap_tail in chunks[1].text

    def test_zero_overlap(self):
        """Zero overlap chunks text contiguously without repeated boundary content."""
        text = "ABCDEFGHIJ" * 10  # 100 chars
        chunker = FixedSizeChunker(chunk_size=20, chunk_overlap=0)
        pages = make_test_pages([text])
        chunks = chunker.chunk_document(pages)

        assert len(chunks) == 5
        assert chunks[0].text == "ABCDEFGHIJABCDEFGHIJ"
        assert chunks[1].text == "ABCDEFGHIJABCDEFGHIJ"
        # Total text preserved
        reconstructed = "".join(c.text for c in chunks)
        assert reconstructed == text

    def test_empty_text_and_empty_pages(self):
        """Empty pages or empty text produce zero chunks."""
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        assert chunker.chunk_document([]) == []
        assert chunker.chunk_document(make_test_pages(["", "   "])) == []


class TestRecursiveCharacterChunker:
    """Unit tests for recursive delimiter-based chunking."""

    def test_recursive_splits_on_paragraph_boundaries(self):
        """Should split on paragraph boundaries (\n\n) before sentence or word boundaries."""
        p1 = "Paragraph 1 is about retrieval systems in RAG pipelines."
        p2 = "Paragraph 2 details embedding models and vector indexing."
        p3 = "Paragraph 3 discusses cross-encoder reranking algorithms."
        text = f"{p1}\n\n{p2}\n\n{p3}"

        chunker = RecursiveCharacterChunker(chunk_size=70, chunk_overlap=10)
        pages = make_test_pages([text])
        chunks = chunker.chunk_document(pages)

        assert len(chunks) == 3
        assert chunks[0].text == p1
        assert chunks[1].text == p2
        assert chunks[2].text == p3
        assert chunks[0].chunking_strategy == "recursive"

    def test_recursive_long_unbroken_paragraph(self):
        """Long paragraph exceeding chunk_size falls back gracefully to sentence, space, or character splits."""
        long_paragraph = "Sentence one is clear. " * 20  # ~460 characters
        chunker = RecursiveCharacterChunker(chunk_size=150, chunk_overlap=30)
        pages = make_test_pages([long_paragraph])
        chunks = chunker.chunk_document(pages)

        assert len(chunks) > 1
        for chunk in chunks:
            # Strip can result in <= chunk_size
            assert len(chunk.text) <= 150


class TestCrossPageAndMetadataTracking:
    """Tests for multi-page chunk attribution, metadata preservation, and ID stability."""

    def test_multi_page_chunk_attribution(self):
        """A chunk spanning across page boundaries must list all contributing page numbers."""
        page1_text = "Page 1 intro text that continues right up to the end."
        page2_text = "Page 2 concluding text that completes the thought."
        pages = make_test_pages([page1_text, page2_text])

        # Large chunk size so it encompasses both pages
        chunker = FixedSizeChunker(chunk_size=300, chunk_overlap=50)
        chunks = chunker.chunk_document(pages)

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.page_numbers == [1, 2]
        assert chunk.start_page == 1
        assert chunk.end_page == 2
        assert chunk.metadata["is_multipage"] is True

    def test_deterministic_chunk_ids(self):
        """Chunk IDs must remain identical across repeated runs with identical document and parameters."""
        pages = make_test_pages(["First page content.", "Second page content."])
        chunker_a = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=10)
        chunker_b = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=10)

        chunks_a = chunker_a.chunk_document(pages)
        chunks_b = chunker_b.chunk_document(pages)

        assert len(chunks_a) == len(chunks_b)
        for ca, cb in zip(chunks_a, chunks_b):
            assert ca.chunk_id == cb.chunk_id
            assert ca.text == cb.text
            assert ca.chunk_index == cb.chunk_index

    def test_chunk_metadata_completeness(self):
        """Every chunk must contain all required schema fields."""
        pages = make_test_pages(["Metadata completeness validation test."])
        chunker = create_chunker(strategy="recursive", chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk_document(pages)

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.chunk_id is not None
        assert chunk.document_id is not None
        assert chunk.document_name == "sample_document.pdf"
        assert chunk.source_path == "/dummy/sample_document.pdf"
        assert chunk.chunk_index == 0
        assert chunk.text == "Metadata completeness validation test."
        assert chunk.page_numbers == [1]
        assert chunk.start_page == 1
        assert chunk.end_page == 1
        assert chunk.chunking_strategy == "recursive"
        assert chunk.chunk_size == 100
        assert chunk.chunk_overlap == 20
        assert isinstance(chunk.to_dict(), dict)

    def test_char_offsets_exact_substring_match(self):
        """Regression test: char_start and char_end must precisely match chunk.text in unified text."""
        p1 = "Paragraph 1 with some initial introductory text."
        p2 = "Paragraph 2 with subsequent detailed explanations and technical data."
        p3 = "Paragraph 3 concluding the sample document with key findings."
        pages = make_test_pages([p1, p2, p3])

        # Construct unified text as chunk_document does
        unified_text = f"{p1}\n\n{p2}\n\n{p3}"

        for strat in ["fixed", "recursive"]:
            chunker = create_chunker(strategy=strat, chunk_size=80, chunk_overlap=25)
            chunks = chunker.chunk_document(pages)
            assert len(chunks) > 1
            for c in chunks:
                sliced = unified_text[c.char_start:c.char_end]
                assert sliced == c.text, f"Offset mismatch in {strat}: '{sliced}' != '{c.text}'"

    def test_recursive_chunk_size_strict_upper_bound(self):
        """Regression test: recursive chunker must never produce chunks exceeding chunk_size."""
        # Carefully crafted uneven sentences where overlap + next split previously exceeded chunk_size
        text = "Short. " + ("Medium sentence of arbitrary text. " * 5) + "\n\n" + ("Longer concluding sentence. " * 4)
        pages = make_test_pages([text])

        chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=35)
        chunks = chunker.chunk_document(pages)

        assert len(chunks) > 1
        for i, c in enumerate(chunks):
            assert len(c.text) <= 100, f"Chunk {i} exceeded chunk_size: {len(c.text)} > 100"
