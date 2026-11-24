"""Unit tests for 3-layer citation parsing, syntax validation, and metadata verification."""

import pytest
from documind.generation.citation import CitationParser
from documind.generation.schemas import Citation
from documind.retrieval.schemas import ContextReference


@pytest.fixture
def parser() -> CitationParser:
    return CitationParser()


@pytest.fixture
def sample_reference() -> ContextReference:
    return ContextReference(
        document_name="BASWE-15-RAG-Concepts-Guide.pdf",
        document_id="doc_rag_01",
        source_path="/data/docs/BASWE-15-RAG-Concepts-Guide.pdf",
        start_page=2,
        end_page=3,
        page_numbers=[2, 3],
        chunk_id="chunk_overlap_01",
        chunk_index=1,
        score=0.85,
        char_start=0,
        char_end=500,
    )


class TestCitationParserUnit:
    """Test suite for CitationParser syntax extraction and metadata verification."""

    def test_citation_parser_single_page(self, parser: CitationParser):
        """Validates single page bracket citation syntax."""
        text = "Chunk overlap repeats text at chunk boundaries [Source: BASWE-15-RAG-Concepts-Guide.pdf, p. 3]."
        citations = parser.extract_citations(text)
        assert len(citations) == 1
        cit = citations[0]
        assert cit.raw_text == "[Source: BASWE-15-RAG-Concepts-Guide.pdf, p. 3]"
        assert cit.document_name == "BASWE-15-RAG-Concepts-Guide.pdf"
        assert cit.start_page == 3
        assert cit.end_page == 3
        assert cit.is_syntax_valid is True
        assert cit.is_metadata_valid is False  # Extraction only checks syntax

    def test_citation_parser_page_range(self, parser: CitationParser):
        """Validates multi-page range bracket citation syntax."""
        text = "Overlap spans multiple sections [Source: BASWE-15-RAG-Concepts-Guide.pdf, pp. 2-3]."
        citations = parser.extract_citations(text)
        assert len(citations) == 1
        cit = citations[0]
        assert cit.start_page == 2
        assert cit.end_page == 3
        assert cit.is_syntax_valid is True

    def test_citation_parser_alternate_page_prefix(self, parser: CitationParser):
        """Validates variations such as 'p.3', 'page 4', and 'pp. 5'."""
        text = "Fact A [Source: doc.pdf, p.3]. Fact B [Source: doc.pdf, page 4]."
        citations = parser.extract_citations(text)
        assert len(citations) == 2
        assert citations[0].start_page == 3
        assert citations[1].start_page == 4

    def test_citation_verification_valid(
        self, parser: CitationParser, sample_reference: ContextReference
    ):
        """Validates that a correctly cited document and page span links to ContextReference."""
        text = "Chunk overlap is 10 to 20 percent [Source: BASWE-15-RAG-Concepts-Guide.pdf, p. 3]."
        citations = parser.extract_citations(text)
        chunk_map = {sample_reference.chunk_id: "Chunk overlap is usually 10 to 20 percent of chunk size."}

        verified, hallucinated = parser.verify_citations(
            citations=citations,
            references=[sample_reference],
            answer_text=text,
            chunk_text_map=chunk_map,
        )

        assert len(hallucinated) == 0
        assert len(verified) == 1
        cit = verified[0]
        assert cit.is_metadata_valid is True
        assert cit.is_evidence_supported is True
        assert cit.matched_chunk_id == sample_reference.chunk_id

    def test_citation_verification_hallucinated_page(
        self, parser: CitationParser, sample_reference: ContextReference
    ):
        """Flags citation as hallucinated when document exists but page is out of range."""
        text = "Some claim [Source: BASWE-15-RAG-Concepts-Guide.pdf, p. 99]."
        citations = parser.extract_citations(text)

        verified, hallucinated = parser.verify_citations(
            citations=citations,
            references=[sample_reference],
            answer_text=text,
        )

        assert len(hallucinated) == 1
        assert hallucinated[0] == "[Source: BASWE-15-RAG-Concepts-Guide.pdf, p. 99]"
        assert len(verified) == 1
        assert verified[0].is_metadata_valid is False
        assert verified[0].matched_chunk_id is None

    def test_citation_verification_hallucinated_document(
        self, parser: CitationParser, sample_reference: ContextReference
    ):
        """Flags citation as hallucinated when document name does not exist in context references."""
        text = "Invented claim [Source: Quantum-Computing-Cookbook.pdf, p. 2]."
        citations = parser.extract_citations(text)

        verified, hallucinated = parser.verify_citations(
            citations=citations,
            references=[sample_reference],
            answer_text=text,
        )

        assert len(hallucinated) == 1
        assert verified[0].is_metadata_valid is False
        assert verified[0].matched_chunk_id is None

    def test_citation_malformed_syntax_rejected(self, parser: CitationParser):
        """Malformed brackets, missing page numbers, or parenthesis citations are rejected."""
        invalid_texts = [
            "Missing page [Source: doc.pdf]",
            "Missing brackets Source: doc.pdf, p. 1",
            "Round parentheses (Source: doc.pdf, p. 1)",
            "Empty brackets []",
        ]
        for t in invalid_texts:
            citations = parser.extract_citations(t)
            assert len(citations) == 0

    def test_multiple_citations_mixed_validity(
        self, parser: CitationParser, sample_reference: ContextReference
    ):
        """Correctly separates valid and invalid citations in multi-citation answers."""
        text = (
            "Valid statement [Source: BASWE-15-RAG-Concepts-Guide.pdf, p. 2]. "
            "Hallucinated statement [Source: Ghost-Doc.pdf, p. 5]."
        )
        citations = parser.extract_citations(text)
        assert len(citations) == 2

        verified, hallucinated = parser.verify_citations(
            citations=citations,
            references=[sample_reference],
        )

        assert len(hallucinated) == 1
        assert hallucinated[0] == "[Source: Ghost-Doc.pdf, p. 5]"
        assert verified[0].is_metadata_valid is True
        assert verified[1].is_metadata_valid is False

    def test_citation_parser_source_id_format(self, parser: CitationParser):
        """Validates deterministic source identifier syntax [S1], [S2], [s3]."""
        text = "Fact one [S1]. Fact two [S2]. Fact three [s3]."
        citations = parser.extract_citations(text)
        assert len(citations) == 3
        assert citations[0].raw_text == "[S1]"
        assert citations[0].source_id == "S1"
        assert citations[0].is_syntax_valid is True
        assert citations[1].source_id == "S2"
        assert citations[2].source_id == "S3"

    def test_citation_source_id_metadata_resolution(self, parser: CitationParser):
        """Resolves [S1] source identifier against trusted ContextReference metadata."""
        ref = ContextReference(
            document_name="BASWE-15-RAG-Concepts-Guide.pdf",
            document_id="doc_rag_01",
            source_path="/data/docs/BASWE-15-RAG-Concepts-Guide.pdf",
            start_page=2,
            end_page=3,
            page_numbers=[2, 3],
            chunk_id="chunk_overlap_01",
            chunk_index=1,
            score=0.85,
            char_start=0,
            char_end=500,
            source_id="S1",
        )
        text = "Overlap is 10 to 20 percent [S1]."
        citations = parser.extract_citations(text)
        chunk_map = {"chunk_overlap_01": "The usual range for chunk overlap is 10 to 20 percent."}

        verified, hallucinated = parser.verify_citations(
            citations=citations,
            references=[ref],
            answer_text=text,
            chunk_text_map=chunk_map,
        )

        assert len(hallucinated) == 0
        assert len(verified) == 1
        cit = verified[0]
        assert cit.source_id == "S1"
        assert cit.document_name == "BASWE-15-RAG-Concepts-Guide.pdf"
        assert cit.start_page == 2
        assert cit.end_page == 3
        assert cit.matched_chunk_id == "chunk_overlap_01"
        assert cit.is_metadata_valid is True
        assert cit.is_evidence_supported is True

    def test_citation_source_id_unmapped_flags_hallucination(self, parser: CitationParser):
        """Flags cited source ID like [S99] as hallucinated when unmapped in retrieved context."""
        ref = ContextReference(
            document_name="BASWE-15-RAG-Concepts-Guide.pdf",
            document_id="doc_rag_01",
            source_path="/data/docs/BASWE-15-RAG-Concepts-Guide.pdf",
            start_page=2,
            end_page=3,
            page_numbers=[2, 3],
            chunk_id="chunk_overlap_01",
            chunk_index=1,
            score=0.85,
            char_start=0,
            char_end=500,
            source_id="S1",
        )
        text = "Invented assertion [S99]."
        citations = parser.extract_citations(text)

        verified, hallucinated = parser.verify_citations(
            citations=citations,
            references=[ref],
            answer_text=text,
        )

        assert len(hallucinated) == 1
        assert hallucinated[0] == "[S99]"
        assert verified[0].is_metadata_valid is False
        assert verified[0].matched_chunk_id is None

