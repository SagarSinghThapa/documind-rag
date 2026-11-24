"""Unit tests for GroundedGenerator, dual-tier abstention, and conservative fallbacks."""

from typing import List, Optional, Union
import pytest

from documind.config import GenerationConfig
from documind.generation.generator import GroundedGenerator, sanitize_answer_text
from documind.generation.llm import MockLLM
from documind.generation.prompts import build_grounded_prompt
from documind.retrieval.context_builder import ContextBuilder
from documind.retrieval.retriever import BaseRetriever
from documind.retrieval.schemas import (
    FormattedContext,
    RetrievalQuery,
    RetrievalResult,
    RetrievedChunk,
)


class StubRetriever(BaseRetriever):
    """Deterministic test retriever returning pre-configured chunks."""

    def __init__(self, chunks: Optional[List[RetrievedChunk]] = None) -> None:
        self.chunks = chunks or []

    def retrieve(
        self,
        query: Union[str, RetrievalQuery],
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> RetrievalResult:
        query_text = query if isinstance(query, str) else query.text
        if not query_text.strip():
            raise ValueError("Query cannot be empty")

        return RetrievalResult(
            query=query_text,
            chunks=list(self.chunks),
            total_retrieved=len(self.chunks),
            total_below_threshold=0,
            embed_latency_ms=1.5,
            search_latency_ms=0.5,
            total_latency_ms=2.0,
        )


@pytest.fixture
def sample_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk_test_01",
        document_id="doc_rag_01",
        document_name="BASWE-15-RAG-Concepts-Guide.pdf",
        source_path="/data/docs/BASWE-15-RAG-Concepts-Guide.pdf",
        chunk_index=1,
        text="Chunk overlap repeats text at chunk boundaries. The usual range is 10 to 20 percent.",
        page_numbers=[2, 3],
        start_page=2,
        end_page=3,
        char_start=0,
        char_end=84,
        score=0.88,
    )


class TestGroundedGeneratorUnit:
    """Test suite verifying GroundedGenerator contracts and safety guarantees."""

    def test_fast_path_abstention_empty_retrieval(self):
        """Pre-generation check abstains instantly when 0 chunks are retrieved."""
        retriever = StubRetriever(chunks=[])
        llm = MockLLM(response_text="Should never be called")
        generator = GroundedGenerator(retriever=retriever, llm=llm)

        answer = generator.generate_answer("What is the temperature of the sun?")
        assert answer.is_abstained is True
        assert answer.abstention_type == "fast_path_empty_context"
        assert answer.grounding_status == "abstained"
        assert answer.answer == generator.config.abstention_phrase
        assert answer.telemetry["fast_path"] is True
        assert answer.telemetry["generation_latency_ms"] == 0.0
        assert len(answer.citations) == 0

    def test_grounded_generation_with_mock_llm(self, sample_chunk: RetrievedChunk):
        """Full pipeline returns verified GroundedAnswer when citations are valid."""
        retriever = StubRetriever(chunks=[sample_chunk])
        canned_response = (
            "Chunk overlap repeats text at boundaries with a range of 10 to 20 percent "
            "[Source: BASWE-15-RAG-Concepts-Guide.pdf, p. 3]."
        )
        llm = MockLLM(response_text=canned_response)
        generator = GroundedGenerator(retriever=retriever, llm=llm)

        answer = generator.generate_answer("What is chunk overlap?")
        assert answer.is_abstained is False
        assert answer.grounding_status == "verified"
        assert answer.abstention_type is None
        assert len(answer.citations) == 1
        assert answer.citations[0].is_metadata_valid is True
        assert answer.citations[0].matched_chunk_id == sample_chunk.chunk_id
        assert sample_chunk.chunk_id in answer.cited_source_ids
        assert answer.telemetry["fast_path"] is False

    def test_grounded_generation_with_source_id_citations(self, sample_chunk: RetrievedChunk):
        """Full pipeline resolves [S1] source identifier citations and verifies GroundedAnswer."""
        retriever = StubRetriever(chunks=[sample_chunk])
        canned_response = (
            "Chunk overlap repeats text at chunk boundaries, usually 10 to 20 percent [S1]."
        )
        llm = MockLLM(response_text=canned_response)
        generator = GroundedGenerator(retriever=retriever, llm=llm)

        answer = generator.generate_answer("What is chunk overlap?")
        assert answer.is_abstained is False
        assert answer.grounding_status == "verified"
        assert answer.abstention_type is None
        assert len(answer.citations) == 1
        cit = answer.citations[0]
        assert cit.source_id == "S1"
        assert cit.is_metadata_valid is True
        assert cit.matched_chunk_id == sample_chunk.chunk_id
        assert cit.document_name == sample_chunk.document_name
        assert cit.start_page == sample_chunk.start_page
        assert cit.end_page == sample_chunk.end_page
        assert sample_chunk.chunk_id in answer.cited_source_ids
        assert cit.is_evidence_supported is True

    def test_model_level_abstention_detection(self, sample_chunk: RetrievedChunk):
        """Marks is_abstained=True when model emits standard abstention phrase."""
        retriever = StubRetriever(chunks=[sample_chunk])
        llm = MockLLM(
            response_text="I cannot answer this question based on the provided document context."
        )
        generator = GroundedGenerator(retriever=retriever, llm=llm)

        answer = generator.generate_answer("How to bake sourdough bread?")
        assert answer.is_abstained is True
        assert answer.abstention_type == "model_explicit"
        assert answer.grounding_status == "abstained"
        assert len(answer.cited_source_ids) == 0

    def test_conservative_fallback_on_unsupported_claim(self, sample_chunk: RetrievedChunk):
        """Flags grounding_status='unsupported' when model gives factual claim with zero citations."""
        retriever = StubRetriever(chunks=[sample_chunk])
        # Model generated text without citations
        llm = MockLLM(response_text="Chunk overlap is between 10 and 20 percent.")
        generator = GroundedGenerator(retriever=retriever, llm=llm)

        answer = generator.generate_answer("What is chunk overlap?")
        assert answer.is_abstained is True
        assert answer.abstention_type == "fallback_unsupported"
        assert answer.grounding_status == "unsupported"
        assert len(answer.citations) == 0

    def test_conservative_fallback_on_hallucinated_citation(self, sample_chunk: RetrievedChunk):
        """Downgrades to unsupported abstention when all citations are hallucinated."""
        retriever = StubRetriever(chunks=[sample_chunk])
        llm = MockLLM(
            response_text="Chunk overlap is vital [Source: Quantum-Mechanics-Manual.pdf, p. 99]."
        )
        generator = GroundedGenerator(retriever=retriever, llm=llm)

        answer = generator.generate_answer("What is chunk overlap?")
        assert answer.is_abstained is True
        assert answer.abstention_type == "fallback_unsupported"
        assert answer.grounding_status == "unsupported"
        assert len(answer.telemetry["hallucinated_citations"]) == 1

    def test_untrusted_text_not_leaked_into_system_role(self, sample_chunk: RetrievedChunk):
        """Ensures document text is isolated in the user role, keeping system instructions trusted."""
        ctx_builder = ContextBuilder()
        formatted_ctx = ctx_builder.build([sample_chunk])
        messages = build_grounded_prompt("What is chunk overlap?", formatted_ctx)

        assert len(messages) == 2
        system_msg = messages[0]
        user_msg = messages[1]

        assert system_msg["role"] == "system"
        # Untrusted chunk text must not appear in system prompt
        assert sample_chunk.text not in system_msg["content"]
        assert "RULES:" in system_msg["content"]

        assert user_msg["role"] == "user"
        # Untrusted chunk text must be isolated in user prompt
        assert sample_chunk.text in user_msg["content"]

    def test_empty_query_raises_value_error(self):
        """Whitespace or empty query string immediately raises ValueError."""
        generator = GroundedGenerator(retriever=StubRetriever(), llm=MockLLM())
        with pytest.raises(ValueError, match="Query cannot be empty"):
            generator.generate_answer("   ")

    def test_grounded_author_answer_verified(self):
        """Author query returns verified status when LLM cites chunk containing author name."""
        author_chunk = RetrievedChunk(
            chunk_id="chunk_author_01",
            document_id="doc_finzaa",
            document_name="FINZAA.pdf",
            source_path="data/FINZAA.pdf",
            chunk_index=0,
            text="Study on cowpea mutagen... By Anjali Singh Id. No.: 241162002...",
            page_numbers=[2, 3],
            start_page=2,
            end_page=3,
            char_start=0,
            char_end=65,
            score=0.90,
        )
        retriever = StubRetriever(chunks=[author_chunk])
        llm = MockLLM(response_text="Anjali Singh is the author of FINZAA.pdf [S1].")
        generator = GroundedGenerator(retriever=retriever, llm=llm)

        ans = generator.generate_answer("Who is the author of FINZAA.pdf?")
        assert ans.is_abstained is False
        assert ans.grounding_status == "verified"
        assert len(ans.citations) == 1
        assert ans.citations[0].is_evidence_supported is True
        assert ans.citations[0].matched_chunk_id == "chunk_author_01"
        assert ans.citations[0].start_page == 2
        assert ans.citations[0].end_page == 3

    def test_sanitize_answer_text_strips_context_headers(self):
        """Sanitizer cleanly removes leaked ### Source: headers and Ref IDs while preserving [S1]."""
        raw = (
            "Anjali Singh is the author of FINZAA.pdf [S1] ### Source: 95cca1e4774a_FINZAA.pdf | "
            "Physical Page(s): pp. 2-7 | Ref ID: 95cca1e4774a_c0000_rec."
        )
        cleaned = sanitize_answer_text(raw)
        assert "### Source:" not in cleaned
        assert "Ref ID:" not in cleaned
        assert "95cca1e4774a_c0000_rec" not in cleaned
        assert "[S1]" in cleaned
        assert "Anjali Singh" in cleaned

    def test_generator_cleans_regurgitated_headers_in_answer(self, sample_chunk: RetrievedChunk):
        """Generator strips context headers from raw answer text while preserving citations."""
        retriever = StubRetriever(chunks=[sample_chunk])
        canned = (
            "Chunk overlap is 10 to 20 percent [S1]. "
            "### Source: BASWE-15-RAG-Concepts-Guide.pdf | Physical Page(s): pp. 2-3 | Ref ID: chunk_test_01"
        )
        llm = MockLLM(response_text=canned)
        generator = GroundedGenerator(retriever=retriever, llm=llm)

        ans = generator.generate_answer("What is chunk overlap?")
        assert ans.is_abstained is False
        assert ans.grounding_status == "verified"
        assert "### Source:" not in ans.answer
        assert "Ref ID:" not in ans.answer
        assert "[S1]" in ans.answer
        assert ans.citations[0].is_metadata_valid is True

    def test_missing_author_metadata_abstains(self):
        """When author metadata is missing and LLM abstains, generator marks model_explicit abstention."""
        anonymous_chunk = RetrievedChunk(
            chunk_id="chunk_anon_01",
            document_id="doc_anon",
            document_name="anon.pdf",
            source_path="data/anon.pdf",
            chunk_index=0,
            text="This report contains technical figures only.",
            page_numbers=[1],
            start_page=1,
            end_page=1,
            char_start=0,
            char_end=45,
            score=0.90,
        )
        retriever = StubRetriever(chunks=[anonymous_chunk])
        llm = MockLLM(response_text="I cannot answer this question based on the provided document context.")
        generator = GroundedGenerator(retriever=retriever, llm=llm)

        ans = generator.generate_answer("Who is the author of anon.pdf?")
        assert ans.is_abstained is True
        assert ans.abstention_type == "model_explicit"
        assert ans.grounding_status == "abstained"

    def test_hallucinated_author_intercepted(self):
        """When LLM hallucinates an author not present in chunk text, citation evidence fails and answer is unsupported."""
        author_chunk = RetrievedChunk(
            chunk_id="chunk_author_01",
            document_id="doc_finzaa",
            document_name="FINZAA.pdf",
            source_path="data/FINZAA.pdf",
            chunk_index=0,
            text="Study on cowpea mutagen... By Anjali Singh Id. No.: 241162002...",
            page_numbers=[2],
            start_page=2,
            end_page=2,
            char_start=0,
            char_end=65,
            score=0.90,
        )
        retriever = StubRetriever(chunks=[author_chunk])
        # Model invents an author name not in chunk text
        llm = MockLLM(response_text="The author is Albert Einstein [S1].")
        generator = GroundedGenerator(retriever=retriever, llm=llm)

        ans = generator.generate_answer("Who wrote this paper?")
        assert ans.is_abstained is True
        assert ans.abstention_type == "fallback_unsupported"
        assert ans.grounding_status == "unsupported"
        assert len(ans.citations) == 1
        assert ans.citations[0].is_evidence_supported is False

    def test_document_id_parameter_forwarding(self):
        """generate_answer forwards explicit document_id parameter to retriever."""
        class TrackingRetriever(BaseRetriever):
            def __init__(self):
                self.received_doc_id = None
            def retrieve(self, query, top_k=None, similarity_threshold=None, document_id=None):
                self.received_doc_id = document_id
                return RetrievalResult(
                    query=str(query), chunks=[], total_retrieved=0, total_below_threshold=0,
                    embed_latency_ms=0, search_latency_ms=0, total_latency_ms=0
                )

        tracker = TrackingRetriever()
        gen = GroundedGenerator(retriever=tracker, llm=MockLLM())
        gen.generate_answer("test query", document_id="doc_target_123")
        assert tracker.received_doc_id == "doc_target_123"


class TestGroundingEvaluatorUnit:
    """Test suite for GroundingEvaluator metrics computation."""

    def test_evaluator_positive_and_negative_splits(self, tmp_path, sample_chunk: RetrievedChunk):
        """Validates grounding metrics calculation across positive and negative queries."""
        import json
        from documind.generation.evaluator import GroundingEvaluator

        eval_data = [
            {
                "query_id": "TEST-01",
                "split": "test_split",
                "query": "What is chunk overlap?",
                "is_negative": False,
                "expected_document": "BASWE-15-RAG-Concepts-Guide.pdf",
                "expected_pages": [2, 3],
                "verified_excerpt": "Chunk overlap repeats text at chunk boundaries.",
            },
            {
                "query_id": "TEST-02-NEG",
                "split": "test_split",
                "query": "How to tune an automotive carburetor?",
                "is_negative": True,
                "expected_document": None,
                "expected_pages": [],
                "verified_excerpt": None,
            },
        ]
        eval_file = tmp_path / "eval_data.json"
        eval_file.write_text(json.dumps(eval_data), encoding="utf-8")

        # Generator returns supported answer for Q1, and fast-paths abstention on Q2 (0 chunks)
        class CustomRetriever(BaseRetriever):
            def retrieve(self, query, top_k=None, similarity_threshold=None):
                q = query if isinstance(query, str) else query.text
                if "carburetor" in q.lower():
                    chunks = []
                else:
                    chunks = [sample_chunk]
                return RetrievalResult(
                    query=q,
                    chunks=chunks,
                    total_retrieved=len(chunks),
                    total_below_threshold=0,
                    embed_latency_ms=1.0,
                    search_latency_ms=1.0,
                    total_latency_ms=2.0,
                )

        canned = (
            "Chunk overlap repeats text at chunk boundaries "
            "[Source: BASWE-15-RAG-Concepts-Guide.pdf, p. 3]."
        )
        generator = GroundedGenerator(
            retriever=CustomRetriever(),
            llm=MockLLM(response_text=canned),
        )

        evaluator = GroundingEvaluator(generator=generator, eval_data_path=str(eval_file))
        metrics = evaluator.evaluate(split="test_split")

        assert metrics.total_queries == 2
        assert metrics.positive_queries == 1
        assert metrics.negative_queries == 1
        assert metrics.answer_correctness == 1.0
        assert metrics.evidence_support_rate == 1.0
        assert metrics.citation_validity_rate == 1.0
        assert metrics.negative_abstention_rate == 1.0
        assert metrics.unsupported_answer_rate == 0.0
        assert metrics.false_answer_rate == 0.0

