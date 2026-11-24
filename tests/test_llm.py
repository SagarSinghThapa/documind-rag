"""Automated unit tests for LLM abstraction, prompt formatting, and offline protection."""

import pytest
from typing import Dict, List

from documind.config import GenerationConfig
from documind.generation import (
    BaseLLM,
    Citation,
    GenerationResult,
    GroundedAnswer,
    MockLLM,
    ModelNotFoundError,
    TransformersLLM,
    build_grounded_prompt,
)
from documind.retrieval.schemas import ContextReference, FormattedContext


@pytest.fixture
def sample_context() -> FormattedContext:
    ref = ContextReference(
        document_name="sample_guide.pdf",
        document_id="doc_abc",
        source_path="data/sample_guide.pdf",
        start_page=3,
        end_page=3,
        page_numbers=[3],
        chunk_id="chk_01",
        chunk_index=0,
        score=0.88,
        char_start=0,
        char_end=150,
    )
    text = (
        "### Source: sample_guide.pdf | Physical Page(s): p. 3 | Ref ID: chk_01\n"
        "> [Document Content - Untrusted Context]\n"
        "> Chunking strategies partition large documents into manageable passages."
    )
    return FormattedContext(
        text=text,
        references=[ref],
        total_chars=len(text),
        estimated_tokens=40,
        chunks_included=1,
        chunks_discarded_budget=0,
        context_build_latency_ms=1.5,
    )


class TestLLMUnit:
    """Unit tests for prompt formatting, MockLLM, and offline model safeguards."""

    def test_prompt_construction(self, sample_context):
        """Prompt compiler produces structured messages with system instructions and untrusted context."""
        query = "What is chunking?"
        messages = build_grounded_prompt(query, sample_context)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "DocuMind" in messages[0]["content"]
        assert "RULES:" in messages[0]["content"]
        assert "[Source: <filename>, p. <page>]" in messages[0]["content"]

        assert messages[1]["role"] == "user"
        assert "Context:" in messages[1]["content"]
        assert "sample_guide.pdf" in messages[1]["content"]
        assert "Question: What is chunking?" in messages[1]["content"]

    def test_empty_context_prompt_formatting(self):
        """Prompt compiler safely formats empty context without crashing."""
        empty_ctx = FormattedContext(
            text="",
            references=[],
            total_chars=0,
            estimated_tokens=0,
            chunks_included=0,
            chunks_discarded_budget=0,
            context_build_latency_ms=0.0,
        )
        messages = build_grounded_prompt("What is RAG?", empty_ctx)

        assert len(messages) == 2
        assert "[No relevant document context found]" in messages[1]["content"]
        assert "Question: What is RAG?" in messages[1]["content"]

    def test_mock_llm_generation(self):
        """MockLLM generates deterministic responses and returns structured GenerationResult."""
        mock = MockLLM(
            responses={"chunking": "Chunking divides documents [Source: sample_guide.pdf, p. 3]."},
            default_response="Default answer [Source: default.pdf, p. 1].",
        )

        res = mock.generate("Explain chunking strategies")
        assert isinstance(res, GenerationResult)
        assert res.text == "Chunking divides documents [Source: sample_guide.pdf, p. 3]."
        assert res.completion_tokens > 0
        assert res.generation_latency_ms > 0.0

        # Unmatched query returns default response
        res_default = mock.generate("Random unrelated query")
        assert res_default.text == "Default answer [Source: default.pdf, p. 1]."

    def test_mock_llm_streaming(self):
        """MockLLM yields words sequentially during stream generation."""
        mock = MockLLM(default_response="One two three four five.")
        tokens = list(mock.stream_generate("Stream test"))

        assert len(tokens) >= 5
        assert "".join(tokens) == "One two three four five."

    def test_transformers_llm_offline_protection(self):
        """TransformersLLM raises ModelNotFoundError when non-existent model is loaded with offline flags."""
        cfg = GenerationConfig(
            model_name="non_existent/fake_model_for_test_123",
            local_files_only=True,
            allow_download=False,
        )
        llm = TransformersLLM(config=cfg)

        with pytest.raises(ModelNotFoundError) as exc_info:
            llm.generate("Hello world")

        assert "offline mode is enforced" in str(exc_info.value)
        assert "download_llm.py" in str(exc_info.value)

    def test_grounded_answer_schema_contract(self, sample_context):
        """GroundedAnswer enforces structured output contract and serializes cleanly to dict."""
        cit = Citation(
            raw_text="[Source: sample_guide.pdf, p. 3]",
            document_name="sample_guide.pdf",
            start_page=3,
            end_page=3,
            matched_chunk_id="chk_01",
            is_syntax_valid=True,
            is_metadata_valid=True,
            is_evidence_supported=True,
        )
        answer = GroundedAnswer(
            query="What is chunking?",
            answer="Chunking partitions documents [Source: sample_guide.pdf, p. 3].",
            cited_source_ids=["chk_01"],
            citations=[cit],
            is_abstained=False,
            abstention_type=None,
            grounding_status="verified",
            context=sample_context,
            telemetry={"prompt_tokens": 50, "completion_tokens": 12, "total_latency_ms": 25.0},
        )

        d = answer.to_dict()
        assert d["query"] == "What is chunking?"
        assert d["is_abstained"] is False
        assert d["grounding_status"] == "verified"
        assert len(d["citations"]) == 1
        assert d["citations"][0]["document_name"] == "sample_guide.pdf"
        assert d["citations"][0]["is_metadata_valid"] is True
        assert d["cited_source_ids"] == ["chk_01"]
