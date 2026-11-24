"""End-to-end grounded generation orchestrator with dual-tier abstention and citation verification."""

import re
import time
from typing import Any, Dict, List, Optional, Union

from documind.config import GenerationConfig
from documind.generation.citation import CitationParser
from documind.generation.llm import BaseLLM
from documind.generation.prompts import build_grounded_prompt
from documind.generation.schemas import Citation, GroundedAnswer
from documind.retrieval.context_builder import ContextBuilder
from documind.retrieval.retriever import BaseRetriever
from documind.retrieval.schemas import FormattedContext, RetrievalQuery


# Common model-level abstention indicators
ABSTENTION_INDICATORS = (
    "cannot answer",
    "can not answer",
    "does not contain",
    "do not contain",
    "not mentioned in the context",
    "not provided in the context",
    "no information",
    "no relevant information",
    "context does not mention",
    "context does not provide",
    "not enough information",
    "insufficient information",
)


def sanitize_answer_text(text: str) -> str:
    """Sanitize model output by stripping internal context headers, chunk IDs, and prompt formatting.
    
    Ensures that trailing regurgitated header lines such as:
    `### Source: 95cca1e4774a_FINZAA.pdf | Physical Page(s): pp. 2-7 | Ref ID: 95cca1e4774a_c0000_rec`
    are cleanly removed while preserving the user-facing answer and inline citation markers like `[S1]`.
    """
    if not text:
        return text

    # Strip any leaked context header blocks or lines starting with '### Source:'
    cleaned = re.sub(r"(?:\n|\s)*###\s*Source:[^\n]*(?:\n|$)?", "", text, flags=re.IGNORECASE)

    # Strip standalone 'Ref ID: ...' or '| Ref ID: ...'
    cleaned = re.sub(r"(?:\|\s*)?Ref ID:\s*[\w_]+", "", cleaned, flags=re.IGNORECASE)

    # Strip standalone 'Physical Page(s): ...'
    cleaned = re.sub(r"(?:\|\s*)?Physical Page\(s\):[^\n\|\]]+", "", cleaned, flags=re.IGNORECASE)

    # Strip standalone raw chunk id hashes (e.g. 95cca1e4774a_c0000_rec)
    cleaned = re.sub(r"\b[a-f0-9]{12}_c\d{4}_(?:rec|chunk)\b", "", cleaned, flags=re.IGNORECASE)

    # Strip any leaked internal document filename with 12-hex hash (e.g. ': 95cca1e4774a_FINZAA.pdf' or '95cca1e4774a_FINZAA.pdf')
    cleaned = re.sub(r"(?::\s*)?[a-f0-9]{12}_[A-Za-z0-9_.-]+", "", cleaned, flags=re.IGNORECASE)

    # Strip trailing pipes or dangling delimiters
    cleaned = re.sub(r"\s*\|\s*", " ", cleaned)

    # Clean up whitespace and collapse repeated spaces
    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()

    return cleaned


class GroundedGenerator:
    """Orchestrates document retrieval, context formatting, LLM generation, and citation verification.
    
    Guarantees:
    1. Dual-tier abstention:
       - Fast-path pre-generation check: Abstains instantly with 0 ms LLM inference when
         retrieval returns 0 candidate chunks or context builder yields empty context.
       - Model-level detection: Detects standardized abstention phrase or explicit variants.
    2. Conservative fallbacks:
       - When answers make factual claims with 0 valid citations or only hallucinated citations,
         the generator downgrades grounding_status to 'unsupported' and marks is_abstained=True.
    3. Three-layer citation verification:
       - Validates syntax, links against ContextReference metadata, and checks factual evidence.
    4. Structured output contract:
       - Emits immutable GroundedAnswer containing clean answer text, structured citations,
         source chunk IDs, abstention flags, and full disaggregated telemetry.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLLM,
        context_builder: Optional[ContextBuilder] = None,
        config: Optional[GenerationConfig] = None,
        citation_parser: Optional[CitationParser] = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._config = config or GenerationConfig()
        self._context_builder = context_builder or ContextBuilder(max_chars=self._config.max_context_chars)
        self._citation_parser = citation_parser or CitationParser()

    @property
    def retriever(self) -> BaseRetriever:
        return self._retriever

    @property
    def llm(self) -> BaseLLM:
        return self._llm

    @property
    def context_builder(self) -> ContextBuilder:
        return self._context_builder

    @property
    def config(self) -> GenerationConfig:
        return self._config

    @property
    def citation_parser(self) -> CitationParser:
        return self._citation_parser

    def _is_model_abstention(self, text: str) -> bool:
        """Check if model output represents an explicit refusal or abstention."""
        lower = text.lower().strip()
        if self._config.abstention_phrase.lower() in lower:
            return True
        return any(ind in lower for ind in ABSTENTION_INDICATORS)

    def generate_answer(
        self,
        query: Union[str, RetrievalQuery],
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        document_id: Optional[str] = None,
    ) -> GroundedAnswer:
        """Execute end-to-end grounded generation for a natural-language query."""
        t0 = time.perf_counter()

        # 1. Validate query input
        if isinstance(query, str):
            query_text = query.strip()
        elif isinstance(query, RetrievalQuery):
            query_text = query.text.strip()
        else:
            raise TypeError("query must be str or RetrievalQuery")

        if not query_text:
            raise ValueError("Query cannot be empty or whitespace-only")

        # 2. Dense Semantic Retrieval
        try:
            retrieval_res = self._retriever.retrieve(
                query,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                document_id=document_id,
            )
        except TypeError:
            retrieval_res = self._retriever.retrieve(
                query,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )

        # 3. Fast-Path Pre-Generation Abstention Check (0 candidate chunks)
        if not retrieval_res.chunks:
            empty_ctx = FormattedContext(
                text="",
                references=[],
                total_chars=0,
                estimated_tokens=0,
                chunks_included=0,
                chunks_discarded_budget=0,
                context_build_latency_ms=0.0,
            )
            total_latency = (time.perf_counter() - t0) * 1000.0
            telemetry: Dict[str, Any] = {
                "retrieval_latency_ms": retrieval_res.total_latency_ms,
                "context_build_latency_ms": 0.0,
                "generation_latency_ms": 0.0,
                "total_latency_ms": total_latency,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "tokens_per_sec": 0.0,
                "fast_path": True,
            }
            return GroundedAnswer(
                query=query_text,
                answer=self._config.abstention_phrase,
                cited_source_ids=[],
                citations=[],
                is_abstained=True,
                abstention_type="fast_path_empty_context",
                grounding_status="abstained",
                context=empty_ctx,
                telemetry=telemetry,
            )

        # 4. Context Window Assembly
        ctx = self._context_builder.build(retrieval_res.chunks)
        if ctx.chunks_included == 0 or not ctx.text.strip():
            total_latency = (time.perf_counter() - t0) * 1000.0
            telemetry = {
                "retrieval_latency_ms": retrieval_res.total_latency_ms,
                "context_build_latency_ms": ctx.context_build_latency_ms,
                "generation_latency_ms": 0.0,
                "total_latency_ms": total_latency,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "tokens_per_sec": 0.0,
                "fast_path": True,
            }
            return GroundedAnswer(
                query=query_text,
                answer=self._config.abstention_phrase,
                cited_source_ids=[],
                citations=[],
                is_abstained=True,
                abstention_type="fast_path_empty_context",
                grounding_status="abstained",
                context=ctx,
                telemetry=telemetry,
            )

        # 5. Prompt Assembly with Context Encapsulation
        messages = build_grounded_prompt(
            query_text, ctx, include_exemplar=self._config.include_exemplar
        )

        # 6. Synchronous LLM Generation
        gen_res = self._llm.generate(
            messages,
            max_new_tokens=self._config.max_new_tokens,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
        )
        raw_answer = gen_res.text.strip()

        return self.finalize_answer(
            query_text=query_text,
            raw_answer=raw_answer,
            retrieval_res=retrieval_res,
            ctx=ctx,
            prompt_tokens=gen_res.prompt_tokens,
            completion_tokens=gen_res.completion_tokens,
            gen_latency_ms=gen_res.generation_latency_ms,
            tokens_per_sec=gen_res.tokens_per_sec,
            t0=t0,
        )

    def finalize_answer(
        self,
        query_text: str,
        raw_answer: str,
        retrieval_res: Any,
        ctx: FormattedContext,
        prompt_tokens: int,
        completion_tokens: int,
        gen_latency_ms: float,
        tokens_per_sec: float,
        t0: float,
    ) -> GroundedAnswer:
        """Perform model abstention detection, 3-layer citation verification, and answer assembly."""
        # 7. Model-Level Abstention Detection
        if self._is_model_abstention(raw_answer):
            total_latency = (time.perf_counter() - t0) * 1000.0
            telemetry = {
                "retrieval_latency_ms": retrieval_res.total_latency_ms,
                "context_build_latency_ms": ctx.context_build_latency_ms,
                "generation_latency_ms": gen_latency_ms,
                "total_latency_ms": total_latency,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "tokens_per_sec": tokens_per_sec,
                "fast_path": False,
            }
            return GroundedAnswer(
                query=query_text,
                answer=raw_answer,
                cited_source_ids=[],
                citations=[],
                is_abstained=True,
                abstention_type="model_explicit",
                grounding_status="abstained",
                context=ctx,
                telemetry=telemetry,
            )

        # 8. Sanitize model answer & perform 3-Layer Citation Extraction & Verification
        clean_answer = sanitize_answer_text(raw_answer)
        citations = self._citation_parser.extract_citations(clean_answer)
        chunk_text_map = {c.chunk_id: c.text for c in retrieval_res.chunks}
        verified_cits, hallucinated_cits = self._citation_parser.verify_citations(
            citations=citations,
            references=ctx.references,
            answer_text=clean_answer,
            chunk_text_map=chunk_text_map,
        )

        valid_cits = [c for c in verified_cits if c.is_metadata_valid]
        cited_source_ids = sorted(
            list({c.matched_chunk_id for c in valid_cits if c.matched_chunk_id})
        )

        # 9. Conservative Fallbacks and Grounding Status
        if not citations:
            # Model generated factual assertions with zero citations
            is_abstained = True
            abstention_type = "fallback_unsupported"
            grounding_status = "unsupported"
        elif not valid_cits:
            # All citations are hallucinated or unverified
            is_abstained = True
            abstention_type = "fallback_unsupported"
            grounding_status = "unsupported"
        elif all(not c.is_evidence_supported for c in valid_cits):
            # All citations failed evidence support check
            is_abstained = True
            abstention_type = "fallback_unsupported"
            grounding_status = "unsupported"
        elif len(valid_cits) < len(citations) or any(
            not c.is_evidence_supported for c in valid_cits
        ):
            # Partial support: some valid citations, some hallucinated or unsupported
            is_abstained = False
            abstention_type = None
            grounding_status = "partially_supported"
        else:
            # Full support: 100% of citations valid and evidence-supported
            is_abstained = False
            abstention_type = None
            grounding_status = "verified"

        total_latency = (time.perf_counter() - t0) * 1000.0
        telemetry = {
            "retrieval_latency_ms": retrieval_res.total_latency_ms,
            "context_build_latency_ms": ctx.context_build_latency_ms,
            "generation_latency_ms": gen_latency_ms,
            "total_latency_ms": total_latency,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tokens_per_sec": tokens_per_sec,
            "hallucinated_citations": hallucinated_cits,
            "fast_path": False,
        }

        return GroundedAnswer(
            query=query_text,
            answer=clean_answer,
            cited_source_ids=cited_source_ids,
            citations=verified_cits,
            is_abstained=is_abstained,
            abstention_type=abstention_type,
            grounding_status=grounding_status,
            context=ctx,
            telemetry=telemetry,
        )
