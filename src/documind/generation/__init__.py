"""Grounded text generation subsystem for DocuMind."""

from documind.generation.citation import CITATION_REGEX, CitationParser
from documind.generation.evaluator import GroundingEvaluator, GroundingMetrics
from documind.generation.generator import ABSTENTION_INDICATORS, GroundedGenerator
from documind.generation.llm import (
    BaseLLM,
    LLMError,
    MockLLM,
    ModelNotFoundError,
    TransformersLLM,
)
from documind.generation.prompts import SYSTEM_PROMPT, build_grounded_prompt
from documind.generation.schemas import (
    Citation,
    GenerationRequest,
    GenerationResult,
    GroundedAnswer,
)

__all__ = [
    # Interfaces and Models
    "BaseLLM",
    "TransformersLLM",
    "MockLLM",
    "LLMError",
    "ModelNotFoundError",
    # Data Contracts & Schemas
    "Citation",
    "GenerationRequest",
    "GenerationResult",
    "GroundedAnswer",
    # Prompt Engine
    "SYSTEM_PROMPT",
    "build_grounded_prompt",
    # Citation & Verification
    "CitationParser",
    "CITATION_REGEX",
    # Generation Orchestrator
    "GroundedGenerator",
    "ABSTENTION_INDICATORS",
    # Evaluation
    "GroundingEvaluator",
    "GroundingMetrics",
]
