"""Generation data models, citation structures, and structured answer contracts."""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union
from documind.retrieval.schemas import FormattedContext


@dataclass(frozen=True)
class GenerationRequest:
    """Encapsulates prompt or messages payload, max tokens, and temperature."""
    prompt: Union[str, List[Dict[str, str]]]
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass(frozen=True)
class GenerationResult:
    """Raw output and hardware metrics from an LLM inference call."""
    text: str
    prompt_tokens: int
    completion_tokens: int
    generation_latency_ms: float
    tokens_per_sec: float


@dataclass(frozen=True)
class Citation:
    """Verifiable citation linking generated claims to physical document sources.
    
    Represents the 3-layer verification hierarchy:
    - Layer 1 (is_syntax_valid): Passed regex bracket and page syntax check.
    - Layer 2 (is_metadata_valid): Linked to an actual retrieved ContextReference.
    - Layer 3 (is_evidence_supported): Factual proposition supported by chunk text.
    """
    raw_text: str                               # e.g. "[Source: BASWE-15-RAG-Concepts-Guide.pdf, p. 3]"
    document_name: str                          # e.g. "BASWE-15-RAG-Concepts-Guide.pdf"
    start_page: int                             # e.g. 3
    end_page: int                               # e.g. 3
    matched_chunk_id: Optional[str] = None      # Linked chunk ID from ContextReference
    is_syntax_valid: bool = True                # Layer 1
    is_metadata_valid: bool = False             # Layer 2
    is_evidence_supported: bool = False         # Layer 3
    source_id: Optional[str] = None             # e.g. "S1", "S2"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GroundedAnswer:
    """Structured output contract separating answer text, citations, abstention, and telemetry."""
    query: str
    answer: str
    cited_source_ids: List[str]                 # Unique chunk IDs successfully cited
    citations: List[Citation]
    is_abstained: bool
    abstention_type: Optional[str]              # "fast_path_empty_context", "model_explicit", "fallback_unsupported", None
    grounding_status: str                       # "verified", "partially_supported", "unsupported", "abstained"
    context: FormattedContext
    telemetry: Dict[str, Any] = field(default_factory=dict)  # tokens, latencies, RAM delta

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "cited_source_ids": list(self.cited_source_ids),
            "citations": [c.to_dict() for c in self.citations],
            "is_abstained": self.is_abstained,
            "abstention_type": self.abstention_type,
            "grounding_status": self.grounding_status,
            "telemetry": dict(self.telemetry),
        }
