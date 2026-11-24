# Phase 4: WP-05 Local LLM and Grounded Generation - Context

**Gathered:** 2026-11-23  
**Status:** Ready for execution (Refined with 15 Senior LLM & Evaluation Architecture Safeguards)  
**Source:** User Requirements, WP-02/WP-03/WP-04 System Architecture, Technical Research & Local-First Constraints

<domain>
## Phase Boundary

Phase 4 designs and implements the grounded text generation subsystem for DocuMind. It takes structured context blocks and citation metadata produced by Phase 3 (`SemanticRetriever` and `ContextBuilder`), prompts a local CPU-compatible LLM under strict grounding constraints, formats precise document and page-level citations, verifies citations against structured `ContextReference` records, applies conservative fallbacks for unsupported claims, and enforces robust abstention when retrieved evidence is absent or insufficient.

This phase is **generation only**. Streamlit web application development belongs to Phase 5 (WP-06) and comprehensive multi-PDF release evaluation belongs to Phase 6 (WP-07).

</domain>

<decisions>
## Implementation Decisions

### 1. Local CPU LLM Engine & Architecture (Safeguards 1, 2)
- **Initial Baseline Model:** `Qwen/Qwen2.5-0.5B-Instruct` is adopted strictly as the **initial lightweight CPU baseline**, not a proven final production model. It provides a fast, reproducible baseline (<1.2 GB RAM, ~30–45 tokens/sec on CPU) while the subsystem remains fully pluggable for larger instruction models (e.g. `Qwen/Qwen2.5-1.5B-Instruct`) or quantized GGUF backends.
- **Configurable Runtime:** Configured via `GenerationConfig` in `src/documind/config.py` with `local_files_only=True` and `device="cpu"` by default. Compatible with the existing Python 3.11 Windows environment (`transformers` + PyTorch CPU).
- **Pluggable Abstraction:**
  - `BaseLLM(ABC)` in `src/documind/generation/llm.py` with `generate()` and `stream_generate()`.
  - `TransformersLLM`: Offline local Hugging Face CausalLM runner (`local_files_only=True`).
  - `MockLLM`: Deterministic mock generator for unit tests and fast CI without heavy model downloads.
  - `OllamaLLM`: Optional adapter for external local Ollama daemons (`localhost:11434`).
- **Offline Safety:** Explicit model provisioning via `scripts/download_llm.py`. The runtime raises `ModelNotFoundError` if weights are missing locally and never triggers silent outbound downloads.

### 2. Three Distinct Layers of Citation Integrity (Safeguards 3, 4, 5)
Citations are validated through three explicit architectural layers:
1. **Layer 1 — Citation Syntax Validity:** Verifies inline citation strings match standard regex syntax:
   `[Source: <filename>, p. <page>]` or `[Source: <filename>, pp. <start>-<end>]`.
2. **Layer 2 — Citation Metadata Validity:** Verifies parsed citations against structured `ContextReference` records emitted by `ContextBuilder`. Ensures cited filenames and physical page numbers actually exist in the retrieved context pool (protects against model-hallucinated sources or out-of-range page numbers).
3. **Layer 3 — Evidence Support:** Evaluates whether cited chunks actually contain supporting factual propositions.
- **Conservative Fallback (Safeguard 5):** If citations are missing on factual claims, or reference invalid pages, `GroundedGenerator` marks `grounding_status="partially_supported"` or `"unsupported"`, and falls back to abstention if no claim is supported. Regex matching is explicitly documented as verifying **syntax and linkage, NOT semantic factual truth**.

### 3. Structured Output Contract (Safeguard 6)
`GroundedAnswer` separates all generation concerns cleanly:
- `query: str`
- `answer: str`
- `cited_source_ids: List[str]` (mapped chunk IDs)
- `citations: List[Citation]` (structured citation objects with validity and mapping flags)
- `is_abstained: bool`
- `abstention_type: Optional[str]` (`"fast_path_empty_context"`, `"model_explicit"`, `"fallback_unsupported"`, or `None`)
- `grounding_status: str` (`"verified"`, `"partially_supported"`, `"unsupported"`, `"abstained"`)
- `context: FormattedContext`
- `telemetry: Dict[str, Any]` (prompt_tokens, completion_tokens, cold_load_ms, time_to_first_token_ms, generation_ms, tokens_per_sec, peak_ram_mb, total_latency_ms)

### 4. Robust Abstention Architecture (Safeguards 7, 8)
- Exact abstention phrase matching is treated as a **supplementary signal**, not the sole detector.
- Abstention detection combines:
  1. *Fast-Path Pre-Generation:* Triggers instantly when 0 chunks or empty context are retrieved, skipping LLM invocation entirely (0 ms LLM latency).
  2. *Model-Level Semantic Detection:* Detects exact phrase matching and semantic variants indicating lack of context ("cannot answer", "does not contain", "no mention").
  3. *Unsupported Fallback:* Downgrades to abstention if the model generates claims but none are supported by context.
- **Measurable Targets (Safeguard 8):** Replaces unverified 100% promises with measurable benchmark targets:
  - Negative Query Abstention Rate target: $\ge 87.5\%$ (at least 7 of 8 negative distractors rejected).
  - Explicitly reports False Answer Rate and Abstention Errors on positive queries.

### 5. Evaluation Methodology & Disclosed Limitations (Safeguards 9, 10, 12, 13)
- **Dataset Partitioning:**
  - Split A (Calibration - 10 queries: 6 pos, 4 neg) is reserved for prompt development and threshold calibration.
  - Split B (Holdout Evaluation - 10 queries: 6 pos, 4 neg) is used for final unbiased reporting.
  - *Disclosed Limitation:* Reusing the Phase 3 holdout provides an indicative local benchmark ($N=10$ per split); full statistical validation across 100+ queries is scheduled for Phase 6.
- **Evaluation Procedures (Safeguard 10):**
  - Answer correctness (answer matches ground truth)
  - Evidence support (claims supported by context text)
  - Citation validity (syntax and metadata linkage)
  - Citation support accuracy (cited chunk supports claim)
  - Abstention precision and recall
  - Unsupported-answer rate
- **Actual CPU Benchmarks (Safeguard 12):** `scripts/evaluate_generation.py` profiles cold model loading, first-token latency (TTFT), generation throughput (tokens/sec), peak RAM delta via `psutil`, and total answer latency.
- **Test Separation (Safeguards 11, 13):** Unit tests strictly use `MockLLM`. Real-model integration tests are isolated under `@pytest.mark.integration` or conditional execution. No brittle hardware latency or RAM assertions are placed in unit tests.

### 6. Non-Negotiable Safeguards (Safeguards 14, 15)
- Zero regressions across existing 90 tests.
- 100% local CPU execution — strictly ₹0 budget, no paid cloud APIs.
- Offline runtime execution (`local_files_only=True`).
- Document content treated as untrusted text within prompt templates.
- Planning only; no production code changes or Git commits until execution is approved.

</decisions>

<canonical_refs>
## Canonical References

- `src/documind/retrieval/schemas.py` — `RetrievedChunk`, `RetrievalResult`, `ContextReference`, `FormattedContext`.
- `src/documind/retrieval/retriever.py` — `SemanticRetriever`.
- `src/documind/retrieval/context_builder.py` — `ContextBuilder`.
- `data/evaluation/retrieval_ground_truth.json` — 20 verified ground-truth queries.
- `docs/architecture/BASELINE_RAG_SPECIFICATION.md` — Stage 7 (LLM Generation) specification.
- `scripts/evaluate_retrieval.py` — Verified Phase 3 retrieval benchmarks.

</canonical_refs>
