# Phase 4: WP-05 Local LLM and Grounded Generation - Research & Architecture Analysis

**Auditor/Architect:** Senior Local LLM Engineer, RAG Architect & AI Evaluation Specialist  
**Date:** 2026-11-23  
**Target:** WP-05 Grounded Generation Subsystem (Refined with 15 Senior Production Safeguards)

---

## 1. Local CPU-Compatible LLM Runtime & Baseline Model Selection

DocuMind operates under a strict ₹0 cloud API budget, 100% offline local privacy boundary, and Windows CPU execution without mandatory dedicated GPUs.

### Runtime Architecture Analysis

| Runtime Engine | Windows CPU Compatibility | C++ Compiler Needed? | Memory Efficiency | Integration Complexity | Verdict |
|:---|:---:|:---:|:---:|:---:|:---|
| **Hugging Face `transformers` + PyTorch CPU** | **Native Windows amd64** | **No** (pure pre-built wheels) | High (supports bfloat16/float32, 1-3 GB RAM) | Lowest (Already installed in `.venv`, 100% Python native) | **Primary Recommended Engine** |
| **`llama-cpp-python` (GGUF)** | Moderate | Yes (often requires MSVC or external pre-built wheels) | Highest (4-bit quantized ~1 GB RAM) | Medium (C++ DLL dependencies on Windows) | Pluggable secondary adapter |
| **Ollama Local Server** | Good | No (external standalone installer) | Good | Low (HTTP requests to `localhost:11434`) | Pluggable optional adapter |

### Model Selection Evaluation (Instruction-Tuned Small LLMs)

| Model Candidate | Parameter Count | Disk / RAM Footprint | CPU Generation Speed (x86_64) | Licensing | Context Window | Grounding & Instruction Following |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **`Qwen/Qwen2.5-0.5B-Instruct`** | 0.49B | ~1.0 GB RAM | **~30–45 tokens/sec** | Apache 2.0 | 32k tokens | **Initial Lightweight CPU Baseline** (Fast, low RAM) |
| **`Qwen/Qwen2.5-1.5B-Instruct`** | 1.54B | ~3.1 GB RAM | **~15–22 tokens/sec** | Apache 2.0 | 32k tokens | Superior reasoning and citation placement |
| **`HuggingFaceTB/SmolLM2-1.7B-Instruct`** | 1.71B | ~3.4 GB RAM | ~14–20 tokens/sec | Apache 2.0 | 8k tokens | Solid instruction adherence |
| **`meta-llama/Llama-3.2-1B-Instruct`** | 1.23B | ~2.5 GB RAM | ~18–25 tokens/sec | Llama 3.2 Community (Gated) | 128k tokens | Strong quality, but requires Hugging Face authentication token |

**Architecture Decision:**
- `Qwen/Qwen2.5-0.5B-Instruct` is chosen as the **initial lightweight CPU baseline**, not a proven final production model. It guarantees fast, reproducible verification and minimal memory overhead during development while the architecture remains pluggable for `Qwen2.5-1.5B-Instruct` or GGUF backends.
- Abstract base class `BaseLLM` isolates the application from the model engine, accompanied by `MockLLM` for fast, offline unit testing.

---

## 2. Three Distinct Layers of Citation Integrity

DocuMind enforces a rigorous 3-layer citation verification hierarchy:

```
Generated Answer Text
         │
         ▼
[Layer 1: Citation Syntax Validity]
  - Regex parser validates format: [Source: <doc>, p. <page>] or [Source: <doc>, pp. <start>-<end>]
  - Rejects malformed brackets, missing page numbers, or unparseable source strings.
         │ Valid Syntax
         ▼
[Layer 2: Citation Metadata Validity]
  - Resolves cited filename and page span against FormattedContext.references.
  - Matches exact document_name and verifies cited page is within [start_page, end_page].
  - Flags hallucinated citations if document or page was not in the retrieved context.
         │ Valid Reference Linkage
         ▼
[Layer 3: Actual Evidence Support]
  - Evaluates whether the specific chunk referenced actually contains the factual claim.
  - Conservative Fallback: Marks grounding_status ("verified", "partially_supported", "unsupported").
  - Downgrades to abstention if 0 claims are supported.
```

> **Crucial Engineering Safeguard:** Regex matching establishes citation syntax and reference linkage, **NOT semantic factual faithfulness**. Evaluating semantic support requires matching factual propositions against the cited chunk excerpt.

---

## 3. Grounding Contract & Prompt Engineering

To prevent hallucinations, prompt templates must enforce four non-negotiable rules:
1. **Strict Context Bounding:** The model is prohibited from using pre-training knowledge to answer factual questions.
2. **Untrusted Input Isolation:** Context text is wrapped in defensive markdown blockquotes (`> [Document Content - Untrusted Context]`) to prevent prompt injection.
3. **Mandatory Inline Citations:** Every factual sentence must be attributed using standard syntax:
   `[Source: <filename>, p. <page>]` or `[Source: <filename>, pp. <start>-<end>]`.
4. **Mandatory Abstention Directive:** If the context is empty or does not provide conclusive evidence, the model must output the exact standardized phrase:
   `"I cannot answer this question based on the provided document context."`

---

## 4. Dual-Tier Abstention Architecture

Abstention occurs at two levels:

```
User Query
    │
    ▼
SemanticRetriever.retrieve()
    │
    ├── 0 Chunks Retrieved / All Below Threshold? ──────► [Fast-Path Abstention]
    │                                                     - Instant response (0 ms LLM latency)
    │                                                     - Zero CPU tokens wasted
    ▼
ContextBuilder.build()
    │
    ├── Context Empty or Over-Budget Discarded? ────────► [Fast-Path Abstention]
    ▼
LLM Inference (temperature=0.0)
    │
    ├── Answer contains Abstention Phrase / Indicators? ─► [Model-Level Abstention]
    │                                                     - is_abstained = True
    ▼
Citation & Evidence Verification
    │
    ├── Zero claims supported by context? ──────────────► [Fallback Abstention]
    │                                                     - is_abstained = True
    ▼
GroundedAnswer (answer, citations, is_abstained=False, grounding_status="verified")
```

---

## 5. Formal Evaluation Methodology & Metrics

Evaluation separates positive factual queries ($Q_+$) from negative distractor queries ($Q_-$) across separate calibration and holdout splits:

### Evaluation Metrics
1. **Answer Correctness ($Q_+$):** Percentage of positive queries where the answer accurately addresses the question based on ground truth.
2. **Evidence Support ($Q_+$):** Percentage of factual claims directly supported by retrieved context text.
3. **Citation Validity ($Q_+$):** Percentage of generated inline citations that are syntactically valid and link to an actual retrieved chunk in `ContextReference`.
4. **Citation Support Accuracy ($Q_+$):** Percentage of valid citations where the cited chunk text actually supports the cited sentence.
5. **Abstention Precision & Recall ($Q_-$):**
   - **Abstention Precision:** $\frac{\text{True Abstentions on Negatives}}{\text{Total Abstentions Triggered}}$
   - **Abstention Recall (Negative Rejection Rate):** $\frac{\text{True Abstentions on Negatives}}{|Q_-|}$ (Target: $\ge 87.5\%$)
6. **Unsupported-Answer Rate:** Percentage of answers containing ungrounded or hallucinated claims (Target: $< 10\%$).

### Disclosed Dataset Limitations
- Partitioning:
  - **Split A (Calibration - 10 queries):** Used for prompt iteration, temperature tuning, and threshold selection.
  - **Split B (Holdout Evaluation - 10 queries):** Used once for holdout evaluation.
- *Limitation:* The $N=10$ query split provides an indicative local benchmark for functional validation; comprehensive statistical validation over 100+ queries is scheduled for Phase 6 (WP-07).

---

## 6. CPU Hardware Profiling & Decoupled Verification

Hardware performance is measured using `scripts/evaluate_generation.py`:
- **Cold Model Load Time (ms):** Initialization and weights mapping into memory.
- **Time to First Token (TTFT, ms):** Prompt encoding and first token generation.
- **Generation Throughput (tokens/sec):** Tokens generated per second on CPU.
- **Peak RAM Footprint (MB):** RSS memory delta measured via `psutil`.
- **Total Answer Latency (ms):** End-to-end pipeline time (retrieval + prompt + LLM).

*Decoupling:* Brittle hardware latency or RAM targets are logged as benchmark metrics rather than hardcoded unit-test failure conditions, ensuring robust execution across varying developer hardware.
