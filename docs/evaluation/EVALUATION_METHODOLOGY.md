# DocuMind 1.0 — Evaluation Methodology & Benchmark Specification

**Repository:** `X:\Projects\Documind`  
**Phase:** Phase 1 — Literature Review and Baseline  
**Document Status:** Authoritative Evaluation Protocol (WP-01 Deliverable 6)  
**Governance Standard:** Quantitative, reproducible evaluation framework; absolute isolation of private candidate resumes; clean public document corpus only; no fabricated results or benchmark questions from uninspected sources.  
**Hardware & Budget Context:** Intel Core i5-1135G7 (4C/8T, 2.4–4.2 GHz), 24 GB RAM; ₹0 financial budget; offline local CPU evaluation harness.

---

## 1. Evaluation Philosophy & Two-Tier Benchmark Architecture

To avoid qualitative guesswork, DocuMind 1.0 employs a rigorous, quantitative evaluation harness. The evaluation framework separates retrieval quality from generation quality to pinpoint pipeline bottlenecks.

### The Two-Tier Benchmark Hierarchy
1. **Tier 1: MVP Smoke-Test Subset (~20–25 Questions):**
   - *Purpose:* Rapid regression testing, fast smoke tests during active iteration, and continuous validation of baseline pipeline integrity on local CPU hardware.
   - *Execution Time:* $< 60$ seconds total runtime on CPU.
   - *Composition:* A balanced subset covering core factual retrieval, single-document lookups, and clear unanswerable queries.
2. **Tier 2: Full Project Benchmark (~50–100 Manually Verified Questions):**
   - *Purpose:* Definitive system validation, component ablation studies (chunk size, hybrid retrieval, cross-encoder reranking), embedding model comparisons, and final release qualification for DocuMind 1.0.
   - *Composition:* Comprehensive, multi-dimensional query distribution spanning all 7 canonical query types with passage-level attribution.

---

## 2. Evaluation Corpus Selection & Privacy Governance

### 2.1 Clean Public Document Corpus Selection Criteria
To guarantee reproducibility while strictly complying with privacy and copyright constraints:
1. **Strict Segregation of Private Data:** The 25 private candidate resume PDFs in `resumes/` are **strictly prohibited** from inclusion in the public evaluation benchmark. They must never be indexed in evaluation suites, used in prompt examples, or transmitted to any external service.
2. **Public Technical Document Requirements:** The evaluation corpus must consist of publicly available, open-license technical documentation, research papers, or government reports exhibiting:
   - Varied document lengths (short 2-page briefs, medium 15-page reports, long 50+ page manuals).
   - Diverse structural layouts (single-column prose, two-column academic formats, numbered headers, lists, embedded tables).
   - Rich technical vocabulary containing both conceptual prose and exact alphanumeric identifiers (e.g. model numbers, version strings, dates, quantitative metrics).

### 2.2 Recommended Public Corpus Composition
- **Doc Set A (Academic / Architecture):** Open-access computer science papers (e.g., Lewis et al. 2020 RAG, Barnett et al. 2024 failure points).
- **Doc Set B (Technical Specifications / Manuals):** Open-source software documentation (e.g., Python library specifications, SQLite technical manuals).
- **Doc Set C (Reports / Tables):** Public statistical or organizational reports containing numerical data and structured tables.

---

## 3. Seven Canonical Question Categories & Distribution

The full 50–100 question benchmark must be stratified across seven distinct query categories to evaluate all failure modes documented by Barnett et al. (2024):

| # | Question Category | Target % (50–100 Benchmark) | Target Count (MVP Subset) | Primary Pipeline Failure Point Addressed | Concrete Query Example |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **1** | **Single-Document Factual** | 20% (10–20 Qs) | 5 Qs | FP2 (Missed Top Ranked): Fact exists in a single isolated chunk. | *"What was the exact Top-20 retrieval accuracy of DPR on Natural Questions?"* |
| **2** | **Multi-Chunk Reasoning** | 20% (10–20 Qs) | 4 Qs | FP7 (Incomplete Answer): Information is split across non-contiguous chunks in the same document. | *"How does the RAG training objective differ between RAG-Sequence and RAG-Token models?"* |
| **3** | **Multi-Document Synthesis** | 15% (8–15 Qs) | 3 Qs | Cross-document synthesis: Requires retrieving evidence from two separate PDF files. | *"Compare the failure points identified by Barnett et al. with the evaluation metrics in Ragas."* |
| **4** | **Exact Keyword / Alphanumeric** | 15% (8–15 Qs) | 3 Qs | Dense retrieval blindspots: Exact model codes, version numbers, or acronyms. | *"What is the exact arXiv identifier for the ColBERT research paper?"* |
| **5** | **Semantic / Paraphrased** | 10% (5–10 Qs) | 3 Qs | Vocabulary mismatch: Query uses completely different words than the document text. | *"What causes language models to struggle with accessing facts stored in their weights?"* |
| **6** | **Comparative & Aggregation** | 10% (5–10 Qs) | 3 Qs | Multi-passage aggregation: Summing, contrasting, or listing items across a document. | *"List all 7 failure points of RAG systems described in the engineering survey."* |
| **7** | **Unanswerable / Out-of-Corpus** | 10% (5–10 Qs) | 3 Qs | FP1 (Missing Content): Queries that cannot be answered from the corpus; tests abstention. | *"What were the Q3 financial earnings of Apple according to the provided papers?"* |

---

## 4. Ground-Truth Annotation Schema

Every benchmark item is stored in a versioned JSON/YAML test dataset (`data/benchmarks/benchmark_v1.json`). **Every item must include verified passage-level ground truth**, not merely expected answers:

```json
{
  "question_id": "DM-EVAL-001",
  "category": "single_document_factual",
  "benchmark_tier": "mvp_subset",
  "query": "What are the two probabilistic formulations of RAG proposed by Lewis et al.?",
  "ground_truth_answer": "Lewis et al. proposed RAG-Sequence, where the same retrieved passage is used to generate the complete sequence, and RAG-Token, where different passages can be used to generate each individual token.",
  "ground_truth_evidence": [
    {
      "doc_id": "lewis2020_rag",
      "source_filename": "lewis_et_al_2020_rag.pdf",
      "page_number": 3,
      "exact_quote_snippet": "In RAG-Sequence, the model uses the same retrieved passage to condition each generated token... In RAG-Token, the model can condition on different passages for each token."
    }
  ],
  "unanswerable": false,
  "required_keywords": ["RAG-Sequence", "RAG-Token", "token", "sequence"]
}
```

---

## 5. Quantitative Evaluation Metrics & Mathematical Formulations

DocuMind enforces a two-stage evaluation methodology separating retrieval from generation.

### 5.1 Retrieval Quality Metrics

1. **Recall@K:**
   $$\text{Recall}@K = \frac{|D_{\text{relevant}} \cap D_{\text{retrieved@K}}|}{|D_{\text{relevant}}|}$$
   Measures whether the required ground-truth evidence chunks were successfully retrieved in the top-$K$ candidate list.

2. **Precision@K:**
   $$\text{Precision}@K = \frac{|D_{\text{relevant}} \cap D_{\text{retrieved@K}}|}{K}$$
   Measures the signal-to-noise ratio within the retrieved context block.

3. **Mean Reciprocal Rank (MRR@K):**
   $$\text{MRR}@K = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
   where $\text{rank}_i$ is the position of the *first* relevant chunk for query $i$ (or $0$ if no relevant chunk appears in top-$K$). Evaluates ranking quality.

4. **Hit Rate@K:**
   $$\text{Hit}@K = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \mathbb{I}(|D_{\text{relevant}} \cap D_{\text{retrieved@K}}| > 0)$$
   Fraction of queries where at least one ground-truth passage was retrieved.

---

### 5.2 Generation Quality & Grounding Metrics

1. **Faithfulness (Factuality Score):**
   $$\text{Faithfulness} = \frac{|\text{Supported Claims in Answer}|}{|\text{Total Atomic Claims in Answer}|}$$
   Evaluated deterministically by decomposing the generated answer into atomic propositional claims and verifying that each claim is entailed by the retrieved context. A score of $1.0$ indicates zero ungrounded hallucinations.

2. **Answer Relevance:**
   Measures how directly the generated text answers the user question without conversational filler or tangential drift.

3. **Refusal Accuracy (Negative Query Precision):**
   $$\text{Refusal Accuracy} = \frac{|\text{Correctly Refused Unanswerable Queries}|}{|\text{Total Unanswerable Queries}|}$$
   Measures whether the system gracefully abstains ($"I cannot answer based on the provided documents."$) when queried on missing facts. Target $\ge 90\%$.

---

### 5.3 Citation Verifiability Metrics

1. **Citation Precision:**
   $$\text{Citation Precision} = \frac{|\text{Citations Matching Ground-Truth Source & Page}|}{|\text{Total Generated Citations}|}$$
   Verifies that generated bracketed references (e.g. `[Doc 1, p. 3]`) accurately map to the source document and page number containing the supporting quote.

2. **Citation Coverage:**
   $$\text{Citation Coverage} = \frac{|\text{Factual Statements with Valid Citations}|}{|\text{Total Factual Statements in Answer}|}$$
   Guarantees that every factual assertion is explicitly anchored to a document citation.

---

### 5.4 Operational Performance Metrics (CPU Hardware Profile)

All performance benchmarks must be recorded on the standardized Intel Core i5-1135G7 (4C/8T) laptop environment:
- **P50 / P95 Retrieval Latency (ms):** Wall-clock time for query embedding and vector search.
- **P50 / P95 End-to-End Latency (sec):** Total time from query submission to complete grounded answer generation.
- **Generation Throughput (tokens/sec):** Local LLM decoding speed on CPU.
- **Peak RAM Footprint (MB / GB):** Resident Set Size (RSS) during concurrent indexing and search operations.

---

## 6. Limitations of LLM-as-Judge & Ground-Truth Priority

While automated evaluation frameworks like Ragas rely on prompting an LLM (typically GPT-4) to grade answers, DocuMind explicitly recognizes the scientific limitations of LLM-as-a-judge:
1. **Self-Preference Bias:** Models tend to award higher scores to outputs generated by their own family or style.
2. **Local CPU Inconsistency:** Running LLM-as-judge on local 3B or 7B open-source models introduces significant evaluation noise, non-determinism, and reasoning errors.
3. **Circular Reasoning:** Evaluating a local model using another uncalibrated local model fails to produce publication-grade, defendable metrics.

**DocuMind Policy:** Human-annotated ground-truth passages and deterministic string/regex/entailment checks serve as the primary, authoritative ground truth. LLM-as-judge scoring is retained only as an auxiliary exploratory tool in Phase 8.

---

## 7. Reproducibility & Benchmark Execution Protocol

1. **Deterministic Environment:** Fixed seed (`seed=42`), zero temperature ($T=0.0$) for local LLM generation.
2. **Hardware Warm-up:** Run 3 throw-away queries before logging latency metrics to ensure OS file caches, threadpools, and model weights are warm.
3. **Execution Artifacts:** Every benchmark run produces a versioned JSON report in `data/benchmarks/reports/run_{timestamp}.json` recording exact git commit SHA, model names, temperature, latency arrays, and metric scores.
