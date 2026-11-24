---
phase: 03-wp-04-semantic-retrieval
plan: 02
status: complete
completed_at: "2026-11-22T23:12:00+05:30"
duration_minutes: 25
tests_passed: 11
tests_failed: 0
total_tests: 11
regressions: 0
---

# Phase 3 Plan 2: ContextBuilder, Dynamic Deduplication & Retrieval Evaluation Summary

## 1. Overview & Objective
Plan 03-02 implemented citation-grounded, budget-enforced context construction with dynamic overlap deduplication and untrusted text encapsulation, alongside a formal Information Retrieval evaluation framework that prevents evaluation leakage, separates positive and negative query metrics, and benchmarks stage latencies.

## 2. Key Accomplishments
1. **ContextBuilder & Untrusted Content Encapsulation:**
   - Implemented `ContextBuilder`, `ContextReference`, and `FormattedContext` in `src/documind/retrieval/context_builder.py`.
   - Structural blockquote encapsulation: wraps chunk text in `> [Document Content - Untrusted Context]` so downstream LLMs never mistake document text for system instructions.
   - Dynamic overlap deduplication: scans maximal common substring between consecutive chunks from the same document; strips duplicate prefix only when verified overlap $\ge \text{min\_overlap\_chars}$ (20 chars). Non-overlapping chunks are untouched.
   - Budget management: packs chunks within `max_context_chars` without mid-word cutting; tracks `chunks_discarded_budget`.
   - Structured citation records: emits `ContextReference` capturing physical page numbers and character offsets.
2. **Ground-Truth Dataset (Leakage-Free Partitioning):**
   - Created `data/evaluation/retrieval_ground_truth.json` with 20 manually verified queries:
     - **Split A (Calibration - 10 queries):** 6 in-domain positive queries grounded in `BASWE-15-RAG-Concepts-Guide.pdf` (pages 1–14) + 4 negative distractor queries.
     - **Split B (Holdout Evaluation - 10 queries):** 6 in-domain positive queries grounded in `Everything-to-know-about-RAG.pdf` (pages 1–92) + 4 negative distractor queries.
   - All ground truth is verified against actual physical PDF page numbers and exact verbatim text excerpts.
3. **RetrievalEvaluator & Metric Math:**
   - Implemented `RetrievalEvaluator` in `src/documind/retrieval/evaluator.py`.
   - Evaluates positive queries and negative queries separately with explicit denominators: Recall@K, Precision@K, MRR, Negative Rejection Rate (NRR), False Positive Rate (FPR), and Attribution Accuracy.
   - Computes Mean, Median (P50), and P95 latency profiles for query embedding, FAISS search, context construction, and total pipeline.
4. **Standalone Evaluation CLI (`scripts/evaluate_retrieval.py`):**
   - Ingests research PDFs (195 vectors total).
   - Swept candidate thresholds on Split A: $\tau = 0.25$ / $0.35$ achieved Recall@5 = 0.833, NRR = 0.750–1.000 (up from 0.000 on baseline).
   - Evaluated holdout Split B under $\tau^* = 0.25$: Recall@5 = 0.667, Precision@5 = 0.333, MRR = 0.667, NRR = 0.750, FPR = 0.250, Attribution Accuracy = 1.000.
   - Disaggregated latency on CPU: Embedding Mean = 43.89 ms (P95 = 52.86 ms), FAISS Search Mean = 0.46 ms, Context Construction Mean = 0.20 ms, Total Pipeline Mean = 44.67 ms (P95 = 53.66 ms).
5. **Testing & Regression Validation:**
   - Authored `tests/test_context_builder.py` (7 tests) and `tests/test_retrieval_eval.py` (4 tests).
   - Full test suite execution across all 90 tests: **90 passed, 0 failures, 0 skipped in 144.80s**. Zero regressions across Phase 1 (42 tests) and Phase 2 (26 tests).
