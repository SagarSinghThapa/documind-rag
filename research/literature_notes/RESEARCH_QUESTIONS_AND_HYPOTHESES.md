# DocuMind 1.0 — Research Questions and Hypotheses

**Repository:** `X:\Projects\Documind`  
**Phase:** Phase 1 — Literature Review and Baseline  
**Document Status:** Authoritative Scientific Hypotheses Register (WP-01 Deliverable 4)  
**Governance Standard:** *Understand → Hypothesize → Experiment → Measure → Decide → Implement → Document*  
**Strict Prohibition:** No hypothesis is marked as proven, and no experimental results are fabricated. All hypotheses establish rigorous experimental protocols to be executed during designated project phases.

---

## 1. Overview & Experimental Framework

Engineering decisions in DocuMind 1.0 must be justified by reproducible empirical measurements rather than qualitative intuition or tutorial dogma. This document defines the primary research questions, formal null and alternative hypotheses, experimental variables, evaluation datasets, and planned decision rules that govern architecture selections across Phases 2 through 8.

---

## 2. Research Questions & Formal Hypotheses

### RQ-01: Impact of Chunking Strategy on Retrieval Coverage
- **Research Question:** Does recursive character chunking with natural boundary preservation achieve higher retrieval Recall@10 and lower context fragmentation than fixed-size sliding-window chunking on structured technical documents?
- **Scientific Rationale:** Fixed-size chunking blindly splits text at arbitrary character or token counts, frequently cutting sentences and multi-token entities in half. Recursive character chunking prioritizes paragraph breaks (`\n\n`) and sentence boundaries (`\n`, ` `), preserving cohesive semantic units and reducing boundary-crossing fragmentation failures (Barnett et al. 2024, FP2/FP7).
- **Hypothesis Formulations:**
  - $H_0$: There is no statistically significant difference in Recall@10 between fixed-size chunking and recursive character chunking ($p \ge 0.05$).
  - $H_1$: Recursive character chunking achieves statistically significantly higher Recall@10 ($p < 0.05$) than fixed-size chunking on the same corpus.
- **Experimental Variables:**
  - *Independent Variable:* Chunking algorithm (`FixedSizeChunker` vs. `RecursiveCharacterChunker`).
  - *Dependent Variables:* Retrieval Recall@10, Mean Reciprocal Rank (MRR@10), boundary truncation count.
  - *Controlled Variables:* Document corpus (identical public PDF collection), target chunk size (512 tokens), overlap ratio (10%), embedding model (`all-MiniLM-L6-v2`), vector index (`FAISS FlatIP`), query set (identical 50-question benchmark).
- **Required Dataset:** 50–100 manually verified public technical document questions with annotated supporting passages.
- **Evaluation Metrics:** Recall@10, MRR@10, Context Precision.
- **Potential Confounders:** Tokenizer differences between chunker and embedding model; documents with atypical whitespace or multi-column layouts.
- **Planned Decision Rule:** If recursive chunking demonstrates $\ge 5\%$ higher Recall@10 without increasing indexing latency by $> 20\%$, select recursive chunking as the default ingestion strategy for Phase 2 MVP.

---

### RQ-02: Comparative Retrieval Quality & Latency of Candidate Embedding Models on CPU
- **Research Question:** Which open-source 384-dimensional embedding model candidate (`all-MiniLM-L6-v2`, `bge-small-en-v1.5`, `e5-small-v2`) provides the optimal balance of retrieval accuracy (MRR@10) and CPU inference latency under strict zero-cost local execution?
- **Scientific Rationale:** While all three models produce 384-dimensional vectors, they differ in training objectives, architecture depth (6-layer vs. 12-layer), sequence context limits (256 tokens for MiniLM vs. 512 for BGE/E5), and prefix formatting requirements (`query: `/`passage: ` for E5). Deeper models may yield higher semantic precision at the cost of increased CPU latency on the Intel i5-1135G7 processor.
- **Hypothesis Formulations:**
  - $H_0$: There is no significant difference in MRR@10 across `all-MiniLM-L6-v2`, `bge-small-en-v1.5`, and `e5-small-v2` ($p \ge 0.05$).
  - $H_1$: `bge-small-en-v1.5` and `e5-small-v2` achieve significantly higher MRR@10 ($p < 0.05$) than `all-MiniLM-L6-v2` on technical queries, but require $> 1.5\times$ higher CPU inference latency per batch.
- **Experimental Variables:**
  - *Independent Variable:* Embedding model (`all-MiniLM-L6-v2` vs. `bge-small-en-v1.5` vs. `e5-small-v2`).
  - *Dependent Variables:* MRR@10, Hit@5, CPU encoding latency per 32-chunk batch (ms), peak RAM footprint (MB).
  - *Controlled Variables:* Text chunks (identical chunked corpus), query set, vector index implementation, CPU thread count (fixed at 4 threads on i5-1135G7).
- **Required Dataset:** 50–100 question evaluation benchmark annotated with ground-truth chunk IDs.
- **Evaluation Metrics:** MRR@10, Hit@1, Hit@5, P50/P95 batch latency (ms), RSS RAM usage (MB).
- **Potential Confounders:** Failure to append required query/passage prefixes for E5; unnormalized vectors skewing cosine similarity calculations.
- **Planned Decision Rule:** Select the candidate that achieves the highest MRR@10 while maintaining single-chunk CPU inference latency under 50 ms. If a deeper model improves MRR@10 by $< 3\%$ but doubles latency, select the faster lightweight model for MVP.

---

### RQ-03: Vector Indexing Trade-offs: FAISS vs. ChromaDB Persistence
- **Research Question:** What are the quantitative trade-offs in search latency, indexing throughput, memory footprint, and document update ergonomics between file-serialized FAISS and embedded directory ChromaDB under local CPU constraints?
- **Scientific Rationale:** FAISS provides raw C++ vector search efficiency with minimal overhead, but lacks native document update/deletion primitives and metadata catalogs. ChromaDB provides an integrated persistence layer and metadata filtering, but introduces SQLite and Python binding overhead.
- **Hypothesis Formulations:**
  - $H_0$: Search latency and memory footprint between FAISS and ChromaDB differ by less than 10% across scales from 1,000 to 10,000 chunks.
  - $H_1$: FAISS achieves $> 2\times$ faster search latency and lower memory consumption, while ChromaDB provides significantly lower latency and complexity for incremental document additions and metadata-filtered queries.
- **Experimental Variables:**
  - *Independent Variable:* Vector index implementation (`FAISS IndexFlatIP` with disk serialization vs. `ChromaDB PersistentClient` with HNSW).
  - *Dependent Variables:* Search latency (P50/P95 in ms), indexing build time (sec/1000 chunks), disk storage overhead (MB), incremental document update time (ms).
  - *Controlled Variables:* Embeddings (identical pre-computed 384-dim float32 vectors), corpus size scales ($N = 1000, 5000, 10000$), query batch size ($B=1$).
- **Required Dataset:** Synthetic and real technical chunk collections scaled up to 10,000 chunks.
- **Evaluation Metrics:** Query latency (ms), Index write time (sec), RAM usage during query (MB), Disk space (MB).
- **Potential Confounders:** Background OS filesystem caching; cold start vs. warm start index query times.
- **Planned Decision Rule:** If ChromaDB search latency remains under 15 ms at 10,000 chunks and its metadata filtering eliminates the need for external mapping databases, select ChromaDB for the MVP. If ChromaDB exceeds memory or latency budgets, select FAISS.

---

### RQ-04: Hybrid Retrieval Efficacy (Dense + BM25 via Reciprocal Rank Fusion)
- **Research Question:** Does hybrid retrieval combining dense semantic bi-encoders with sparse lexical BM25 via Reciprocal Rank Fusion (RRF) achieve higher Recall@10 and MRR@10 than dense-only or BM25-only retrieval across diverse technical query types?
- **Scientific Rationale:** Dense retrieval excels at semantic paraphrasing but fails on exact part numbers, identifiers, and rare terms (Karpukhin et al. 2020). BM25 excels at exact keyword matching but suffers from vocabulary mismatch on conceptual questions (Robertson & Zaragoza 2009). RRF provides an unsupervised, rank-based combination that should dominate both single-mode retrievers (Cormack et al. 2009).
- **Hypothesis Formulations:**
  - $H_0$: Hybrid RRF retrieval achieves no statistically significant improvement in Recall@10 over the best individual retriever ($p \ge 0.05$).
  - $H_1$: Hybrid RRF retrieval achieves statistically significantly higher Recall@10 ($p < 0.05$) than dense-only retrieval, particularly on exact keyword and technical entity queries.
- **Experimental Variables:**
  - *Independent Variable:* Retrieval method (`Dense-only`, `BM25-only`, `Hybrid-RRF`).
  - *Dependent Variables:* Recall@10, MRR@10, query latency (ms).
  - *Controlled Variables:* Corpus, query set (stratified by semantic vs. keyword queries), top-$K$ candidates retrieved ($K=10$).
- **Required Dataset:** Full 50–100 question benchmark containing both exact-keyword queries (e.g. specific tool names, model versions) and semantic conceptual queries.
- **Evaluation Metrics:** Recall@10, MRR@10, Latency overhead.
- **Potential Confounders:** Inconsistent tokenization between BM25 and embedding tokenizer; choice of RRF smoothing constant $k$ (standard $k=60$).
- **Planned Decision Rule:** If Hybrid RRF improves overall Recall@10 by $\ge 8\%$ on keyword queries without degrading semantic MRR@10 by more than $2\%$, adopt Hybrid RRF as the standard retrieval engine in Phase 4 Version 1.0.

---

### RQ-05: Cross-Encoder Reranking: Ranking Quality Gain vs. CPU Latency Penalty
- **Research Question:** Does second-stage cross-encoder reranking (e.g. `bge-reranker-base`) over top-15 retrieved candidates improve MRR@10 and generation Faithfulness enough to justify the additional CPU latency?
- **Scientific Rationale:** Cross-encoders perform full all-to-all cross-attention between query and passage tokens (Nogueira & Cho 2019), resolving subtle relevance distinctions that bi-encoders miss. However, cross-encoders are computationally expensive ($O(N \cdot M)$) and run sequentially on CPU.
- **Hypothesis Formulations:**
  - $H_0$: Cross-encoder reranking yields no significant improvement in MRR@5 or downstream generation Faithfulness over first-stage retrieval ($p \ge 0.05$).
  - $H_1$: Cross-encoder reranking improves MRR@5 by $\ge 15\%$ ($p < 0.01$), but increases end-to-end retrieval latency by 400–800 ms on CPU.
- **Experimental Variables:**
  - *Independent Variable:* Reranking presence and model (`No Reranker` vs. `bge-reranker-base` over top-15 candidates).
  - *Dependent Variables:* MRR@5, NDCG@5, Context Precision, CPU latency (ms), downstream answer Faithfulness.
  - *Controlled Variables:* First-stage retrieval candidates (top-15), local LLM generator, generation prompt.
- **Required Dataset:** 50–100 question evaluation benchmark with multi-chunk distractor passages.
- **Evaluation Metrics:** MRR@5, NDCG@5, Latency (ms), Faithfulness score.
- **Potential Confounders:** First-stage candidate pool truncation (if the ground-truth chunk is not in the top-15, the reranker cannot fix it).
- **Planned Decision Rule:** If cross-encoder reranking increases MRR@5 by $\ge 12\%$ and average reranking latency on CPU remains $< 600$ ms for 15 chunks, incorporate cross-encoder reranking into Phase 4 Version 1.0.

---

### RQ-06: Chunk Size & Overlap Trade-offs on Retrieval Precision and Context Dilution
- **Research Question:** What combination of chunk size (256, 512, 1024 tokens) and overlap ratio (0%, 10%, 20%) minimizes context fragmentation while preventing context dilution and "lost in the middle" attention degradation?
- **Scientific Rationale:** Small chunks (256 tokens) optimize dense retrieval precision but fragment complex ideas across boundaries. Large chunks (1024 tokens) preserve broad context but dilute embedding vectors with irrelevant noise and consume large portions of the LLM context window (Liu et al. 2023). Overlap prevents boundary cutoffs at the cost of duplicate context.
- **Hypothesis Formulations:**
  - $H_0$: Varying chunk size and overlap ratio produces no significant difference in combined retrieval recall and generation faithfulness.
  - $H_1$: A chunk size of 512 tokens with 10% overlap achieves the optimal Pareto frontier between retrieval Recall@10 and answer Faithfulness.
- **Experimental Variables:**
  - *Independent Variables:* Chunk size (256, 512, 1024 tokens) and Overlap (0%, 10%, 20%) in a $3 \times 3$ grid search.
  - *Dependent Variables:* Recall@10, Total indexed chunks, Context Precision, downstream Faithfulness.
  - *Controlled Variables:* Ingestion engine (`RecursiveCharacterChunker`), embedding model, evaluation questions.
- **Required Dataset:** Document evaluation benchmark covering multi-chunk and single-fact queries.
- **Evaluation Metrics:** Recall@10, Context Precision, Generation Faithfulness, Indexing Duration.
- **Potential Confounders:** Discrepancies between character-length approximations and actual model tokenizer token counts.
- **Planned Decision Rule:** Select the chunk size and overlap combination yielding the highest combined harmonic mean (F1-score) of Recall@10 and Context Precision for Phase 2 default configuration.

---

### RQ-07: Grounded Prompting & Abstention Accuracy on Unanswerable Queries
- **Research Question:** Can local 3B–7B parameter open-source language models running on CPU reliably abstain from answering out-of-corpus queries (achieving $\ge 90\%$ refusal accuracy) and generate verifiable inline citations `[Doc, p. X]` without external proprietary APIs?
- **Scientific Rationale:** Production RAG systems frequently fail when users ask questions unanswerable from the uploaded documents, leading to catastrophic hallucinations (Barnett et al. 2024, FP1). Strict negative constraints and Pydantic structured schemas should force the LLM to populate an `unknowns` or refusal status rather than guessing.
- **Hypothesis Formulations:**
  - $H_0$: Local 3B/7B models fail to achieve $> 75\%$ refusal accuracy on unanswerable queries under prompt constraints.
  - $H_1$: Structured grounded prompting combined with refusal instructions enables local 3B/7B models to achieve $\ge 90\%$ refusal accuracy and $\ge 85\%$ citation precision without commercial APIs.
- **Experimental Variables:**
  - *Independent Variables:* Prompt template design (Unconstrained baseline vs. Grounded system prompt with explicit refusal vs. Pydantic schema enforcement).
  - *Dependent Variables:* Refusal Accuracy on unanswerable queries (%), Citation Precision (%), Hallucination Rate (%).
  - *Controlled Variables:* Local LLM runtime (e.g. quantized Llama-3.2-3B or Qwen-2.5-3B on Ollama/llama.cpp), temperature (0.0), retrieved context.
- **Required Dataset:** 20 unanswerable queries (information completely missing from corpus) + 30 answerable queries.
- **Evaluation Metrics:** Refusal Accuracy, Citation Precision, Citation Coverage, Hallucination Rate.
- **Potential Confounders:** Instruction-following fidelity variance across different model quantization levels (e.g. Q4_K_M vs. Q8_0).
- **Planned Decision Rule:** If grounded structured prompting achieves $\ge 90\%$ refusal on negative queries while maintaining $\ge 80\%$ citation precision on positive queries, approve the prompt and schema architecture for Phase 5 MVP.

---

## 3. Summary Mapping of Hypotheses to Project Phases

| Hypothesis ID | Core Focus | Primary Experimental Phase | MVP vs. Version 1.0 Milestone | Decision Impact |
| :---: | :--- | :---: | :---: | :--- |
| **RQ-01** | Recursive vs. Fixed Chunking | Phase 2 | **MVP-Critical** | Determines primary PDF chunking strategy. |
| **RQ-02** | Embedding Candidate Shootout | Phase 3 | **MVP-Critical** | Selects single local embedding model for Release 1. |
| **RQ-03** | FAISS vs. Chroma Persistence | Phase 3 | **MVP-Critical** | Selects vector storage and persistence engine. |
| **RQ-04** | Hybrid Dense + BM25 Fusion | Phase 4 | **Version 1.0** | Decides adoption of Reciprocal Rank Fusion. |
| **RQ-05** | Cross-Encoder Reranking | Phase 4 | **Version 1.0** | Establishes whether reranking justifies CPU latency. |
| **RQ-06** | Chunk Size & Overlap Grid Search | Phase 2 | **MVP-Critical** | Sets default chunk token size and overlap ratio. |
| **RQ-07** | Grounding & Refusal Accuracy | Phase 5 | **MVP-Critical** | Finalizes prompt template and structured citation schema. |
