# DocuMind 1.0 — Primary Research Paper Catalogue

**Repository:** `X:\Projects\Documind`  
**Phase:** Phase 1 — Literature Review and Baseline  
**Document Status:** Authoritative Research Catalogue (WP-01 Deliverable 2)  
**Hardware & Budget Context:** Intel Core i5-1135G7 (4C/8T, 2.4–4.2 GHz), 24 GB RAM, No discrete GPU; ₹0 financial budget; strictly offline / local CPU inference.

---

## 1. Overview & Cataloguing Methodology

This catalogue provides an authoritative, verified bibliographic record of primary scientific publications and foundational technical papers underpinning the architecture, retrieval algorithms, grounding mechanisms, and evaluation methodologies of **DocuMind 1.0**.

### Categorization Standards
1. **Primary Research Papers:** Peer-reviewed conference/journal publications or formal scientific preprints introducing original algorithms, mathematical formulations, empirical evaluations, or failure taxonomies.
2. **Reading Priority:**
   - **P1 (MVP Foundation):** Direct architectural relevance to building the naive baseline pipeline, chunking, embedding comparison, basic retrieval, grounded prompting, and manual evaluation.
   - **P2 (Version 1.0 Optimization):** Hybrid search (BM25 + RRF), cross-encoder reranking, context budget optimization, and automated metric evaluation (Ragas).
   - **P3 (Advanced / Experimental):** Corrective RAG (CRAG), Self-RAG, HyDE, hierarchical summarization (RAPTOR), and agentic control loops.
3. **Verification Policy:** Every entry includes verified authors, publication venues, DOIs or arXiv identifiers, and official open-access links. No citations or performance claims are fabricated.

---

## 2. Foundational RAG & Dense Retrieval Papers

### 2.1 Lewis et al. (2020) — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- **Full Authors:** Patrick Lewis, Ethan Perez, Aleksandara Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela.
- **Year:** 2020
- **Publication Venue:** Advances in Neural Information Processing Systems 33 (NeurIPS 2020), virtual.
- **Identifiers:** arXiv:2005.11401 [cs.CL] | DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
- **Official Source Link:** [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401) | [NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f7dd82e614f024b39110c-Abstract.html)
- **Research Problem:** Pre-trained sequence-to-sequence models store factual knowledge implicitly in parametric weights, leading to factual hallucinations, lack of explainability, and inability to update knowledge dynamically without costly full-model retraining.
- **Main Contribution:** Formulated the end-to-end differentiable RAG paradigm combining a neural retriever (Dense Passage Retrieval, DPR) with a seq2seq generator (BART). Introduced two probabilistic formulations:
  1. *RAG-Sequence:* Conditions on a single top document to generate the entire output sequence ($P(y|x) \approx \sum_{z} P(z|x) \prod_i P(y_i|x, z, y_{1:i-1})$).
  2. *RAG-Token:* Marginally conditions on different retrieved passages at each decoding step ($P(y|x) = \prod_i \sum_{z} P(z|x) P(y_i|x, z, y_{1:i-1})$).
- **Limitations:** End-to-end backpropagation through the retriever requires massive GPU clusters; fixed 100-word passage chunking ignores multi-page document layout; evaluated on open-domain trivia benchmarks (Natural Questions, TriviaQA) rather than localized technical documents.
- **DocuMind Relevance:** Establishes the core theoretical distinction between parametric memory (the LLM) and non-parametric memory (the vector index). Demonstrates that external retrieval drastically reduces factual hallucination.
- **Reading Priority:** **P1 (MVP Foundation)**

---

### 2.2 Karpukhin et al. (2020) — Dense Passage Retrieval for Open-Domain Question Answering (DPR)
- **Full Authors:** Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih.
- **Year:** 2020
- **Publication Venue:** Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP 2020), pp. 6769–6781.
- **Identifiers:** arXiv:2004.04906 [cs.CL] | DOI: [10.18653/v1/2020.emnlp-main.550](https://doi.org/10.18653/v1/2020.emnlp-main.550)
- **Official Source Link:** [https://aclanthology.org/2020.emnlp-main.550/](https://aclanthology.org/2020.emnlp-main.550/) | [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)
- **Research Problem:** Traditional information retrieval relies heavily on sparse keyword matching (TF-IDF, BM25), which fails when questions and passages do not share lexical tokens (synonymy and vocabulary mismatch).
- **Main Contribution:** Demonstrated that a dual-encoder architecture (two independent BERT networks mapping queries and passages into continuous $d$-dimensional space) trained using contrastive loss with in-batch negatives significantly outperforms BM25 on top-20 retrieval accuracy (e.g., 78.4% vs. 59.1% on Natural Questions).
- **Limitations:** Suffers on queries with rare specialized entities, exact numeric identifiers, or specific acronyms where lexical overlap is a stronger signal than semantic proximity. Requires indexing millions of dense vectors.
- **DocuMind Relevance:** Grounds DocuMind's dense vector retrieval stage. Explains the mathematical mechanics of bi-encoder dot-product similarity and highlights why dense retrieval alone is insufficient for technical or resume data without complementary sparse retrieval.
- **Reading Priority:** **P1 (MVP Foundation)**

---

### 2.3 Reimers & Gurevych (2019) — Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
- **Full Authors:** Nils Reimers, Iryna Gurevych.
- **Year:** 2019
- **Publication Venue:** Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP 2019), pp. 3982–3992.
- **Identifiers:** arXiv:1908.10084 [cs.CL] | DOI: [10.18653/v1/D19-1410](https://doi.org/10.18653/v1/D19-1410)
- **Official Source Link:** [https://aclanthology.org/D19-1410/](https://aclanthology.org/D19-1410/) | [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)
- **Research Problem:** Standard BERT produces sentence representations requiring quadratic cross-attention ($O(n^2)$), requiring ~65 hours to find the most similar pair among 10,000 sentences.
- **Main Contribution:** Introduced Siamese and triplet network structures to derive semantically meaningful sentence embeddings that can be compared via cosine similarity. Reduced search time from 65 hours to ~5 milliseconds using FAISS while preserving accuracy. Created the `sentence-transformers` library.
- **Limitations:** Fixed pooling strategies (mean pooling over token embeddings) compress variable-length paragraphs into a single vector, occasionally diluting fine-grained factual specifics.
- **DocuMind Relevance:** Forms the exact software foundation (`sentence-transformers`) used by candidate embedding models (`all-MiniLM-L6-v2`, `bge-small-en-v1.5`, `e5-small-v2`) in DocuMind's zero-cost CPU environment.
- **Reading Priority:** **P1 (MVP Foundation)**

---

## 3. Sparse & Hybrid Retrieval Papers

### 3.1 Robertson & Zaragoza (2009) — The Probabilistic Relevance Framework: BM25 and Beyond
- **Full Authors:** Stephen Robertson, Hugo Zaragoza.
- **Year:** 2009
- **Publication Venue:** Foundations and Trends in Information Retrieval, Vol. 3, No. 4, pp. 333–389.
- **Identifiers:** DOI: [10.1561/1500000019](https://doi.org/10.1561/1500000019)
- **Official Source Link:** [https://www.nowpublishers.com/article/Details/INR-019](https://www.nowpublishers.com/article/Details/INR-019) | [Open Access Preprint / PDF](https://www.staff.city.ac.uk/~sb317/papers/foundations_bm25_review.pdf)
- **Research Problem:** Rigorous mathematical grounding of term-frequency saturation and document-length normalization for lexical search.
- **Main Contribution:** Comprehensive formulation of BM25 scoring:
  $$\text{Score}(D, Q) = \sum_{i=1}^n \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
  Explains how parameter $k_1$ regulates term frequency saturation and $b$ controls length penalization.
- **Limitations:** Zero semantic generalization; completely misses synonyms, paraphrases, and conceptual matches.
- **DocuMind Relevance:** Informs DocuMind Version 1.0 hybrid retrieval. Essential for handling exact document identifiers, acronyms, and specialized technical terms that bi-encoders miss.
- **Reading Priority:** **P2 (Version 1.0 Optimization)**

---

### 3.2 Cormack, Clarke, & Buettcher (2009) — Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods
- **Full Authors:** Gordon V. Cormack, Charles L. A. Clarke, Stefan Buettcher.
- **Year:** 2009
- **Publication Venue:** Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '09), pp. 758–759.
- **Identifiers:** DOI: [10.1145/1571941.1572114](https://doi.org/10.1145/1571941.1572114)
- **Official Source Link:** [https://doi.org/10.1145/1571941.1572114](https://doi.org/10.1145/1571941.1572114)
- **Research Problem:** Combining retrieval results from disparate systems (e.g., BM25 scores with unbounded values vs. dense cosine similarities bounded in $[-1, 1]$) requires arbitrary score calibration and normalizations that fail across queries.
- **Main Contribution:** Formulated Reciprocal Rank Fusion (RRF), an unsupervised, rank-based aggregation formula:
  $$\text{RRF\_Score}(d \in D) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
  where $r_m(d)$ is the rank of document $d$ in system $m$, and $k$ is a smoothing constant (typically $k=60$). Proved that RRF consistently outperforms individual rankers and complex Condorcet fusions without requiring score calibration or training data.
- **Limitations:** Discards relative score margins (a document winning by a huge cosine margin gets the same rank as a marginal winner).
- **DocuMind Relevance:** The exact mathematical foundation for fusing dense vector search with sparse BM25 search in DocuMind Version 1.0. Zero-cost, zero-training, and trivial to compute in Python on CPU.
- **Reading Priority:** **P2 (Version 1.0 Optimization)**

---

### 3.3 Khattab & Zaharia (2020) — ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT
- **Full Authors:** Omar Khattab, Matei Zaharia.
- **Year:** 2020
- **Publication Venue:** Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '20), pp. 39–48.
- **Identifiers:** arXiv:2004.12832 [cs.IR] | DOI: [10.1145/3397271.3401075](https://doi.org/10.1145/3397271.3401075)
- **Official Source Link:** [https://dl.acm.org/doi/10.1145/3397271.3401075](https://dl.acm.org/doi/10.1145/3397271.3401075) | [arXiv:2004.12832](https://arxiv.org/abs/2004.12832)
- **Research Problem:** Bi-encoders are fast but compress all token interactions into one vector; cross-encoders are expressive but prohibitively slow ($O(N \cdot M)$).
- **Main Contribution:** Introduced *late interaction* via the `MaxSim` operator: query and passage tokens are embedded independently into vectors, and similarity is computed as the sum of maximum cosine similarities across all query tokens:
  $$S(Q, D) = \sum_{i \in Q} \max_{j \in D} (E(Q)_i \cdot E(D)_j^\top)$$
- **Limitations:** High memory footprint (storing token-level vector bags requires significant RAM/disk, e.g., 10–50x larger than single dense vectors).
- **DocuMind Relevance:** Assessed as a research alternative. Disqualified from DocuMind MVP due to CPU memory and disk constraints on the i5-1135G7 laptop; retained as an experimental reference for Version 1.0 late interaction.
- **Reading Priority:** **P3 (Experimental)**

---

## 4. Reranking & Context Management Papers

### 4.1 Nogueira & Cho (2019) — Passage Re-ranking with BERT
- **Full Authors:** Rodrigo Nogueira, Kyunghyun Cho.
- **Year:** 2019
- **Publication Venue:** arXiv preprint arXiv:1901.04085 [cs.IR].
- **Identifiers:** arXiv:1901.04085 [cs.IR] | DOI: [10.48550/arXiv.1901.04085](https://doi.org/10.48550/arXiv.1901.04085)
- **Official Source Link:** [https://arxiv.org/abs/1901.04085](https://arxiv.org/abs/1901.04085)
- **Research Problem:** First-stage retrieval (BM25 or bi-encoders) must search millions of passages in milliseconds, forcing trade-offs that sacrifice deep cross-token semantic interaction.
- **Main Contribution:** Demonstrated that feeding concatenated query-passage pairs `[CLS] Query [SEP] Passage [SEP]` through full multi-head self-attention layers in BERT allows every query token to attend directly to every passage token. Achieved dramatic state-of-the-art gains on MS MARCO (evaluating top-1000 candidate pools).
- **Limitations:** Computational cost scales quadratically with input length. Evaluating 100 passages on CPU takes multiple seconds, necessitating a strict candidate filter (e.g., reranking top-15 candidates only).
- **DocuMind Relevance:** Grounds DocuMind Version 1.0 second-stage reranking (e.g., using `bge-reranker-base` or `ms-marco-MiniLM-L-6-v2`). Proves why a two-stage retrieve-then-rerank pipeline achieves superior precision on CPU hardware.
- **Reading Priority:** **P2 (Version 1.0 Optimization)**

---

### 4.2 Liu et al. (2023) — Lost in the Middle: How Language Models Use Long Contexts
- **Full Authors:** Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang.
- **Year:** 2023 (Revised 2024)
- **Publication Venue:** Transactions of the Association for Computational Linguistics (TACL 2024), Vol. 12, pp. 157–173.
- **Identifiers:** arXiv:2307.03172 [cs.CL] | DOI: [10.1162/tacl_a_00638](https://doi.org/10.1162/tacl_a_00638)
- **Official Source Link:** [https://aclanthology.org/2024.tacl-1.9/](https://aclanthology.org/2024.tacl-1.9/) | [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)
- **Research Problem:** LLM generation performance degrades non-linearly when relevant information is positioned in the middle of long input contexts, even in models fine-tuned for 32k+ tokens.
- **Main Contribution:** Proved the "U-shaped" attention phenomenon: language models retrieve and reason over facts placed at the extreme beginning or extreme end of the prompt with high accuracy, but suffer degradation (often >30% drop in accuracy) when crucial information is placed in the center of multi-document context windows.
- **Limitations:** Tested primarily on multi-document question answering and key-value retrieval; does not prescribe a universal prompt structure for structured JSON outputs.
- **DocuMind Relevance:** Directly dictates context construction in DocuMind Phase 5. Informs our prompt assembly algorithm: reranked passages must be sorted such that the highest-scoring passage is placed first or last, rather than buried in the middle.
- **Reading Priority:** **P1 (MVP Foundation) & P2 (Optimization)**

---

## 5. RAG Failure Analysis & Evaluation Frameworks

### 5.1 Barnett et al. (2024) — Seven Failure Points When Engineering a Retrieval Augmented Generation System
- **Full Authors:** Scott Barnett, Stefanus Kurniawan, Srikanth Thudumu, Zach Brannelly, Mohamed Abdelrazek.
- **Year:** 2024
- **Publication Venue:** IEEE Software / arXiv preprint arXiv:2401.05856 [cs.SE].
- **Identifiers:** arXiv:2401.05856 [cs.SE] | DOI: [10.48550/arXiv.2401.05856](https://doi.org/10.48550/arXiv.2401.05856)
- **Official Source Link:** [https://arxiv.org/abs/2401.05856](https://arxiv.org/abs/2401.05856)
- **Research Problem:** Naive RAG systems built in tutorials frequently fail in production due to lack of systematic failure taxonomy and root-cause understanding.
- **Main Contribution:** Categorized the 7 canonical failure modes of real-world RAG architectures:
  1. *FP1 — Missing Content:* Required information does not exist in the source corpus.
  2. *FP2 — Missed Top Ranked:* The correct chunk exists in the database but fails to rank in the top-$K$.
  3. *FP3 — Not in Context (Budget / Truncation):* The chunk was retrieved but dropped due to context window limits.
  4. *FP4 — Not Extracted:* The chunk was in the prompt, but the LLM overlooked it ("lost in the middle" or prompt noise).
  5. *FP5 — Wrong Format:* The output failed schema requirements (e.g., missing bracketed citations).
  6. *FP6 — Incorrect Specificity:* The answer was too vague or excessively granular.
  7. *FP7 — Incomplete Answer:* Multi-hop or multi-chunk questions received only partial coverage.
- **Limitations:** Focuses on qualitative case studies and engineering lessons rather than a formal mathematical metric framework.
- **DocuMind Relevance:** Serves as DocuMind's explicit defensive engineering blueprint across Phases 1 through 8. Every pipeline stage in DocuMind is designed to guard against one of these 7 failure points.
- **Reading Priority:** **P1 (MVP Foundation)**

---

### 5.2 Es et al. (2023) — RAGAS: Automated Evaluation of Retrieval Augmented Generation
- **Full Authors:** Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert.
- **Year:** 2023 (Published EACL 2024)
- **Publication Venue:** Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations (EACL 2024), pp. 150–158.
- **Identifiers:** arXiv:2309.15217 [cs.CL] | DOI: [10.18653/v1/2024.eacl-demo.16](https://doi.org/10.18653/v1/2024.eacl-demo.16)
- **Official Source Link:** [https://aclanthology.org/2024.eacl-demo.16/](https://aclanthology.org/2024.eacl-demo.16/) | [arXiv:2309.15217](https://arxiv.org/abs/2309.15217)
- **Research Problem:** Traditional NLP metrics (BLEU, ROUGE) measure surface-level n-gram overlap, which correlates poorly with factual truthfulness and reasoning in generative RAG.
- **Main Contribution:** Formalized reference-free evaluation metrics isolating retrieval and generation:
  - *Faithfulness:* Fraction of claims in the generated answer that can be logically inferred from the retrieved context.
  - *Answer Relevance:* Degree to which the generated answer directly addresses the prompt question.
  - *Context Precision:* Signal-to-noise ratio of retrieved contexts (higher rank for ground-truth passages).
  - *Context Recall:* Fraction of required ground-truth sentences present in the retrieved chunks.
- **Limitations:** Relying on LLM-as-judge introduces model bias, non-determinism, and vulnerability to self-preference. In zero-cost CPU environments, local small models (3B/7B) may lack the reasoning capacity to grade complex faithfulness reliably.
- **DocuMind Relevance:** Establishes the evaluation metric definitions for Phase 8. DocuMind adapts these formulas into a deterministic, human-verified ground-truth harness for the 50–100 question benchmark.
- **Reading Priority:** **P2 (Version 1.0 Optimization)**

---

### 5.3 Gao et al. (2023) — Retrieval-Augmented Generation for Large Language Models: A Survey
- **Full Authors:** Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Qianyu Guo, Meng Wang, Haofen Wang.
- **Year:** 2023
- **Publication Venue:** arXiv preprint arXiv:2312.10997 [cs.CL].
- **Identifiers:** arXiv:2312.10997 [cs.CL] | DOI: [10.48550/arXiv.2312.10997](https://doi.org/10.48550/arXiv.2312.10997)
- **Official Source Link:** [https://arxiv.org/abs/2312.10997](https://arxiv.org/abs/2312.10997)
- **Research Problem:** Rapid expansion of RAG literature lacked a unified taxonomy categorizing the evolution from naive pipelines to advanced and modular architectures.
- **Main Contribution:** Formulated the three-stage evolutionary paradigm of RAG systems:
  1. *Naive RAG:* Indexing → Retrieval → Generation (prone to retrieval failures and generation hallucinations).
  2. *Advanced RAG:* Pre-retrieval optimization (query rewriting, routing) + Post-retrieval optimization (reranking, context compression).
  3. *Modular RAG:* Decoupled components with dynamic routing, self-correction, memory, and multi-source fusion.
- **Limitations:** Broad survey; does not provide implementation code or concrete benchmarks on consumer CPU hardware.
- **DocuMind Relevance:** Provides the macro-architectural framing that maps DocuMind's two-release strategy: Release 1 (MVP) implements Naive-to-Hardened RAG; Release 2 (Version 1.0) implements Advanced and Modular RAG.
- **Reading Priority:** **P1 (MVP Foundation)**

---

## 6. Advanced, Adaptive, & Self-Correcting RAG Papers

### 6.1 Asai et al. (2023) — Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
- **Full Authors:** Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi.
- **Year:** 2023 (Published ICLR 2024)
- **Publication Venue:** The Twelfth International Conference on Learning Representations (ICLR 2024).
- **Identifiers:** arXiv:2310.11511 [cs.CL] | DOI: [10.48550/arXiv.2310.11511](https://doi.org/10.48550/arXiv.2310.11511)
- **Official Source Link:** [https://openreview.net/forum?id=hSyW5tfBOv](https://openreview.net/forum?id=hSyW5tfBOv) | [arXiv:2310.11511](https://arxiv.org/abs/2310.11511)
- **Research Problem:** Standard RAG retrieves indiscriminately, even when the query does not require external facts, and cannot self-evaluate whether retrieved passages actually support its claims.
- **Main Contribution:** Introduced special reflection tokens (`[Retrieve]`, `[IsRel]`, `[IsSup]`, `[IsUse]`) enabling an LLM to dynamically decide *when* to retrieve, assess whether passages are relevant, evaluate whether its own generation is supported, and critique overall response utility.
- **Limitations:** Requires custom fine-tuning of language models on reflection tokens. Cannot be executed natively on off-the-shelf quantized general-purpose local LLMs without prompt overhead.
- **DocuMind Relevance:** Examined as a research reference. Informs prompt-based verification passes in DocuMind Phase 5/6, but ruled out as an MVP dependency due to training requirements.
- **Reading Priority:** **P3 (Experimental)**

---

### 6.2 Yan et al. (2024) — Corrective Retrieval Augmented Generation (CRAG)
- **Full Authors:** Shi-Qi Yan, Jia-Chen Gu, Yunzhu Zhu, Zhen-Hua Ling.
- **Year:** 2024
- **Publication Venue:** arXiv preprint arXiv:2401.15884 [cs.CL].
- **Identifiers:** arXiv:2401.15884 [cs.CL] | DOI: [10.48550/arXiv.2401.15884](https://doi.org/10.48550/arXiv.2401.15884)
- **Official Source Link:** [https://arxiv.org/abs/2401.15884](https://arxiv.org/abs/2401.15884)
- **Research Problem:** If first-stage retrieval fetches irrelevant or misleading passages, the generator inevitably suffers from hallucination ("garbage in, garbage out").
- **Main Contribution:** Designed an evaluator module that scores retrieved passages into three confidence bands:
  - *Correct:* Context is refined and compressed into knowledge strips.
  - *Incorrect:* Internal retrieval is discarded and replaced with web search.
  - *Ambiguous:* Both internal passages and web sources are merged.
- **Limitations:** External web search violates DocuMind's strict zero-cost and privacy constraints. Local confidence scoring adds latency on CPU hardware.
- **DocuMind Relevance:** The trial notebook promised CRAG in Cell 35 but never implemented it. DocuMind adapts CRAG's *internal* confidence gating for Phase 6 (triggering fallback retrieval or graceful abstention without external web search).
- **Reading Priority:** **P3 (Experimental / Version 1.0)**

---

### 6.3 Gao et al. (2022) — Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)
- **Full Authors:** Luyu Gao, Xueguang Ma, Jimmy Lin, Jamie Callan.
- **Year:** 2022 (Published ACL 2023)
- **Publication Venue:** Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1761–1777.
- **Identifiers:** arXiv:2212.10496 [cs.IR] | DOI: [10.18653/v1/2023.acl-long.100](https://doi.org/10.18653/v1/2023.acl-long.100)
- **Official Source Link:** [https://aclanthology.org/2023.acl-long.100/](https://aclanthology.org/2023.acl-long.100/) | [arXiv:2212.10496](https://arxiv.org/abs/2212.10496)
- **Research Problem:** Dense query-to-passage search struggles when the query is brief, abstract, or conceptually distant from the phrasing of source passages.
- **Main Contribution:** Hypothetical Document Embeddings (HyDE). Instructs an instruction-tuned LLM to generate a hypothetical passage answering the query; embeds this hypothetical document; and uses the resulting vector to retrieve real documents from the index. Maps query-space to document-space.
- **Limitations:** Doubles LLM inference passes (latency penalty on CPU); hallucinations in the hypothetical document can derail search on niche factual queries.
- **DocuMind Relevance:** Documented as an optional query-transformation experiment in Phase 6. Disqualified from MVP due to CPU inference latency constraints.
- **Reading Priority:** **P3 (Experimental)**

---

### 6.4 Sarthi et al. (2024) — RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval
- **Full Authors:** Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, Christopher D. Manning.
- **Year:** 2024 (Published ICLR 2024)
- **Publication Venue:** The Twelfth International Conference on Learning Representations (ICLR 2024).
- **Identifiers:** arXiv:2401.18059 [cs.CL] | DOI: [10.48550/arXiv.2401.18059](https://doi.org/10.48550/arXiv.2401.18059)
- **Official Source Link:** [https://openreview.net/forum?id=GN92VU4jCc](https://openreview.net/forum?id=GN92VU4jCc) | [arXiv:2401.18059](https://arxiv.org/abs/2401.18059)
- **Research Problem:** Traditional chunking limits retrieval to isolated, granular text windows, making it impossible to answer holistic or thematic questions spanning entire multi-chapter documents.
- **Main Contribution:** Constructs a multi-layer tree structure by recursively clustering text chunks based on semantic vectors and generating abstractive summaries of clusters using an LLM. Queries can retrieve from leaves (granular facts) or higher-level nodes (broad thematic summaries).
- **Limitations:** Upfront indexing cost is high (hundreds of LLM summarization calls per document), which is computationally burdensome on local CPU runtimes.
- **DocuMind Relevance:** Grounds hierarchical and parent-child retrieval research in Phase 2 and Phase 6. Inspires a lightweight, non-recursive parent-child chunk mapping for DocuMind.
- **Reading Priority:** **P3 (Experimental)**

---

## 7. Master Bibliographic Reference Table

| Paper Key | Primary Authors | Year | Core Topic | Venue | Status / Access | Priority |
| :--- | :--- | :---: | :--- | :--- | :---: | :---: |
| **Lewis et al.** | Lewis, Perez, Piktus, et al. | 2020 | Foundational RAG Formulation | NeurIPS 2020 | Verified Open Access | **P1 (MVP)** |
| **Karpukhin et al.** | Karpukhin, Oğuz, Min, et al. | 2020 | Dense Passage Retrieval (DPR) | EMNLP 2020 | Verified Open Access | **P1 (MVP)** |
| **Reimers & Gurevych** | Reimers, Gurevych | 2019 | Sentence-BERT Bi-Encoders | EMNLP 2019 | Verified Open Access | **P1 (MVP)** |
| **Robertson & Zaragoza** | Robertson, Zaragoza | 2009 | Probabilistic BM25 Formulation | FnTIR 2009 | Verified Open Access | **P2 (V1.0)** |
| **Cormack et al.** | Cormack, Clarke, Buettcher | 2009 | Reciprocal Rank Fusion (RRF) | SIGIR 2009 | Verified Open Access | **P2 (V1.0)** |
| **Nogueira & Cho** | Nogueira, Cho | 2019 | Cross-Encoder Passage Reranking | arXiv 2019 | Verified Open Access | **P2 (V1.0)** |
| **Liu et al.** | Liu, Lin, Hewitt, et al. | 2023 | Lost in the Middle Attention Bias | TACL 2024 | Verified Open Access | **P1 / P2** |
| **Barnett et al.** | Barnett, Kurniawan, et al. | 2024 | Seven Engineering Failure Points | IEEE / arXiv | Verified Open Access | **P1 (MVP)** |
| **Es et al.** | Es, James, Espinosa-Anke, et al. | 2023 | RAGAS Automated Evaluation | EACL 2024 | Verified Open Access | **P2 (V1.0)** |
| **Gao et al.** | Gao, Xiong, Gao, et al. | 2023 | RAG Survey & Taxonomy | arXiv 2023 | Verified Open Access | **P1 (MVP)** |
| **Asai et al.** | Asai, Wu, Wang, et al. | 2023 | Self-RAG Dynamic Reflection | ICLR 2024 | Verified Open Access | **P3 (Exp)** |
| **Yan et al.** | Yan, Gu, Zhu, Ling | 2024 | Corrective RAG (CRAG) | arXiv 2024 | Verified Open Access | **P3 (Exp)** |
| **Gao et al.** | Gao, Ma, Lin, Callan | 2022 | HyDE Query Generation | ACL 2023 | Verified Open Access | **P3 (Exp)** |
| **Sarthi et al.** | Sarthi, Abdullah, Tuli, et al. | 2024 | RAPTOR Tree Indexing | ICLR 2024 | Verified Open Access | **P3 (Exp)** |
| **Khattab & Zaharia** | Khattab, Zaharia | 2020 | ColBERT Late Interaction | SIGIR 2020 | Verified Open Access | **P3 (Exp)** |
