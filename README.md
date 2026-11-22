# DocuMind 1.0

## Project Purpose
DocuMind is a modular, general-purpose Retrieval-Augmented Generation (RAG) system built from scratch to deliver accurate, grounded, and verifiable responses over diverse document collections.

> **Project Independence:** DocuMind is an entirely independent clean-slate initiative developed from the ground up. It is strictly separate from the legacy RAG chatbot, sharing no code, architecture, or dependencies with earlier implementations.

## Vision
To establish an extensible, production-grade RAG architecture that bridges deep document understanding with language models, emphasizing modularity, transparency, and empirical rigor across all components.

## Engineering Methodology
DocuMind follows a research-driven, hypothesis-tested engineering lifecycle:

```text
Understand -> Hypothesize -> Experiment -> Measure -> Decide -> Implement -> Document
```

- **Understand:** Define retrieval and generation challenges; analyze domain constraints and synthesize relevant literature.
- **Hypothesize:** Formulate concrete, testable hypotheses for architectural patterns, data pipelines, or algorithmic choices.
- **Experiment:** Execute controlled, isolated, and reproducible experiments.
- **Measure:** Evaluate outcomes using quantitative retrieval and generation metrics.
- **Decide:** Justify technology and design decisions strictly through empirical evidence.
- **Implement:** Construct modular, maintainable, and well-tested components.
- **Document:** Record architectural decisions (ADRs), benchmark results, and operational guidelines.

## Current Project Status
- **Current Phase:** Foundation / Phase 0
- **Status:** Initial repository structure initialized; research, documentation, and evaluation scaffolding established.
- **Note:** Implementation details, dependency selections, embedding models, vector databases, chunking strategies, and retrieval pipelines are deliberately pending research and experiments.

## Development Roadmap & Release Strategy

### Release Milestones
The project executes against two distinct release milestones while strictly adhering to the comprehensive nine-phase master plan:

- **Release 1: DocuMind MVP (Target: 2–3 weeks)**
  - A functional, demonstrable baseline RAG application.
  - Scope is strictly bounded to an operational basic RAG pipeline: PDF ingestion, chunking, embeddings, vector retrieval, grounded generation, citations, a functional interface, and initial evaluation.
  - Advanced capabilities, query rewriting, and complex workflows are deliberately deferred to avoid premature complexity.
- **Release 2: DocuMind 1.0**
  - A technically substantial, comprehensive, and resume-worthy production-grade application.
  - Progressively incorporates advanced retrieval, reranking, query transformation, advanced RAG workflows, robust engineering, thorough ablation studies, end-to-end evaluation, and deployment.

### Release Prioritization
All engineering tasks and research explorations across the master plan are categorized into three prioritization tiers:
- **MVP-Critical:** Essential baseline capabilities required to deliver a functional, end-to-end pipeline within the 2–3 week MVP delivery window.
- **Version 1.0:** Robust engineering, advanced retrieval and generation pipelines, production hardening, and comprehensive evaluation.
- **Research / Experimental:** Exploratory hypotheses, alternative models, and architectural variations. Advanced features will only be adopted when justified by empirical experiment results.

### Core Project Principles
Throughout both release milestones, all work strictly adheres to these non-negotiable principles:
- **Research before implementation:** Formulate problem definitions and review relevant literature prior to coding.
- **Baseline before optimization:** Establish measurable, naive baselines before introducing complex enhancements.
- **Controlled experiments:** Isolate variables and benchmark against fixed evaluation datasets.
- **Evidence-based technical decisions:** Adopt algorithms, models, and architectures only when supported by quantitative metrics.
- **Reproducibility:** Maintain deterministic pipelines, versioned evaluation sets, and repeatable experimental setups.
- **Documentation of failures and limitations:** Transparently document negative findings, trade-offs, and operational boundaries alongside successes.

### Master Development Plan (Nine Phases)
The original nine-phase master plan remains authoritative. Neither release replaces or simplifies this plan:

1. **Phase 0 — Project Foundation** *(Current)*
   - Repository, documentation structure, research archive, project governance and initial audit.
2. **Phase 1 — Literature Review and Baseline**
   - Foundational RAG literature, baseline implementation, fixed evaluation question set and baseline benchmark.
3. **Phase 2 — Document Ingestion and Chunking**
   - PDF parsing, chunking strategies, chunk size/overlap, metadata preservation and parent-child retrieval.
4. **Phase 3 — Embedding and Indexing**
   - Embedding model experiments, vector indexing comparisons, persistence and index updates.
5. **Phase 4 — Retrieval Optimization**
   - Dense retrieval, BM25, hybrid search, reranking and context-budget optimization.
6. **Phase 5 — Generation and Grounding**
   - LLM and inference comparisons, grounded prompting, citation generation, unsupported queries and claim/evidence validation.
7. **Phase 6 — Advanced RAG**
   - Query rewriting, multi-query retrieval, adaptive/corrective retrieval, multi-document reasoning and conversational memory.
8. **Phase 7 — Product Engineering**
   - FastAPI, Streamlit prototype, collections, chat, citation viewer and conversation history. React is conditional on justification.
9. **Phase 8 — Validation and Deployment**
   - Evaluation, ablation studies, integration and failure testing, resource profiling, deployment and final documentation.

## Repository Structure
```text
DocuMind/
├── docs/
│   ├── architecture/
│   ├── decisions/
│   ├── experiments/
│   ├── findings/
│   ├── evaluation/
│   └── project_reports/
├── research/
│   ├── papers/
│   ├── technical_references/
│   └── literature_notes/
├── data/
│   ├── raw/
│   ├── processed/
│   └── evaluation/
├── src/
├── tests/
├── notebooks/
├── scripts/
├── configs/
├── README.md
└── .gitignore
```
