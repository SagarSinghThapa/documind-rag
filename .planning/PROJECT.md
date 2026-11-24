# DocuMind

## What This Is

DocuMind is a modular, general-purpose, local-first Retrieval-Augmented Generation (RAG) system built from scratch to deliver accurate, grounded, and verifiable responses over document collections. Designed to run 100% locally on CPU hardware with zero paid APIs or cloud infrastructure, it serves as a robust, production-grade application tailored for technical interviews, portfolio demonstrations, and resume presentation.

## Core Value

Deliver accurate, grounded answers with verifiable page-level citations from local documents under a strict ₹0 cloud API budget and local-first privacy boundary.

## Requirements

### Validated

<!-- Shipped and confirmed valuable -->
(None yet — existing WP-02 code requires independent environment repair and audit verification before validation)

### Active

<!-- Current scope: Milestone 1 (DocuMind MVP) -->
- [ ] **AUDIT-01**: Working Python virtual environment established with verified dependency compatibility on Windows CPU
- [ ] **AUDIT-02**: Comprehensive independent audit of PDF extraction, normalization, chunking, deterministic IDs, and privacy
- [ ] **AUDIT-03**: All existing unit and regression tests passing with full test verification evidence
- [ ] **EMBED-01**: CPU-compatible local embedding model selected and integrated without paid APIs
- [ ] **EMBED-02**: Persistent local vector index (FAISS or Chroma) with document and page metadata preservation
- [ ] **RETR-01**: Query embedding generation and Top-K vector similarity retrieval
- [ ] **RETR-02**: Relevance threshold filtering and structured context window construction
- [ ] **GEN-01**: CPU-compatible local LLM integrated for grounded text generation
- [ ] **GEN-02**: Strict page-level citations and graceful abstention for unsupported questions
- [ ] **APP-01**: Streamlit web interface with multi-PDF upload, document manager, and indexing controls
- [ ] **APP-02**: Conversational Q&A chat interface with interactive source citations and error handling
- [ ] **EVAL-01**: End-to-end integration testing, retrieval quality metrics, and citation accuracy evaluation
- [ ] **EVAL-02**: Performance benchmarking, memory envelope profiling (<= 24 GB RAM), and presentation documentation

### Out of Scope

<!-- Explicit boundaries to prevent scope creep during Milestone 1 MVP -->
- BM25 and hybrid retrieval — Deferred to Milestone 2 (Advanced DocuMind)
- Reciprocal Rank Fusion (RRF) — Deferred to Milestone 2
- Cross-encoder reranking — Deferred to Milestone 2
- Query rewriting / expansion / HyDE — Deferred to Milestone 2
- Multi-document comparative analysis — Deferred to Milestone 2
- Standalone evaluation dashboard — Deferred to Milestone 2
- Cloud APIs (OpenAI, Anthropic, Cohere, Pinecone) — Strict ₹0 budget and privacy constraint
- Cloud deployment / remote hosting — Strictly local-first execution

## Context

- Brownfield codebase: Phase 0 foundation, Phase 1 research docs, and WP-02 ingestion pipeline exist.
- Codebase mapping completed in `.planning/codebase/`.
- Crucial nuance: WP-02 is not yet independently verified because the current environment failed to execute pytest (missing venv/dependencies). Phase 1 will repair the environment and audit WP-02 before proceeding.

## Constraints

- **Budget**: ₹0 financial budget; zero paid external APIs or cloud services.
- **Privacy**: Local-first processing; private documents (e.g. `resumes/`) must never leave the machine.
- **Hardware**: Windows 11 x86_64, Intel Core i5 CPU, maximum 24 GB RAM envelope.
- **Architecture**: Modular Python architecture using clean abstractions and standard dataclasses; avoid unnecessary heavy frameworks.
- **Execution**: Plan and execute one phase at a time; never automatically advance without explicit user approval.
- **Git Control**: No git commits, pushes, or remote modifications without explicit user approval.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Local-first CPU architecture | Zero cost, total privacy, reproducibility in technical demos | ✓ Good |
| WP-02 audit as Phase 1 | Existing tests failed in current environment; cannot build on unverified foundation | ✓ Good |
| Defer hybrid retrieval & reranking | Milestone 1 MVP prioritizes a working, tested end-to-end baseline first | ✓ Good |
| Sequential phase execution | One work package at a time with testing and verification before phase completion | ✓ Good |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-11-22 after project initialization*
