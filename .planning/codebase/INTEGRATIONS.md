# External Integrations

**Analysis Date:** 2026-11-22

## APIs & External Services

**Cloud APIs & Paid Services:**
- None. DocuMind is governed by a strict ₹0 financial budget and private data boundary policy.
- Zero outbound network traffic or external cloud LLM/embedding API dependencies (no OpenAI, Anthropic, Cohere, Pinecone, or AWS).
- 100% of text extraction, normalization, chunking, and future vector embedding/generation runs locally on host CPU hardware.

## Data Storage

**Local File Storage:**
- `resumes/`: Local collection of 25 sample candidate PDF resumes used for ingestion and resume screening experiments. Completely excluded from version control via `.gitignore`.
- `data/raw/`: Directory for unnormalized raw input documents.
- `data/processed/`: Staging directory for serialized chunks and normalized document outputs.
- `data/evaluation/`: Ground-truth datasets and evaluation benchmarks.

**Databases & Vector Stores:**
- Currently: In-memory structured dataclasses (`DocumentPage`, `TextChunk` in `src/documind/ingestion/schemas.py`).
- Planned (Phases 3–5): Local-only embedded vector database (e.g., FAISS or ChromaDB) running strictly in-process with zero network daemons or paid cloud hosting.

**Caching:**
- Local memory / `.pytest_cache` for test runner caching.

## Authentication & Identity

**Auth Provider:**
- None. DocuMind is designed as a local-first engineering and evaluation system without multi-tenant authentication or web user accounts.

## Monitoring & Observability

**Error Tracking & Diagnostics:**
- In-memory execution reports via `DocumentReport` and `IngestionReport` (`src/documind/ingestion/schemas.py`).
- Captures `documents_submitted`, `documents_processed`, `documents_failed`, `total_pages`, `pages_extracted`, `empty_pages`, `chunks_generated`, `error_message`, and wall-clock extraction latency.
- Structured command-line output via `scripts/test_ingestion.py`.

**Logs:**
- Standard output (`sys.stdout`) with UTF-8 encoding reconfiguration.

## CI/CD & Deployment

**Hosting:**
- Local execution on developer workstation.

**CI / Automation:**
- Local `pytest` suite enforcing 42 regression and unit test invariants.

## Environment Configuration

**Development:**
- No API keys or external secrets required.
- Requires Python 3.11+ environment with `pymupdf>=2.1.0` and `pytest>=8.0.0`.

---

*Integrations analysis: 2026-11-22*
*Update after adding external services or data stores*
