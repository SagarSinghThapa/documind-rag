# Phase 1: WP-02 Independent Audit and Environment Repair - Context

**Gathered:** 2026-11-22  
**Status:** Ready for planning  
**Source:** User Requirements & Codebase Mapping

<domain>
## Phase Boundary

Phase 1 focuses exclusively on establishing a verifiable, working execution environment, independently auditing the implemented WP-02 ingestion and chunking pipeline, executing all tests to establish an empirical baseline pass count, resolving any blocking defects, and producing an evidence-based audit report.

No downstream RAG features (embeddings, vector store, retrieval, or LLMs) will be implemented during this phase.

</domain>

<decisions>
## Implementation Decisions

### Environment Setup
- Create a dedicated Python virtual environment (`.venv`) at the repository root (`D:\rag\.venv`).
- Install dependencies from `requirements.txt` (`pymupdf>=2.1.0`, `pytest>=8.0.0`).
- Ensure Windows 11 x86_64 CPU compatibility without requiring C++ build tools or external DLLs.

### Test Execution & Audit Verification
- WP-02 is NOT treated as approved until all tests are executed and passed in the local `.venv`.
- Run `pytest` across all existing test files in `tests/`:
  - `tests/test_chunker.py`
  - `tests/test_pdf_loader.py`
  - `tests/test_text_normalizer.py`
  - `tests/test_ingestion_pipeline.py`
- Confirm exact test counts, pass rates, and execution runtimes.

### Component-by-Component Independent Audit
- **PDF Extraction**: Audit `PDFLoader` on sample PDFs (`research/technical_references/`). Verify `%PDF-` signature checking, file size bounds, blank page detection, and scanned/image-only PDF diagnostic warnings.
- **Text Normalization**: Audit `TextNormalizer` on edge cases: Unicode NFC composition, control character stripping, CRLF $\rightarrow$ LF, and de-hyphenation.
- **Chunking Algorithms**: Audit `FixedSizeChunker` and `RecursiveCharacterChunker` for strict chunk size bounds, character offset alignment (`unified_text[char_start:char_end] == chunk.text`), and cross-page interval attribution.
- **Deterministic IDs**: Verify that document IDs are SHA-256 hashes of file contents and chunk IDs are deterministic.
- **Privacy Boundary**: Confirm that candidate resumes in `resumes/` are strictly local, ignored by `.gitignore`, and never leaked.

### Defect Remediation & Reporting
- If any tests fail or blocking defects are discovered, repair them directly in `src/documind/` and add regression tests.
- Produce an evidence-based audit report at `docs/project_reports/WP_02_INDEPENDENT_AUDIT_REPORT.md`.
- Explicit user approval is mandatory before Phase 2 begins.

### GSD Safety Invariants
- Preserve the existing codebase and architecture.
- Do not make git commits or remote modifications without explicit user approval.
- Do not alter private documents.

</decisions>

<canonical_refs>
## Canonical References

- `docs/architecture/INGESTION_PIPELINE.md` — Ingestion architecture and extraction engine selection.
- `docs/architecture/BASELINE_RAG_SPECIFICATION.md` — Baseline RAG pipeline specifications.
- `docs/project_reports/WP_02_INDEPENDENT_AUDIT.md` — Prior audit report for reference.
- `requirements.txt` — Minimal dependency specification.
- `src/documind/` — Existing ingestion pipeline modules.
- `tests/` — Automated test suite.

</canonical_refs>

<specifics>
## Specific Ideas

- Use `.venv\Scripts\python.exe` and `.venv\Scripts\pytest.exe` to guarantee deterministic isolation from host environment.
- Use `scripts/test_ingestion.py` as a smoke test runner for real PDF validation.

</specifics>

<deferred>
## Deferred Ideas

- Embedding generation and vector storage — Deferred to Phase 2 (WP-03).
- Semantic retrieval and context construction — Deferred to Phase 3 (WP-04).
- Local LLM generation and citations — Deferred to Phase 4 (WP-05).
- Streamlit application — Deferred to Phase 5 (WP-06).

</deferred>
