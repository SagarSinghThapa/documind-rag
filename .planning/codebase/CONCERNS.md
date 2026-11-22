# Codebase Concerns

**Analysis Date:** 2026-11-22

## Tech Debt

**Virtual Environment & Host Python Dependencies:**
- Issue: No virtual environment (`.venv`) is currently instantiated in `D:\rag`. The host system's default global Python 3.11 environment does not have `pymupdf` or `pytest` installed.
- Why: Code was previously developed/verified in a Python 3.13 environment (`C:\Users\Asus\...`), and the repository was moved into `D:\rag`.
- Impact: Running `pytest` or `python scripts/test_ingestion.py` fails with `ModuleNotFoundError: No module named 'pymupdf'`.
- Fix approach: Create a local virtual environment (`python -m venv .venv`), activate it, and install dependencies (`pip install -r requirements.txt`).

**Untracked Git Repository State:**
- Issue: The repository is on `master` with no commits yet; all initial files are untracked.
- Why: Fresh project repository setup.
- Impact: GSD workflows and branching mechanisms rely on an existing HEAD commit base.
- Fix approach: Create an initial commit tracking the baseline codebase, documentation, and `.planning` directory.

## Known Bugs

**Character-to-Token Ratio Assumptions:**
- Symptoms: Chunk sizes are measured in characters (`chunk_size: 1000`), which represents an approximation of 200–250 tokens in English prose.
- Trigger: Processing documents with dense technical formulas, hexadecimal hashes, or code snippets where the character-to-token ratio drops significantly.
- Workaround: Default chunk size is conservatively set to 1000 characters, well within the 512-token context limits of common embedding models (e.g., `all-MiniLM-L6-v2` or `bge-small-en-v1.5`).
- Root cause: By design, DocuMind deliberately avoids heavy tokenizer dependencies (`tiktoken`, `transformers`) in the ingestion stage to maintain minimal local dependencies.

## Security Considerations

**Candidate Resume Privacy:**
- Risk: Ingestion of real candidate resumes containing Personally Identifiable Information (PII) such as phone numbers, addresses, and email addresses.
- Current mitigation: The `resumes/` folder is explicitly excluded from git tracking via `.gitignore`. Processing is 100% local with zero outbound network calls.
- Recommendations: Maintain strict local-only policy; never commit or push raw resume files to remote git repositories.

**PDF Parsing Vulnerabilities & Malformed Files:**
- Risk: Malformed or malicious PDF files causing memory corruption or infinite loops in PDF parsers.
- Current mitigation: PyMuPDF operates with memory-safe C-bindings, enclosed in `try...finally: doc.close()` blocks. File size is strictly capped at 50 MB (`max_file_size_bytes`). Header signature (`%PDF-`) is validated prior to parsing.

## Performance Bottlenecks

**Large PDF Extraction:**
- Problem: Very large PDFs (>500 pages) could increase wall-clock extraction time.
- Measurement: Benchmark on 92-page PDF showed 840 ms total latency (9.14 ms/page) using PyMuPDF (compared to 16,418 ms with `pypdf`).
- Cause: Single-threaded CPU page extraction.
- Improvement path: PyMuPDF is already 14x–19x faster than standard Python parsers. If multi-hundred page documents are ingested in bulk, batch processing across Python `multiprocessing.Pool` can be added to `IngestionPipeline.ingest_files()`.

## Fragile Areas

**Whitespace Trimming and Character Offset Sync:**
- Why fragile: Slicing `unified_text[char_start:char_end]` must strictly match `chunk.text`. If chunk text is stripped of whitespace without adjusting `char_start` and `char_end`, character offsets drift.
- Common failures: Offsets include leading/trailing whitespace, causing length mismatches between sliced text and chunk text.
- Safe modification: Any modification to `split_text()` or `chunk_document()` in `src/documind/ingestion/chunker.py` must preserve the leading and trailing whitespace offset adjustments (`adj_start = c_start + leading_ws`, `adj_end = c_end - trailing_ws`).
- Test coverage: Enforced by regression test `test_char_offsets_exact_substring_match` in `tests/test_chunker.py`.

## Missing Critical Features (Pending Roadmap Phases)

**Downstream RAG Subsystems:**
- Only Phase 2 (Document Ingestion & Chunking) is implemented. The following phases are defined in `docs/architecture/BASELINE_RAG_SPECIFICATION.md` but pending implementation:
  - Phase 3: Embedding Model Selection & Generation (Local dense embeddings).
  - Phase 4: Local Vector Storage & Indexing (e.g., FAISS / ChromaDB).
  - Phase 5: Vector Retrieval & Relevance Evaluation.
  - Phase 6: Grounded Generation with Citation Verification (Local LLM via llama.cpp or Ollama).
  - Phase 7: Interactive Interface & End-to-End Evaluation.

---

*Concerns analysis: 2026-11-22*
*Update after resolving concerns or discovering new issues*
