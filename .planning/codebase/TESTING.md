# Testing Patterns

**Analysis Date:** 2026-11-22

## Test Framework

**Runner:**
- `pytest` (minimum version `>=8.0.0`, tested with `pytest 9.1.1`)
- Run without third-party plugins or heavy runners

**Assertion Library:**
- Standard Python `assert` statements enriched by pytest introspection

**Run Commands:**
```bash
pytest                                   # Run all 42 tests
pytest tests/test_chunker.py             # Run chunker tests only
pytest tests/test_pdf_loader.py          # Run PDFLoader tests only
pytest tests/test_ingestion_pipeline.py  # Run end-to-end integration tests
pytest tests/test_text_normalizer.py     # Run text normalizer tests
pytest -v -k "regression"                # Run regression tests only
```

## Test File Organization

**Location:**
- Dedicated `tests/` directory at the repository root matching the package structure

**Naming:**
- Files: `test_<module_name>.py`
- Test functions: `test_<feature_or_behavior>_<expected_outcome>()`

**Structure:**
```
tests/
├── test_chunker.py             # Tests FixedSizeChunker, RecursiveCharacterChunker, offsets, page spans
├── test_pdf_loader.py          # Tests PDF loading, magic bytes, hashing, encryption, blank pages
├── test_text_normalizer.py     # Tests Unicode NFC, de-hyphenation, control chars, whitespace
└── test_ingestion_pipeline.py  # End-to-end pipeline execution with mock and real PDFs
```

## Test Suite Invariants

The test suite enforces 42 passing tests covering the following critical guarantees:

1. **Page Numbering Invariance:**
   - 1-indexed page preservation verified across multi-page documents.
   - Sequential page ordering without skipped indices for non-empty pages.

2. **Deterministic Hashing:**
   - Document ID is an exact SHA-256 hash of the binary file content.
   - Identical inputs produce byte-identical chunk IDs.

3. **Character Offset Alignment:**
   - Slicing `unified_text[char_start:char_end]` must match `chunk.text` with 100% precision.
   - Whitespace stripping must adjust `char_start` and `char_end` so no offset drift occurs.

4. **Strict Chunk Size Bounds:**
   - In recursive chunking, chunk length must not exceed `chunk_size` characters, even on uneven separator splits.

5. **Cross-Page Page Attribution:**
   - Mathematical interval overlap ($\max(C_{\text{start}}, P_{\text{start}}) < \min(C_{\text{end}}, P_{\text{end}})$) accurately records all contributing page numbers in `page_numbers: List[int]`.

6. **Error Isolation:**
   - Corrupted PDF headers, 0-byte files, and password-protected files fail gracefully with descriptive error reports without throwing unhandled exceptions.

## Mocking & Fixtures

**Pytest Fixtures:**
- Temporary directories via pytest built-in `tmp_path` fixture.
- Synthetic PDF generation using PyMuPDF in-memory or temporary files for deterministic test cases.
- Synthetic `DocumentPage` records for unit testing chunkers in isolation without needing actual PDF files on disk.

```python
# Example synthetic page fixture pattern
def create_sample_page(page_num: int, text: str, doc_id: str = "test_doc") -> DocumentPage:
    return DocumentPage(
        document_id=doc_id,
        document_name=f"{doc_id}.pdf",
        source_path=f"/fake/{doc_id}.pdf",
        page_number=page_num,
        text=text,
        metadata={}
    )
```

---

*Testing analysis: 2026-11-22*
*Update after test framework changes*
