# Coding Conventions

**Analysis Date:** 2026-11-22

## Naming Patterns

**Files:**
- Snake case for all Python modules: `pdf_loader.py`, `text_normalizer.py`, `schemas.py`
- Test files prefixed with `test_`: `test_chunker.py`, `test_pdf_loader.py`
- Markdown files in `docs/` and `research/`: UPPER_SNAKE_CASE for major specs (`INGESTION_PIPELINE.md`), lowercase for standard templates

**Classes:**
- PascalCase for all classes: `PDFLoader`, `TextNormalizer`, `FixedSizeChunker`, `RecursiveCharacterChunker`, `DocumentPage`, `TextChunk`
- Exceptions end with `Error`: `PDFExtractionError`
- Abstract base classes prefixed with `Base`: `BaseChunker`

**Functions & Methods:**
- snake_case for all functions and methods: `compute_document_id()`, `validate_pdf()`, `split_text()`, `chunk_document()`
- Internal / private helper methods prefixed with a single underscore: `_split_text_with_separator()`, `_merge_splits()`
- Factory functions prefixed with `create_`: `create_chunker()`

**Variables & Constants:**
- Variables and attributes: snake_case (`chunk_size`, `char_start`, `page_number`)
- Constants: UPPER_SNAKE_CASE (`PDF_MAGIC_BYTES`)

**Data Models:**
- Standard Python `@dataclass` decorators
- Default factory patterns: `metadata: Dict[str, Any] = field(default_factory=dict)`
- Method `to_dict() -> Dict[str, Any]` provided via `dataclasses.asdict(self)`

## Code Style

**Formatting & Typing:**
- PEP 8 compliant Python style
- Explicit type annotations on all function signatures (`def ingest_file(self, file_path: Union[str, Path]) -> Tuple[List[TextChunk], DocumentReport]:`)
- Clean docstrings with Google-style `Args:` and `Returns:` sections
- Explicit typing imports from standard library `typing` (`List`, `Dict`, `Tuple`, `Optional`, `Union`, `Any`)

## Import Organization

**Order:**
1. Standard library imports (`import hashlib`, `import os`, `import time`, `from pathlib import Path`, `from typing import ...`)
2. External 3rd-party dependencies (`import pymupdf`, `import pytest`)
3. Internal project modules (`from documind.config import IngestionConfig`, `from documind.ingestion.schemas import ...`)

**Grouping:**
- Blank line separating each import group
- Explicit module imports, avoiding wildcard imports (`from module import *`)

## Error Handling

**Patterns:**
- Custom typed exceptions derived from `Exception`: `class PDFExtractionError(Exception): pass`
- Safe resource management with `try...finally`:
  ```python
  doc = None
  try:
      doc = pymupdf.open(file_path)
      # extraction logic
  finally:
      if doc is not None:
          doc.close()
  ```
- File validation guards before opening binary streams (`validate_pdf()` returns `(bool, Optional[str])`)
- Non-raising batch operations: document processing failures are captured in `DocumentReport.error_message` with `success=False` rather than crashing the pipeline

## Logging & Diagnostics

**Framework:**
- Structured metric tracking via dataclasses (`DocumentReport`, `IngestionReport`)
- Direct console output for CLI runners using standard library `print()` and `sys.stdout.reconfigure(encoding="utf-8")`
- Wall-clock benchmarking using high-resolution monotonic timer: `time.perf_counter()`

## Comments & Documentation

**When to Comment:**
- Algorithm and mathematical explanations (e.g. interval intersection formula `max(C_start, P_start) < min(C_end, P_end)`)
- Explaining character offset alignment adjustments for whitespace trimming
- Explicit docstrings at the top of every module explaining architectural purpose and invariants

---

*Conventions analysis: 2026-11-22*
*Update after style changes*
