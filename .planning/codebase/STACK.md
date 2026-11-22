# Technology Stack

**Analysis Date:** 2026-11-22

## Languages

**Primary:**
- Python 3 (Tested on 3.11.9 and 3.13.1) — All application logic, extraction pipeline, chunking algorithms, data schemas, and test suites (`src/documind/`, `tests/`, `scripts/`)

**Secondary:**
- Markdown — Specifications, architectural decision records, literature reviews, and research catalogues (`docs/`, `research/`)

## Runtime

**Environment:**
- CPython 3.11+ / 3.13+ on Windows x86_64, Linux, or macOS
- Target hardware profile: Intel Core i5-1135G7 (4 cores / 8 threads), 24 GB RAM, CPU-only local execution

**Package Manager:**
- `pip`
- Lockfile / Manifest: `requirements.txt` present

## Frameworks

**Core:**
- Standard Python Standard Library — `dataclasses`, `hashlib`, `pathlib`, `unicodedata`, `re`, `abc`
- PyMuPDF (`pymupdf>=2.1.0`) — C-level MuPDF bindings (`fitz`) for fast PDF page parsing, layout preservation, and document validation

**Testing:**
- `pytest>=8.0.0` — Test discovery, parameterized test fixtures, regression assertions

**Build/Dev:**
- Standard Python packaging (`setuptools`)

## Key Dependencies

**Critical:**
- `pymupdf` (v2.1.0+) — Core extraction engine. Delivers 14x–19x faster parsing than `pypdf`, layout-aware text streams, PDF header validation, and native detection of image-only/scanned documents without external DLL dependencies (`src/documind/ingestion/pdf_loader.py`).
- `pytest` (v8.0.0+) — Full automated test runner for unit, integration, and regression tests (`tests/`).

**Infrastructure:**
- Zero external infrastructure or cloud dependencies. Operates 100% offline and local-first.

## Configuration

**Environment:**
- Completely self-contained; no `.env` files or external credentials required.
- ₹0 API budget constraint: no OpenAI, Anthropic, or paid cloud APIs.

**Configuration Models:**
- `IngestionConfig` dataclass (`src/documind/config.py`):
  - `chunk_size`: 1000 characters
  - `chunk_overlap`: 150 characters
  - `chunking_strategy`: "recursive" (or "fixed")
  - `separators`: `["\n\n", "\n", ". ", " ", ""]`
  - `max_file_size_bytes`: 52,428,800 bytes (50 MB)
  - `supported_extensions`: `[".pdf"]`
  - `extract_empty_pages`: `False`

## Platform Requirements

**Development:**
- OS: Windows 10/11, Linux, or macOS
- Python 3.11 or higher
- Git repository tracking

**Production:**
- Offline local CPU execution. Zero outbound network transmission required.

---

*Stack analysis: 2026-11-22*
*Update after major dependency changes*
