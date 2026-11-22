# Architecture

**Analysis Date:** 2026-11-22

## Pattern Overview

**Overall:** Modular Layered RAG Pipeline (Clean Architecture / Pipes and Filters)

**Key Characteristics:**
- **Local-First & Offline:** 100% CPU-executable without external networks, APIs, or cloud services.
- **Strict Provenance & Metadata Invariance:** Every chunk tracks immutable source IDs, 1-indexed physical page numbers, character offsets, and chunking parameters.
- **Fail-Safe Decoupling:** Ingestion failures on individual documents fail gracefully into structured error reports without halting batch processing.
- **Lightweight Dataclasses:** Minimal dependency footprint using standard Python dataclasses for all core schema representations.

## Layers

**1. Configuration Layer (`src/documind/config.py`):**
- Purpose: Defines immutable default parameters and operational limits for ingestion and chunking.
- Contains: `IngestionConfig` (frozen dataclass).
- Depends on: Python standard library only.
- Used by: `PDFLoader`, `BaseChunker`, `IngestionPipeline`.

**2. Domain Models & Schemas (`src/documind/ingestion/schemas.py`):**
- Purpose: Encapsulates pure domain representations and serialized transfer models.
- Contains: `DocumentPage`, `TextChunk`, `DocumentReport`, `IngestionReport`.
- Depends on: Python standard library (`dataclasses`, `typing`).
- Used by: All pipeline stages, tests, and CLI test scripts.

**3. Ingestion & Transformation Components (`src/documind/ingestion/`):**
- Purpose: Concrete processors for loading, cleaning, and partitioning document text.
- Contains:
  - `PDFLoader` (`src/documind/ingestion/pdf_loader.py`): Validates file headers (`%PDF-`), computes SHA-256 document IDs, and extracts text page-by-page via PyMuPDF.
  - `TextNormalizer` (`src/documind/ingestion/text_normalizer.py`): NFC normalization, control character stripping, CRLF normalization, and optional hyphen repair.
  - `BaseChunker`, `FixedSizeChunker`, `RecursiveCharacterChunker` (`src/documind/ingestion/chunker.py`): Character-based window and hierarchical separator chunking with exact interval page overlap tracking.
- Depends on: `config.py`, `schemas.py`, `pymupdf`.
- Used by: `IngestionPipeline`.

**4. Orchestration Layer (`src/documind/ingestion/pipeline.py`):**
- Purpose: Unified entry point for end-to-end batch or single-file processing.
- Contains: `IngestionPipeline` class coordinating loader, normalizer, chunker, and timing diagnostics.
- Depends on: All ingestion components and schemas.
- Used by: `scripts/test_ingestion.py`, test suites, and future retrieval modules.

## Data Flow

**Document Ingestion Flow:**
1. **File Input:** Caller supplies file path(s) (`Path` or `str`) to `IngestionPipeline.ingest_file()` or `ingest_files()`.
2. **Validation:** `PDFLoader.validate_pdf()` verifies file existence, size ($\le$ 50 MB), and `%PDF-` header signature.
3. **Hashing:** `PDFLoader.compute_document_id()` generates a cryptographic SHA-256 hash of the binary file content.
4. **Page Extraction:** `PDFLoader.load_pdf()` iterates 0-indexed page streams via PyMuPDF, maps to 1-indexed `DocumentPage` records, and detects blank/scanned pages.
5. **Normalization:** `TextNormalizer.normalize()` cleans text artifacts while preserving line structure and paragraphs.
6. **Chunking:** `BaseChunker.chunk_document()` concatenates pages into a unified text buffer, splits according to strategy (fixed or recursive), adjusts character offsets for stripped whitespace, and computes contributing page spans ($\max(C_{\text{start}}, P_{\text{start}}) < \min(C_{\text{end}}, P_{\text{end}})$).
7. **Reporting:** Generates `DocumentReport` and aggregated `IngestionReport` with execution statistics and latency metrics.

**State Management:**
- Stateless processing pipeline. Intermediate document representations are passed explicitly between pure functions and methods.

## Key Abstractions

**`BaseChunker` (`src/documind/ingestion/chunker.py`):**
- Purpose: Abstract base class enforcing consistent chunking contracts and unified page attribution math across chunking algorithms.
- Implementations: `FixedSizeChunker`, `RecursiveCharacterChunker`.
- Factory: `create_chunker(strategy, chunk_size, chunk_overlap, separators)`.

**`TextChunk` (`src/documind/ingestion/schemas.py`):**
- Purpose: Immutable representation of a retrieved document unit ready for downstream embedding, containing complete provenance (`document_id`, `chunk_id`, `page_numbers`, `char_start`, `char_end`).

## Entry Points

**Command-Line Runner:**
- Location: `scripts/test_ingestion.py`
- Invocation: `python scripts/test_ingestion.py [files...] [--chunk-size N] [--strategy {recursive,fixed}]`
- Responsibilities: Parses CLI flags, executes ingestion across target documents, and outputs a formatted execution summary.

**Python Module Import:**
- Location: `src/documind/ingestion/__init__.py`
- Exports: `IngestionPipeline`, `PDFLoader`, `TextNormalizer`, `FixedSizeChunker`, `RecursiveCharacterChunker`, `create_chunker`, `IngestionConfig`, `DocumentPage`, `TextChunk`, `DocumentReport`, `IngestionReport`.

## Error Handling

**Strategy:**
- Fail-safe isolation per document.
- Custom exception `PDFExtractionError` for unrecoverable errors.
- Non-fatal per-file errors (encrypted PDFs, corrupt headers, empty files) populate `report.success = False` and `report.error_message`, allowing remaining batch documents to finish without interruption.

---

*Architecture analysis: 2026-11-22*
*Update after structural changes*
