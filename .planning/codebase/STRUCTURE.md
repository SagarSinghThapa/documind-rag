# Codebase Structure

**Analysis Date:** 2026-11-22

## Directory Layout

```
D:/rag/
├── configs/                     # Environment and pipeline configuration files (.gitkeep)
├── data/                        # Datasets and processed artifacts
│   ├── evaluation/              # Benchmark questions, ground-truth answers (.gitkeep)
│   ├── processed/               # Extracted chunks, normalized document outputs (.gitkeep)
│   └── raw/                     # Raw unparsed input documents (.gitkeep)
├── docs/                        # Architectural and engineering documentation
│   ├── architecture/            # Baseline RAG and Ingestion pipeline specifications
│   ├── decisions/               # Architecture Decision Records (ADRs) (.gitkeep)
│   ├── evaluation/              # Evaluation methodology and metrics docs
│   ├── experiments/             # Experiment logs and ablation study notes (.gitkeep)
│   ├── findings/                # Empirical research conclusions (.gitkeep)
│   └── project_reports/         # Work package audits and milestone summaries
├── notebooks/                   # Jupyter exploration and benchmark notebooks (.gitkeep)
├── research/                    # Literature reviews and technical references
│   ├── legacy_references/       # Historical baseline reference PDFs and archives
│   ├── literature_notes/        # Literature reviews, research gap register, comparison matrices
│   ├── papers/                  # Research paper inventory and catalogue
│   ├── technical_references/    # Foundational RAG technical reference PDFs
│   └── tutorials/               # Exploratory Jupyter tutorial notebooks
├── resumes/                     # 25 sample candidate resumes for screening tests (gitignored)
├── scripts/                     # Operational test and utility scripts
│   └── test_ingestion.py        # CLI smoke test runner for ingestion pipeline
├── src/                         # Production application source code
│   └── documind/                # Core DocuMind Python package
│       ├── config.py            # Global and ingestion configuration dataclasses
│       └── ingestion/           # Document ingestion, normalization, and chunking subsystem
│           ├── chunker.py       # Fixed and recursive chunking implementations
│           ├── pdf_loader.py    # PyMuPDF-based PDF text and metadata extractor
│           ├── pipeline.py      # IngestionPipeline orchestrator
│           ├── schemas.py       # DocumentPage, TextChunk, and report dataclasses
│           └── text_normalizer.py # Conservative text cleaner
├── tests/                       # Automated test suite
│   ├── test_chunker.py          # Unit & regression tests for chunking algorithms
│   ├── test_ingestion_pipeline.py # End-to-end integration tests for IngestionPipeline
│   ├── test_pdf_loader.py       # Unit tests for PDFLoader extraction and validation
│   └── test_text_normalizer.py  # Unit tests for text normalization
├── .gitignore                   # Version control ignore rules (resumes, caches, venv)
├── README.md                    # Project overview, engineering principles, roadmap
└── requirements.txt             # Minimal local-first Python dependencies
```

## Directory Purposes

**`src/documind/`:**
- Purpose: Core application package containing production code.
- Contains: Python modules organized by functional subsystem (`ingestion/`, future `retrieval/`, `generation/`).
- Key files: `config.py`, `ingestion/pipeline.py`, `ingestion/schemas.py`.

**`tests/`:**
- Purpose: Unit, integration, and regression test suite executed via `pytest`.
- Contains: `test_*.py` files testing each corresponding module in `src/documind/`.
- Key files: `test_chunker.py`, `test_pdf_loader.py`, `test_ingestion_pipeline.py`, `test_text_normalizer.py`.

**`docs/`:**
- Purpose: Living documentation, specifications, architecture audits, and evaluation frameworks.
- Key files: `docs/architecture/BASELINE_RAG_SPECIFICATION.md`, `docs/architecture/INGESTION_PIPELINE.md`, `docs/project_reports/WP_02_INDEPENDENT_AUDIT.md`.

**`research/`:**
- Purpose: Rigorous literature notes, technical paper catalogues, and comparison matrices supporting the hypothesis-driven methodology.
- Key files: `research/literature_notes/LITERATURE_REVIEW.md`, `research/literature_notes/TECHNICAL_COMPARISON_MATRIX.md`.

**`scripts/`:**
- Purpose: Runnable CLI utilities and smoke-testing scripts.
- Key files: `scripts/test_ingestion.py`.

**`resumes/`:**
- Purpose: 25 representative resume PDFs used for real-world document ingestion testing.
- Privacy Note: Kept strictly local, ignored in `.gitignore`.

## Key File Locations

**Entry Points:**
- `scripts/test_ingestion.py`: Command-line interface for smoke testing ingestion and chunking.
- `src/documind/ingestion/__init__.py`: Package export interface for programmatic consumers.

**Configuration:**
- `src/documind/config.py`: Default constants (`IngestionConfig`).
- `requirements.txt`: Package dependency versions.

**Core Logic:**
- `src/documind/ingestion/pipeline.py`: Orchestration of loader $\rightarrow$ normalizer $\rightarrow$ chunker.
- `src/documind/ingestion/chunker.py`: Text chunking algorithms and page mapping mathematics.
- `src/documind/ingestion/pdf_loader.py`: Binary PDF extraction via PyMuPDF.
- `src/documind/ingestion/text_normalizer.py`: Regex and Unicode NFC normalization.

**Testing:**
- `tests/test_chunker.py`: Tests fixed and recursive chunkers, character offset slicing, and upper-bound constraints.
- `tests/test_pdf_loader.py`: Tests encryption handling, empty page detection, and file validation.

## Naming Conventions

**Files:**
- Snake case for all Python modules: `pdf_loader.py`, `text_normalizer.py`.
- Test files prefixed with `test_`: `test_chunker.py`.
- All-caps markdown files for top-level documentation: `README.md`, `BASELINE_RAG_SPECIFICATION.md`.

**Directories:**
- Lowercase snake_case: `documind`, `ingestion`.

**Classes & Functions:**
- Classes: PascalCase (`PDFLoader`, `RecursiveCharacterChunker`, `DocumentPage`).
- Functions and methods: snake_case (`ingest_file`, `compute_document_id`).
- Constants: UPPER_SNAKE_CASE (`PDF_MAGIC_BYTES`).

## Where to Add New Code

**New Retrieval Component (e.g., Embedder / VectorStore):**
- Implementation: `src/documind/retrieval/embedder.py`, `src/documind/retrieval/vector_store.py`
- Schemas: `src/documind/retrieval/schemas.py`
- Configuration: `src/documind/config.py` (add `RetrievalConfig`)
- Tests: `tests/test_embedder.py`, `tests/test_vector_store.py`

**New Generation Component (e.g., Local LLM Generator / Citation Formatter):**
- Implementation: `src/documind/generation/generator.py`, `src/documind/generation/citations.py`
- Tests: `tests/test_generator.py`

**New CLI Script:**
- Implementation: `scripts/run_rag.py` or `scripts/evaluate.py`

---

*Structure analysis: 2026-11-22*
*Update after adding new directories or moving key files*
