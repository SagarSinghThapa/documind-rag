"""Command-line smoke test script for DocuMind PDF Ingestion & Chunking Pipeline.

Usage:
    python scripts/test_ingestion.py [path_to_pdf_1] [path_to_pdf_2] ...
    python scripts/test_ingestion.py --chunk-size 800 --chunk-overlap 100
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add src to sys.path so documind can be imported without installation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from documind.ingestion.pipeline import IngestionPipeline


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="DocuMind PDF Ingestion & Chunking Smoke Test Runner"
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="One or more PDF files to ingest. If omitted, uses public research PDFs.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size in characters (default: 1000).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Chunk overlap in characters (default: 150).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="recursive",
        choices=["recursive", "fixed"],
        help="Chunking strategy ('recursive' or 'fixed', default: 'recursive').",
    )

    args = parser.parse_args()

    # Default to public research PDFs if none specified
    repo_root = Path(__file__).resolve().parent.parent
    if not args.files:
        default_pdfs = [
            repo_root / "research" / "technical_references" / "BASWE-15-RAG-Concepts-Guide.pdf",
            repo_root / "research" / "technical_references" / "Everything-to-know-about-RAG.pdf",
        ]
        target_files = [p for p in default_pdfs if p.exists()]
        if not target_files:
            print("Error: No PDF files specified and default research PDFs were not found.")
            sys.exit(1)
    else:
        target_files = [Path(f) for f in args.files]

    print("=" * 70)
    print("DocuMind Ingestion Pipeline — Smoke Test Runner")
    print("=" * 70)
    print(f"Target files ({len(target_files)}):")
    for f in target_files:
        print(f"  • {f}")
    print(f"Configuration: Strategy={args.strategy}, Size={args.chunk_size} chars, Overlap={args.chunk_overlap} chars\n")

    pipeline = IngestionPipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunking_strategy=args.strategy,
    )

    t0 = time.perf_counter()
    chunks, report = pipeline.ingest_files(target_files)
    total_time = time.perf_counter() - t0

    print("-" * 70)
    print("Ingestion Execution Summary:")
    print("-" * 70)
    print(f"  Documents Submitted:  {report.documents_submitted}")
    print(f"  Documents Processed:  {report.documents_processed}")
    print(f"  Documents Failed:     {report.documents_failed}")
    print(f"  Total Pages Extracted:{report.pages_extracted}")
    print(f"  Empty Pages Detected: {report.empty_pages}")
    print(f"  Total Chunks Created: {report.chunks_generated}")
    print(f"  Wall-Clock Duration:  {total_time:.3f} seconds")
    print("-" * 70)

    if report.errors:
        print("\nWarnings / Errors Encountered:")
        for err in report.errors:
            print(f"  [{err['type'].upper()}] {err['file']}: {err['message']}")

    print("\nDocument-by-Document Breakdown:")
    for d_rep in report.document_reports:
        status_sym = "✓" if d_rep.success else "✗"
        print(f"  [{status_sym}] {d_rep.document_name}: {d_rep.pages_extracted} pages, {d_rep.chunks_generated} chunks ({d_rep.extraction_time_sec*1000:.1f} ms)")

    if chunks:
        print("\n" + "=" * 70)
        print("Sample Generated Chunks Preview (First 3 Chunks):")
        print("=" * 70)
        for i, c in enumerate(chunks[:3]):
            print(f"\n--- Chunk {i+1} [ID: {c.chunk_id}] ---")
            print(f"Document: {c.document_name} | Doc ID: {c.document_id[:12]}...")
            print(f"Pages: {c.page_numbers} (Start: p.{c.start_page}, End: p.{c.end_page})")
            print(f"Length: {len(c.text)} characters | Multi-page span: {c.metadata.get('is_multipage', False)}")
            snippet = c.text[:180].replace("\n", " ")
            print(f"Excerpt: \"{snippet}...\"")

    print("\n" + "=" * 70)
    print("Smoke Test Complete.")
    print("=" * 70)

    if report.documents_failed > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
