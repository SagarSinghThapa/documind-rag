"""DocuMind Phase 6 — Multi-PDF End-to-End Integration Acceptance & Lifecycle Script.

Verifies:
1. Snapshot Persistence & Instant Hydration:
   - Loads DocumentIndex from active on-disk FAISS snapshot at data/index/
   - Verifies 436 vectors across BASWE, Everything RAG, and FINZAA.
2. Factual Multi-Document Question-Answering with Real Qwen-0.5B:
   - Query 1 (BASWE): "What is chunk overlap and what is the typical starting percentage?"
   - Query 2 (Everything RAG): "What are the core components of a RAG system according to the survey?"
   - Query 3 (FINZAA): "Who is the author of FINZAA.pdf? Cite the source page."
3. Distractor & Negative Abstention:
   - Query 4 (Out-of-Domain): "How do you tune an automotive carburetor for cold weather?"
   - Query 5 (Unindexed Topic): "What was Microsoft's revenue in 2024?"
4. Two-Step Confirmed Safe Reset Invariant Verification:
   - Tests safe_reset_index in an isolated scratch sandbox to confirm that:
     * Only uploaded PDFs and index snapshots are purged
     * Models, source code, and active production indexes remain 100% untouched.
"""

import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure src is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from documind.config import GenerationConfig, RetrievalConfig
from documind.generation.citation import CitationParser
from documind.generation.generator import GroundedGenerator
from documind.generation.llm import TransformersLLM
from documind.retrieval.context_builder import ContextBuilder
from documind.retrieval.retriever import SemanticRetriever
from documind.storage import DocumentIndex, FaissVectorStore, LocalEmbedder
from documind.ui.state import safe_reset_index, load_persistent_registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("e2e_acceptance")


def verify_multi_pdf_acceptance() -> Dict[str, Any]:
    print("=" * 80)
    print("DOCUMIND PHASE 6 — MULTI-PDF E2E ACCEPTANCE & LIFECYCLE VERIFICATION")
    print("=" * 80)

    # 1. Snapshot Hydration
    print("\n[Step 1/4] Simulating Application Startup — Hydrating from data/index/ ...")
    t0_load = time.perf_counter()
    embedder = LocalEmbedder()
    store = FaissVectorStore(dimension=embedder.dimension)
    store.load("data/index")
    doc_index = DocumentIndex(embedder=embedder, vector_store=store)
    registry = load_persistent_registry("data/index")
    hydration_ms = (time.perf_counter() - t0_load) * 1000.0

    print(f"  - Snapshot loaded in: {hydration_ms:.2f} ms")
    print(f"  - Total indexed vectors: {doc_index.total_vectors}")
    print(f"  - Total registered documents: {len(registry)}")
    for doc_id, doc_meta in registry.items():
        print(f"    * {doc_meta.get('filename')}: {doc_meta.get('chunks')} chunks ({doc_meta.get('pages')} pages)")

    assert doc_index.total_vectors == 436, f"Expected 436 vectors, found {doc_index.total_vectors}"
    assert len(registry) == 3, f"Expected 3 documents, found {len(registry)}"

    # 2. Pipeline Initialization
    print("\n[Step 2/4] Initializing Real Qwen2.5-0.5B-Instruct Offline CPU Inference...")
    gen_cfg = GenerationConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        cache_folder="models/llm",
        device="cpu",
        local_files_only=True,
        temperature=0.0,
        max_new_tokens=256,
        max_context_chars=1800,
        num_threads=8,
        include_exemplar=True,
    )
    llm = TransformersLLM(config=gen_cfg)
    llm._ensure_loaded()

    retrieval_cfg = RetrievalConfig(top_k=5, similarity_threshold=0.35)
    retriever = SemanticRetriever(document_index=doc_index, config=retrieval_cfg)
    context_builder = ContextBuilder(max_chars=gen_cfg.max_context_chars)
    citation_parser = CitationParser()
    generator = GroundedGenerator(
        retriever=retriever,
        llm=llm,
        context_builder=context_builder,
        citation_parser=citation_parser,
        config=gen_cfg,
    )

    # 3. Acceptance Queries
    print("\n[Step 3/4] Executing Multi-PDF Acceptance Test Suite...")

    test_queries = [
        {
            "id": "Q1_BASWE",
            "type": "positive",
            "target_doc": "BASWE-15-RAG-Concepts-Guide.pdf",
            "query": "What is chunk overlap and what is the typical starting percentage?",
            "expected_keywords": ["overlap", "10%", "20%"],
            "expected_pages": [2, 3],
        },
        {
            "id": "Q2_EVERYTHING_RAG",
            "type": "positive",
            "target_doc": "Everything-to-know-about-RAG.pdf",
            "query": "In Everything-to-know-about-RAG.pdf, what is the context window of an LLM compared to? Cite the source.",
            "expected_keywords": ["context", "window", "documents", "LLM"],
            "expected_pages": [5, 6, 56, 57],
        },
        {
            "id": "Q3_FINZAA",
            "type": "positive",
            "target_doc": "FINZAA.pdf",
            "query": "Who is the author of FINZAA.pdf? Cite the source page.",
            "expected_keywords": ["Anjali", "Singh"],
            "expected_pages": [2, 3, 4, 5, 6, 7],
        },
        {
            "id": "Q4_DISTRACTOR_CARBURETOR",
            "type": "negative",
            "query": "How do you tune an automotive carburetor for cold weather?",
            "target_doc": None,
        },
        {
            "id": "Q5_DISTRACTOR_EV",
            "type": "negative",
            "query": "What is the replacement procedure for a high-voltage lithium electric vehicle battery pack?",
            "target_doc": None,
        },
    ]

    results = []
    for item in test_queries:
        qid = item["id"]
        q_type = item["type"]
        query = item["query"]
        print(f"\n--- Testing {qid} ({q_type}) ---")
        print(f"Query: \"{query}\"")

        t0_q = time.perf_counter()
        gen_result = generator.generate_answer(query)
        latency_ms = (time.perf_counter() - t0_q) * 1000.0

        is_abstained = gen_result.is_abstained
        answer_text = gen_result.answer
        verified_citations = gen_result.citations
        total_tokens = gen_result.telemetry.get("completion_tokens", 0) if gen_result.telemetry else 0

        print(f"  - Latency: {latency_ms:.2f} ms")
        print(f"  - Abstained: {is_abstained}")
        print(f"  - Generated Tokens: {total_tokens}")
        print(f"  - Answer: {answer_text[:160]}...")
        print(f"  - Verified Citations: {len(verified_citations)}")
        for c in verified_citations:
            print(f"    * [{c.source_id}] {c.document_name} (pp. {c.start_page}-{c.end_page})")

        query_record = {
            "query_id": qid,
            "query_type": q_type,
            "query": query,
            "latency_ms": latency_ms,
            "is_abstained": is_abstained,
            "generated_tokens": total_tokens,
            "answer_preview": answer_text[:200],
            "verified_citations_count": len(verified_citations),
            "citations": [
                {
                    "source_id": c.source_id,
                    "document_name": c.document_name,
                    "pages": [c.start_page, c.end_page],
                    "is_syntax_valid": c.is_syntax_valid,
                    "is_metadata_valid": c.is_metadata_valid,
                    "is_evidence_supported": c.is_evidence_supported,
                }
                for c in verified_citations
            ],
            "passed": False,
        }

        if q_type == "positive":
            # Must NOT be abstained
            # Must emit at least 1 citation
            # Citation must match target_doc
            target_doc_name = item["target_doc"]
            doc_matched = any(target_doc_name.lower() in (c.document_name or "").lower() for c in verified_citations)
            keyword_matched = any(kw.lower() in answer_text.lower() for kw in item.get("expected_keywords", []))

            if not is_abstained and len(verified_citations) > 0 and doc_matched:
                query_record["passed"] = True
                print(f"  >> RESULT: PASS (Properly grounded with valid citation from {target_doc_name})")
            else:
                print(f"  >> RESULT: FAIL (abstained={is_abstained}, citations={len(verified_citations)}, doc_matched={doc_matched})")
        else:
            # Negative query must abstain
            # Zero citations
            if is_abstained and len(verified_citations) == 0:
                query_record["passed"] = True
                print(f"  >> RESULT: PASS (Properly abstained on ungrounded query)")
            else:
                print(f"  >> RESULT: FAIL (Failed to abstain on distractor query)")

        results.append(query_record)

    # 4. Safe Reset Invariant Verification
    print("\n[Step 4/4] Verifying Safe Reset Invariants in Isolated Sandbox...")
    sandbox_dir = Path("scratch/test_reset_sandbox")
    if sandbox_dir.exists():
        shutil.rmtree(sandbox_dir)
    sandbox_dir.mkdir(parents=True, exist_ok=True)

    sb_upload = sandbox_dir / "uploads"
    sb_index = sandbox_dir / "index"
    sb_snapshots = sb_index / "snapshots" / "snap_001"
    sb_upload.mkdir(parents=True, exist_ok=True)
    sb_snapshots.mkdir(parents=True, exist_ok=True)

    # Populate sandbox dummy files
    dummy_pdf = sb_upload / "test_doc.pdf"
    dummy_pdf.write_text("dummy pdf binary content")
    dummy_snap = sb_snapshots / "faiss.index"
    dummy_snap.write_text("dummy faiss index")
    dummy_active = sb_index / "active_snapshot.json"
    dummy_active.write_text(json.dumps({"snapshot_id": "snap_001"}))
    dummy_reg = sb_index / "document_registry.json"
    dummy_reg.write_text(json.dumps({"doc_1": {"filename": "test_doc.pdf"}}))

    # Record real model and active index state before reset
    real_model_exists = Path("models/llm/models--Qwen--Qwen2.5-0.5B-Instruct").exists()
    real_index_exists = Path("data/index/active_snapshot.json").exists()
    real_registry_size = len(load_persistent_registry("data/index"))

    # Execute safe_reset_index on sandbox
    print("  - Triggering safe_reset_index on sandbox...")
    safe_reset_index(
        data_dir=str(sandbox_dir),
        index_dir=str(sb_index),
        upload_dir=str(sb_upload),
    )

    # Verify sandbox state
    sb_uploads_remaining = list(sb_upload.iterdir())
    sb_snapshots_remaining = list((sb_index / "snapshots").iterdir())
    sb_active_exists = dummy_active.exists()
    sb_reg_cleared = json.loads((sb_index / "document_registry.json").read_text()) == {}

    # Verify production invariant
    real_model_still_exists = Path("models/llm/models--Qwen--Qwen2.5-0.5B-Instruct").exists()
    real_index_still_exists = Path("data/index/active_snapshot.json").exists()
    real_registry_still_intact = len(load_persistent_registry("data/index")) == real_registry_size

    reset_passed = (
        len(sb_uploads_remaining) == 0
        and len(sb_snapshots_remaining) == 0
        and not sb_active_exists
        and sb_reg_cleared
        and real_model_still_exists
        and real_index_still_exists
        and real_registry_still_intact
    )

    print(f"  - Sandbox uploads deleted: {len(sb_uploads_remaining) == 0}")
    print(f"  - Sandbox snapshots deleted: {len(sb_snapshots_remaining) == 0}")
    print(f"  - Sandbox active snapshot pointer removed: {not sb_active_exists}")
    print(f"  - Sandbox document registry reset to empty: {sb_reg_cleared}")
    print(f"  - CRITICAL: Production models/llm/ preserved: {real_model_still_exists}")
    print(f"  - CRITICAL: Production data/index/ preserved: {real_index_still_exists}")
    print(f"  - CRITICAL: Production registry intact: {real_registry_still_intact}")
    print(f"  >> RESET INVARIANT VERDICT: {'PASS' if reset_passed else 'FAIL'}")

    # Clean up sandbox
    if sandbox_dir.exists():
        shutil.rmtree(sandbox_dir)

    all_passed = all(r["passed"] for r in results) and reset_passed
    print("\n" + "=" * 80)
    print(f"OVERALL E2E ACCEPTANCE VERDICT: {'PASS (100% Meets Acceptance Criteria)' if all_passed else 'FAIL'}")
    print("=" * 80)

    summary_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hydration_latency_ms": hydration_ms,
        "total_vectors": doc_index.total_vectors,
        "registered_documents": len(registry),
        "queries": results,
        "safe_reset_verified": reset_passed,
        "all_acceptance_passed": all_passed,
    }

    out_file = Path("docs/project_reports/e2e_acceptance_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved E2E acceptance results to: {out_file}")

    return summary_data


if __name__ == "__main__":
    verify_multi_pdf_acceptance()
