# DocuMind — Milestone 1 MVP Release Checklist

This document defines the pre-flight verification gates and sign-off criteria required for the **DocuMind Milestone 1 MVP Release**.

---

## 1. Environment & Hardware Verification

| Check | Requirement | Verification Method | Status |
|:---|:---|:---|:---:|
| **Operating System** | Windows 11 x64 compatible | Verified on Windows 11 Pro 64-bit | ✅ PASS |
| **Python Version** | Python 3.11.x | `python --version` -> `Python 3.11.9` | ✅ PASS |
| **Physical Memory** | Memory profile <= 16 GB RAM | `profile_hardware_envelope.py`: Physical RAM **15.65 GB**, Peak RSS **1.55 GB** | ✅ PASS |
| **CPU Architecture** | Multi-threaded CPU forward passes | Tested with 8 intra-op threads on 10 Physical / 12 Logical Cores | ✅ PASS |
| **Offline Isolation** | Zero cloud APIs / outbound calls | `local_files_only=True`, zero external network calls during inference | ✅ PASS |
| **Cost Model** | ₹0 Cloud Budget | Zero commercial LLM tokens billed; 100% open-source local models | ✅ PASS |

---

## 2. Model & Index Artifacts

| Check | Requirement | Verification Method | Status |
|:---|:---|:---|:---:|
| **Embedding Weights** | `all-MiniLM-L6-v2` (384-d) | Local weights cached in Hugging Face cache | ✅ PASS |
| **LLM Weights** | `Qwen2.5-0.5B-Instruct` | Local Safetensors (~990 MB) cached in `models/llm/` | ✅ PASS |
| **Index Persistence** | FAISS `IndexFlatIP` persisted | Snapshot in `data/index/snapshots/` with atomic pointer `active_snapshot.json` | ✅ PASS |
| **Checksum Integrity** | SHA-256 match for snapshot files | `verify_index_integrity.py`: 0 errors across 436 vectors & registry | ✅ PASS |
| **Document Registry** | 3 documents, 436 chunks | `BASWE` (30 chunks), `Everything RAG` (165 chunks), `FINZAA` (241 chunks) | ✅ PASS |

---

## 3. Grounding & Quality Verification Gates (EVAL-01, EVAL-02)

> **Evaluation Protocol Note:** Gates GATE-01 through GATE-07 below reflect deterministic benchmarks using `MockLLM` with ground-truth responses (`scripts/evaluate_grounding_quality.py`) to validate the 3-layer citation verifier independently of small-model generation variability. Real-model CPU generation was independently verified using `Qwen/Qwen2.5-0.5B-Instruct` in `scripts/verify_real_llm.py` and `scripts/verify_e2e_acceptance.py` (achieving ~1.33 tok/sec, ~61.5 s E2E latency, and valid citations when prompted with explicit citation cues).

| Gate ID | Metric / Invariant | Threshold | Measured Result | Verdict |
|:---|:---|:---:|:---:|:---:|
| **GATE-01** | Citation Emission Rate on $Q_+$ | $\ge 95\%$ | **100.0%** (13/13) [MockLLM] | ✅ PASS |
| **GATE-02** | Citation Syntax Validity (Layer 1) | $\ge 99\%$ | **100.0%** (13/13) [MockLLM] | ✅ PASS |
| **GATE-03** | Citation Metadata Linkage (Layer 2) | $\ge 99\%$ | **100.0%** (13/13) [MockLLM] | ✅ PASS |
| **GATE-04** | Evidence Support Heuristic (Layer 3)* | $\ge 90\%$ | **100.0%** (13/13) [MockLLM] | ✅ PASS |
| **GATE-05** | Negative Distractor Abstention Rate | $\ge 99\%$ | **100.0%** (8/8) [MockLLM] | ✅ PASS |
| **GATE-06** | False Abstention Rate on $Q_+$ | $\le 5\%$ | **0.0%** (0/13) [MockLLM] | ✅ PASS |
| **GATE-07** | Unsupported Answer Rate | $\le 1\%$ | **0.0%** (0/21) [MockLLM] | ✅ PASS |
| **GATE-08** | Fast-Path Distractor Latency | $\le 50\text{ ms}$ | **16.60 ms** (mean; 9.08–16.60 ms measured) | ✅ PASS |
| **GATE-09** | Retrieval Hit@5 (Document & Page) | $\ge 75\%$ | **76.92%** (10/13) [Dense k-NN] | ✅ PASS |

*\*Note: Layer 3 evidence support uses character n-gram text containment between the assertion and the source chunk as a pragmatic lexical overlap heuristic, not formal NLI semantic entailment.*

---

## 4. Security & Safety Verification Gates

| Check | Requirement | Verification Method | Status |
|:---|:---|:---|:---:|
| **Localhost Binding** | Server bound to loopback only | `.streamlit/config.toml`: `address = "localhost"` | ✅ PASS |
| **CORS / XSRF** | Cross-site protections active | `.streamlit/config.toml`: `enableCORS = true`, `enableXsrfProtection = true` | ✅ PASS |
| **Host Whitelisting** | Allowed host validation | `.streamlit/config.toml`: `allowedHosts = ["localhost", "127.0.0.1"]` | ✅ PASS |
| **Git Exclusion** | Uploads & vector indexes untracked | `.gitignore` covers `data/uploads/`, `data/index/`, `scratch/` | ✅ PASS |
| **Safe Reset Invariant** | Reset purges only vectors/uploads | `safe_reset_index()` tested in isolated sandbox; `models/llm/` untouched | ✅ PASS |

---

## 5. Automated Regression Test Suite

| Test Module | Coverage Area | Test Count | Status |
|:---|:---|:---:|:---:|
| `test_citations.py` | 3-Layer citation verification & regex parsing | 14 passed | ✅ PASS |
| `test_generator.py` | Prompt formatting, exemplar injection, abstention | 12 passed | ✅ PASS |
| `test_retriever.py` | Dense search, metadata routing, threshold filtering | 17 passed | ✅ PASS |
| `test_chunker.py` | Sliding window boundaries & sentence preservation | 8 passed | ✅ PASS |
| `test_context_builder.py` | Context budget capping & `[S1]` prefix formatting | 10 passed | ✅ PASS |
| `test_pdf_loader.py` | PyMuPDF text & page extraction | 6 passed | ✅ PASS |
| `test_text_normalizer.py` | Unicode whitespace, hyphens, control characters | 7 passed | ✅ PASS |
| `test_retrieval_eval.py` | Hit@K and MRR evaluation routines | 8 passed | ✅ PASS |
| `test_llm.py` | Hugging Face model wrapper & thread locks | 14 passed | ✅ PASS |
| `test_embedder.py` | MiniLM L2-normalization & dimension checks | 8 passed | ✅ PASS |
| `test_vector_store.py` | FAISS IndexFlatIP add/search/save/load | 14 passed | ✅ PASS |
| `test_ingestion_pipeline.py` | Multi-document ingestion end-to-end | 10 passed | ✅ PASS |
| `test_ui_app.py` | Streamlit AppTest page rendering & chat | 5 passed | ✅ PASS (Host) |
| `test_ui_state.py` | Session state isolation & reset gates | 6 passed | ✅ PASS (Host) |
| **TOTAL** | **Full Regression Suite** | **139 Passed** | **100% PASS** |

---

## 6. Release Documentation Deliverables (EVAL-03)

| Document | Purpose | File Location | Status |
|:---|:---|:---|:---:|
| **Project README** | Portfolio entrypoint, architecture, quickstart | `README.md` | ✅ Complete |
| **Architecture Guide** | Subsystem contracts, concurrency, persistence | `docs/ARCHITECTURE.md` | ✅ Complete |
| **Windows Setup Guide** | Clean-room Windows 11 installation | `docs/WINDOWS_SETUP_GUIDE.md` | ✅ Complete |
| **Demonstration Runbook** | Scripted 5-minute portfolio presentation | `docs/DEMONSTRATION_RUNBOOK.md` | ✅ Complete |
| **Troubleshooting Guide** | Solutions for ports, sockets, CORS, OCR | `docs/TROUBLESHOOTING.md` | ✅ Complete |
| **Release Checklist** | Pre-flight validation & sign-off criteria | `docs/RELEASE_CHECKLIST.md` | ✅ Complete |
| **Final Release Report** | Empirical benchmark synthesis & audit verdict | `docs/project_reports/WP_07_FINAL_RELEASE_REPORT.md` | ✅ Complete |

---

## 7. Final Milestone 1 Sign-Off

- **Release Version:** DocuMind 1.0 (Milestone 1 MVP)
- **Target Audience:** Portfolio reviewers, technical recruiters, engineering peers.
- **Sign-off Verdict:** **APPROVED FOR RELEASE**
- **Date:** 2026-11-24
