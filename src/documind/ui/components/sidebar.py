"""Sidebar component: Document upload, registry management, two-step reset, and telemetry."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
import streamlit as st

from documind.ui.ingestion_manager import IngestionManager
from documind.ui.state import safe_reset_index


def get_current_rss_mb() -> float:
    """Return the current process RSS memory in megabytes."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)


def render_sidebar(ingestion_mgr: Optional[IngestionManager] = None) -> None:
    """Render the DocuMind sidebar controls and document management panels."""
    mgr = ingestion_mgr or IngestionManager()

    with st.sidebar:
        st.title("📄 DocuMind")
        st.caption("v1.0.0 • Local-First Grounded RAG")
        st.markdown(
            "🔒 **Privacy Boundary:** Strictly Offline • ₹0 Cloud Budget • Local CPU Execution"
        )
        st.divider()

        # -------------------------------------------------------------
        # 1. Document Upload Section (APP-01)
        # -------------------------------------------------------------
        st.subheader("1. Ingest Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Select text-based PDF documents (max 50 MB each).",
        )

        col_upload, col_sample = st.columns([1, 1])

        with col_upload:
            if st.button("Index Uploads", type="primary", disabled=not bool(uploaded_files)):
                with st.status("Ingesting documents...", expanded=True) as status:
                    progress_bar = st.progress(0.0)

                    def on_progress(text: str, frac: float):
                        status.write(text)
                        progress_bar.progress(frac)

                    summary = mgr.ingest_and_index(
                        uploaded_files=uploaded_files,
                        document_index=st.session_state.document_index,
                        registry=st.session_state.indexed_docs,
                        progress_callback=on_progress,
                    )

                    for msg in summary.messages:
                        st.info(msg)
                    for err in summary.errors:
                        st.error(err)

                    status.update(
                        label=f"Ingested {summary.indexed_files}/{summary.total_files} files ({summary.total_chunks} chunks).",
                        state="complete" if not summary.errors else "error",
                        expanded=False,
                    )
                st.rerun()

        with col_sample:
            if st.button("Load Samples", help="Pre-load included technical reference PDFs"):
                sample_pdfs = [
                    Path("research/technical_references/BASWE-15-RAG-Concepts-Guide.pdf"),
                    Path("research/technical_references/Everything-to-know-about-RAG.pdf"),
                ]
                existing_samples = [p for p in sample_pdfs if p.exists()]
                if not existing_samples:
                    st.warning("Sample PDFs not found in research/technical_references/")
                else:
                    with st.status("Loading sample technical PDFs...", expanded=True) as status:
                        summary = mgr.ingest_and_index(
                            uploaded_files=existing_samples,
                            document_index=st.session_state.document_index,
                            registry=st.session_state.indexed_docs,
                            progress_callback=lambda txt, frac: status.write(txt),
                        )
                        for msg in summary.messages:
                            st.info(msg)
                        status.update(
                            label=f"Loaded {summary.indexed_files} sample guides ({summary.total_chunks} chunks).",
                            state="complete",
                            expanded=False,
                        )
                    st.rerun()

        st.divider()

        # -------------------------------------------------------------
        # 2. Document Registry & Manager (APP-02)
        # -------------------------------------------------------------
        st.subheader("2. Document Registry")
        indexed_docs: Dict[str, Any] = st.session_state.get("indexed_docs", {})
        total_vectors = st.session_state.document_index.total_vectors

        st.metric(
            label="Indexed Knowledge Base",
            value=f"{len(indexed_docs)} Documents",
            delta=f"{total_vectors} Chunks / Vectors",
        )

        if indexed_docs:
            with st.expander("Active Documents List", expanded=True):
                for doc_id, meta in indexed_docs.items():
                    size_kb = meta.get("size_bytes", 0) / 1024.0
                    st.markdown(
                        f"**{meta.get('filename')}**  \n"
                        f"• Pages: {meta.get('pages', 1)} | Chunks: {meta.get('chunks', 0)} | Size: {size_kb:.1f} KB  \n"
                        f"• Hash: `{doc_id[:10]}...`"
                    )
        else:
            st.info("No documents indexed yet. Upload PDFs or load samples above.")

        st.divider()

        # -------------------------------------------------------------
        # 3. Safe Index Reset (APP-04, Safeguard 8)
        # -------------------------------------------------------------
        st.subheader("3. Index Management")
        with st.expander("⚠️ Danger Zone (Reset Index)", expanded=False):
            st.write(
                "Clears all in-memory and persistent vector index snapshots. Model weights and codebase files are NEVER touched."
            )
            confirm_reset = st.checkbox(
                "Confirm permanent deletion of indexed vectors",
                value=False,
                key="confirm_reset_checkbox",
            )
            if st.button(
                "Clear Index & Reset",
                disabled=not confirm_reset,
                type="secondary",
                help="Permanently clears current document index.",
            ):
                safe_reset_index()
                st.success("Document index cleared successfully.")
                st.rerun()

        st.divider()

        # -------------------------------------------------------------
        # 4. Settings & System Telemetry
        # -------------------------------------------------------------
        st.subheader("4. Runtime Telemetry")
        streaming_toggle = st.checkbox(
            "Enable token streaming (provisional draft)",
            value=st.session_state.get("enable_streaming", False),
            help="Displays tokens in real-time. Notice: text remains provisional until citation verification completes.",
            key="streaming_toggle_widget",
        )
        st.session_state.enable_streaming = streaming_toggle

        rss_mb = get_current_rss_mb()
        generator = st.session_state.get("generator")
        model_name = getattr(generator.config, "model_name", "Qwen2.5-0.5B-Instruct") if generator else "Qwen2.5-0.5B"

        st.caption(f"**Model:** `{model_name}`")
        st.caption(f"**Process RAM:** `{rss_mb:.1f} MB`")
        st.caption("**Hardware:** CPU (8 threads, local execution)")
