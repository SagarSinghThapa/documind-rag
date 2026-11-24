"""Main entry point for DocuMind Streamlit Web Application."""

import streamlit as st

from documind.ui.components.chat import render_chat_interface
from documind.ui.components.sidebar import render_sidebar
from documind.ui.state import init_session_state


def main() -> None:
    st.set_page_config(
        page_title="DocuMind — Grounded RAG with Verifiable Citations",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize per-session state and hydrate snapshots
    init_session_state()

    # Render sidebar controls, uploads, registry, and telemetry
    render_sidebar()

    # Main page header
    st.title("📄 DocuMind")
    st.caption("Local-First Retrieval-Augmented Generation with Verifiable Page Citations")
    st.markdown(
        "🔒 **Privacy Boundary:** Strictly Offline • ₹0 Cloud Cost • Local CPU Execution • 100% Data Confidentiality"
    )

    # Surface dismissible error notices if any
    error_notice = st.session_state.get("error_notice")
    if error_notice:
        col_err, col_dismiss = st.columns([0.88, 0.12])
        with col_err:
            st.error(f"**Notice:** {error_notice}")
        with col_dismiss:
            if st.button("Dismiss", key="dismiss_error_btn"):
                st.session_state.error_notice = None
                st.rerun()

    st.divider()

    # Render conversational chat interface
    render_chat_interface()


if __name__ == "__main__":
    main()
