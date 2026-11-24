"""Headless automated test suite for DocuMind Streamlit Web Application using AppTest."""

import os
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

from documind.storage.schemas import TextChunk

APP_PATH = str(Path(__file__).resolve().parent.parent / "src" / "documind" / "app.py")


@pytest.fixture(autouse=True)
def configure_test_environment(tmp_path, monkeypatch):
    """Ensure safe isolated directories and MockLLM for headless UI tests."""
    test_data = tmp_path / "data"
    test_uploads = test_data / "uploads"
    test_index = test_data / "index"

    test_uploads.mkdir(parents=True, exist_ok=True)
    test_index.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("DOCUMIND_USE_MOCK_LLM", "1")
    monkeypatch.setenv("DOCUMIND_DATA_DIR", str(test_data))
    monkeypatch.setenv("DOCUMIND_UPLOAD_DIR", str(test_uploads))
    monkeypatch.setenv("DOCUMIND_INDEX_DIR", str(test_index))


def test_app_initial_render():
    """Verify application renders cleanly without exceptions and mounts core widgets."""
    at = AppTest.from_file(APP_PATH, default_timeout=15).run()
    assert not at.exception, f"AppTest raised unexpected exception: {at.exception}"

    # Verify Header & Branding
    titles = [t.value for t in at.title]
    assert any("DocuMind" in t for t in titles)

    # Verify Sidebar Upload & Registry components
    assert len(at.sidebar.file_uploader) >= 1
    assert any("Indexed Knowledge Base" in getattr(m, "label", "") for m in at.sidebar.metric)

    # Verify Chat Input widget
    assert len(at.chat_input) >= 1
    assert at.chat_input[0].placeholder == "Ask a question about your documents..."

    # Verify initial empty messages list
    assert at.session_state.messages == []


def test_app_empty_index_fast_path_abstention():
    """Verify instant fast-path abstention when submitting a query against an empty index."""
    at = AppTest.from_file(APP_PATH, default_timeout=15).run()
    assert not at.exception

    # Submit query against empty vector store
    at.chat_input[0].set_value("What is chunk overlap?").run()
    assert not at.exception

    # Verify message history contains user query and assistant abstention
    messages = at.session_state.messages
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What is chunk overlap?"

    assistant_msg = messages[1]
    assert assistant_msg["role"] == "assistant"
    grounded_ans = assistant_msg.get("grounded_answer")
    assert grounded_ans is not None
    assert grounded_ans.is_abstained is True
    assert grounded_ans.abstention_type == "fast_path_empty_context"
    assert grounded_ans.grounding_status == "abstained"
    assert grounded_ans.telemetry.get("fast_path") is True


def test_app_chat_interaction_with_mock_grounded_answer():
    """Verify chat flow with seeded document chunks producing verified citations."""
    at = AppTest.from_file(APP_PATH, default_timeout=15).run()
    assert not at.exception

    # Seed in-memory chunk into session_state document_index
    chunk = TextChunk(
        chunk_id="test_doc_c0",
        document_id="test_doc",
        document_name="BASWE-15-RAG-Concepts-Guide.pdf",
        source_path="research/technical_references/BASWE-15-RAG-Concepts-Guide.pdf",
        page_numbers=[2],
        start_page=2,
        end_page=2,
        chunk_index=0,
        text="A typical starting point is 10% to 20% chunk overlap to maintain contextual continuity.",
        chunking_strategy="recursive",
        chunk_size=1000,
        chunk_overlap=150,
    )
    at.session_state.document_index.add_chunks([chunk])
    assert at.session_state.document_index.total_vectors == 1

    # Configure MockLLM response with valid [S1] citation syntax
    mock_llm = at.session_state.generator.llm
    mock_llm.default_response = (
        "Chunk overlap typically ranges from 10% to 20% [S1]."
    )

    # Submit technical query
    at.chat_input[0].set_value("What is the typical chunk overlap?").run()
    assert not at.exception

    messages = at.session_state.messages
    assert len(messages) == 2
    grounded_ans = messages[1]["grounded_answer"]
    assert grounded_ans is not None
    assert grounded_ans.is_abstained is False
    assert grounded_ans.grounding_status == "verified"
    assert len(grounded_ans.citations) >= 1
    assert grounded_ans.citations[0].is_metadata_valid is True
    assert grounded_ans.citations[0].document_name == "BASWE-15-RAG-Concepts-Guide.pdf"


def test_app_sidebar_safe_reset():
    """Verify two-step confirmed reset clears in-memory and disk state while preserving models/."""
    at = AppTest.from_file(APP_PATH, default_timeout=15).run()
    assert not at.exception

    # Seed test data
    chunk = TextChunk(
        chunk_id="reset_c0",
        document_id="reset_doc",
        document_name="test.pdf",
        source_path="data/uploads/test.pdf",
        page_numbers=[1],
        start_page=1,
        end_page=1,
        chunk_index=0,
        text="Sample text to be reset.",
        chunking_strategy="recursive",
        chunk_size=1000,
        chunk_overlap=150,
    )
    at.session_state.document_index.add_chunks([chunk])
    at.session_state.indexed_docs["reset_doc"] = {"filename": "test.pdf", "chunks": 1}
    at.session_state.messages.append({"role": "user", "content": "Prior question"})
    assert at.session_state.document_index.total_vectors == 1

    # Locate reset confirmation checkbox in sidebar
    checkboxes = [c for c in at.sidebar.checkbox if getattr(c, "key", "") == "confirm_reset_checkbox"]
    assert len(checkboxes) == 1
    reset_checkbox = checkboxes[0]

    # Check the box and trigger rerun
    reset_checkbox.check().run()
    assert not at.exception

    # Locate reset button
    buttons = [b for b in at.sidebar.button if "Clear Index & Reset" in b.label]
    assert len(buttons) == 1
    reset_btn = buttons[0]

    # Click reset button
    reset_btn.click().run()
    assert not at.exception

    # Verify state reset
    assert at.session_state.document_index.total_vectors == 0
    assert len(at.session_state.indexed_docs) == 0
    assert len(at.session_state.messages) == 0

    # Verify models/ directory remains strictly intact
    assert Path("models/llm").exists()


def test_app_malformed_empty_input_graceful():
    """Verify empty or whitespace-only inputs are safely ignored without exceptions."""
    at = AppTest.from_file(APP_PATH, default_timeout=15).run()
    assert not at.exception

    # Submit whitespace-only string
    at.chat_input[0].set_value("     \t\n  ").run()
    assert not at.exception

    # Messages must remain empty
    assert len(at.session_state.messages) == 0
