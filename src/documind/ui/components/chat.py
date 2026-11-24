"""Conversational chat interface with verified grounded answers and optional provisional streaming."""

import time
from typing import Any, Dict, List, Optional
import streamlit as st

from documind.generation.generator import GroundedGenerator
from documind.generation.prompts import build_grounded_prompt
from documind.generation.schemas import GroundedAnswer
from documind.retrieval.schemas import FormattedContext
from documind.ui.components.citations import (
    render_abstention_callout,
    render_citations,
    render_grounding_badge,
    render_telemetry_expander,
)
from documind.ui.state import get_inference_lock


def _render_historical_message(msg: Dict[str, Any]) -> None:
    """Render a previously stored message from session history."""
    role = msg.get("role", "user")
    with st.chat_message(role):
        if role == "user":
            st.markdown(msg.get("content", ""))
        else:
            answer: Optional[GroundedAnswer] = msg.get("grounded_answer")
            if answer:
                render_grounding_badge(answer.grounding_status, answer.is_abstained)
                if answer.is_abstained:
                    render_abstention_callout(
                        phrase=answer.answer,
                        abstention_type=answer.abstention_type,
                        fast_path=(answer.abstention_type == "fast_path_empty_context"),
                    )
                else:
                    st.markdown(answer.answer)
                render_citations(answer.citations, answer.context)
                render_telemetry_expander(answer.telemetry)
            else:
                st.markdown(msg.get("content", ""))


def _handle_synchronous_generation(
    query_text: str, generator: GroundedGenerator
) -> None:
    """Generate answer synchronously and verify citations before rendering (Primary Default)."""
    with st.spinner("Retrieving evidence & generating grounded answer..."):
        with get_inference_lock():
            answer = generator.generate_answer(query_text)

    render_grounding_badge(answer.grounding_status, answer.is_abstained)
    if answer.is_abstained:
        render_abstention_callout(
            phrase=answer.answer,
            abstention_type=answer.abstention_type,
            fast_path=(answer.abstention_type == "fast_path_empty_context"),
        )
    else:
        st.markdown(answer.answer)

    render_citations(answer.citations, answer.context)
    render_telemetry_expander(answer.telemetry)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer.answer,
        "grounded_answer": answer,
    })


def _handle_streaming_generation(
    query_text: str, generator: GroundedGenerator
) -> None:
    """Stream provisional tokens with strict labeling and perform post-generation verification."""
    t0 = time.perf_counter()

    # 1. Retrieve Candidate Chunks
    retrieval_res = generator.retriever.retrieve(query_text)

    # Fast-Path Check: if 0 chunks retrieved, abstain instantly with 0 tokens generated
    if not retrieval_res.chunks:
        empty_ctx = FormattedContext(
            text="",
            references=[],
            total_chars=0,
            estimated_tokens=0,
            chunks_included=0,
            chunks_discarded_budget=0,
            context_build_latency_ms=0.0,
        )
        total_latency = (time.perf_counter() - t0) * 1000.0
        telemetry: Dict[str, Any] = {
            "retrieval_latency_ms": retrieval_res.total_latency_ms,
            "context_build_latency_ms": 0.0,
            "generation_latency_ms": 0.0,
            "total_latency_ms": total_latency,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "tokens_per_sec": 0.0,
            "fast_path": True,
        }
        answer = GroundedAnswer(
            query=query_text,
            answer=generator.config.abstention_phrase,
            cited_source_ids=[],
            citations=[],
            is_abstained=True,
            abstention_type="fast_path_empty_context",
            grounding_status="abstained",
            context=empty_ctx,
            telemetry=telemetry,
        )
        render_grounding_badge(answer.grounding_status, answer.is_abstained)
        render_abstention_callout(answer.answer, answer.abstention_type, fast_path=True)
        render_telemetry_expander(answer.telemetry)
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer.answer,
            "grounded_answer": answer,
        })
        return

    # 2. Context Window Assembly
    ctx = generator.context_builder.build(retrieval_res.chunks)
    if ctx.chunks_included == 0 or not ctx.text.strip():
        total_latency = (time.perf_counter() - t0) * 1000.0
        telemetry = {
            "retrieval_latency_ms": retrieval_res.total_latency_ms,
            "context_build_latency_ms": ctx.context_build_latency_ms,
            "generation_latency_ms": 0.0,
            "total_latency_ms": total_latency,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "tokens_per_sec": 0.0,
            "fast_path": True,
        }
        answer = GroundedAnswer(
            query=query_text,
            answer=generator.config.abstention_phrase,
            cited_source_ids=[],
            citations=[],
            is_abstained=True,
            abstention_type="fast_path_empty_context",
            grounding_status="abstained",
            context=ctx,
            telemetry=telemetry,
        )
        render_grounding_badge(answer.grounding_status, answer.is_abstained)
        render_abstention_callout(answer.answer, answer.abstention_type, fast_path=True)
        render_telemetry_expander(answer.telemetry)
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer.answer,
            "grounded_answer": answer,
        })
        return

    # 3. Stream Tokens with Explicit Provisional Draft Banner
    messages = build_grounded_prompt(
        query_text, ctx, include_exemplar=generator.config.include_exemplar
    )

    info_banner = st.info(
        "ℹ️ **Generating provisional draft...** (Citations and claims remain unverified until stream completes)"
    )
    draft_container = st.empty()
    tokens_accumulated: List[str] = []
    t_gen0 = time.perf_counter()

    with get_inference_lock():
        token_stream = generator.llm.stream_generate(
            messages,
            max_new_tokens=generator.config.max_new_tokens,
            temperature=generator.config.temperature,
            top_p=generator.config.top_p,
        )
        for token in token_stream:
            tokens_accumulated.append(token)
            draft_container.markdown("".join(tokens_accumulated) + " ▌")

    gen_latency_ms = (time.perf_counter() - t_gen0) * 1000.0
    raw_answer = "".join(tokens_accumulated).strip()

    # Clear provisional elements
    info_banner.empty()
    draft_container.empty()

    # 4. Post-Stream Verification & Grounding Contract
    prompt_tokens = len(str(messages).split())
    completion_tokens = len(raw_answer.split())
    tokens_per_sec = (completion_tokens / (gen_latency_ms / 1000.0)) if gen_latency_ms > 0 else 0.0

    answer = generator.finalize_answer(
        query_text=query_text,
        raw_answer=raw_answer,
        retrieval_res=retrieval_res,
        ctx=ctx,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        gen_latency_ms=gen_latency_ms,
        tokens_per_sec=tokens_per_sec,
        t0=t0,
    )

    render_grounding_badge(answer.grounding_status, answer.is_abstained)
    if answer.is_abstained:
        render_abstention_callout(
            phrase=answer.answer,
            abstention_type=answer.abstention_type,
            fast_path=False,
        )
    else:
        st.markdown(answer.answer)

    render_citations(answer.citations, answer.context)
    render_telemetry_expander(answer.telemetry)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer.answer,
        "grounded_answer": answer,
    })


def render_chat_interface() -> None:
    """Render historical conversation, quick-start pills, input bar, and grounded generator."""
    # 1. Render Historical Messages
    for msg in st.session_state.messages:
        _render_historical_message(msg)

    # 2. Quick-Start Technical Sample Queries (Visible when message history is empty)
    clicked_sample = None
    if not st.session_state.messages:
        st.markdown("##### 💡 Quick-Start Questions")
        col1, col2, col3 = st.columns(3)
        if col1.button("What is chunk overlap?", key="btn_sample_1", use_container_width=True):
            clicked_sample = "What is chunk overlap?"
        if col2.button("How does hybrid search work?", key="btn_sample_2", use_container_width=True):
            clicked_sample = "How does hybrid search work?"
        if col3.button("What is the token-to-word rule of thumb?", key="btn_sample_3", use_container_width=True):
            clicked_sample = "What is the token-to-word rule of thumb?"

    # 3. Chat Input Bar
    user_input = st.chat_input("Ask a question about your documents...")

    active_query = clicked_sample or user_input
    if active_query:
        query_text = active_query.strip()
        if not query_text:
            return

        # Render user message
        st.session_state.messages.append({"role": "user", "content": query_text})
        with st.chat_message("user"):
            st.markdown(query_text)

        # Generate Assistant Answer
        generator: GroundedGenerator = st.session_state.generator
        with st.chat_message("assistant"):
            if st.session_state.get("enable_streaming", False):
                _handle_streaming_generation(query_text, generator)
            else:
                _handle_synchronous_generation(query_text, generator)
