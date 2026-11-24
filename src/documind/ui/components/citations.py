"""Citation presentation, expandable evidence accordions, and 3-layer verification badging."""

from typing import Any, Dict, List, Optional
import streamlit as st

from documind.generation.schemas import Citation
from documind.retrieval.schemas import FormattedContext


def render_grounding_badge(grounding_status: str, is_abstained: bool = False) -> None:
    """Render visually distinct grounding status badges with strict truthfulness.
    
    CRITICAL RULE: Unsupported or partially supported claims are NEVER rendered
    with a green verified checkmark.
    """
    if is_abstained or grounding_status == "abstained":
        st.info("ℹ️ **Grounding Status:** Abstained (No Conclusive Evidence)")
    elif grounding_status == "verified":
        st.success("✅ **Grounding Status:** Verified (All claims supported by citations)")
    elif grounding_status == "partially_supported":
        st.warning("⚠️ **Grounding Status:** Partially Supported (Some factual claims lack explicit citations)")
    elif grounding_status == "unsupported":
        st.error("🛑 **Grounding Status:** Unsupported Claims Intercepted (Conservative Fallback Triggered)")
    else:
        st.info(f"ℹ️ **Grounding Status:** {grounding_status.capitalize()}")


def render_abstention_callout(
    phrase: str,
    abstention_type: Optional[str] = None,
    fast_path: bool = False,
) -> None:
    """Render an informative, formatted callout for abstained queries."""
    if fast_path or abstention_type == "fast_path_empty_context":
        desc = "Fast-path pre-generation rejection: No retrieved document passages exceeded the 0.35 similarity threshold. 0 LLM tokens generated."
    elif abstention_type == "model_explicit":
        desc = "Model-level detection: The language model concluded the retrieved document context contains insufficient evidence to answer."
    elif abstention_type == "fallback_unsupported":
        desc = "Safety fallback: The model produced factual claims without valid citations to the retrieved context; ungrounded claims were intercepted."
    else:
        desc = "Insufficient factual evidence in the uploaded document collection."

    st.warning(f"**{phrase}**\n\n*Reason:* {desc}")


def render_citations(citations: List[Citation], context: Optional[FormattedContext] = None) -> None:
    """Render expandable citation cards preserving source IDs, page numbers, and chunk text."""
    if not citations:
        return

    st.markdown("##### 📚 Sources & Verifiable Citations")

    ref_map = {}
    if context and context.references:
        for r in context.references:
            ref_map[r.chunk_id] = r
            if r.source_id:
                ref_map[r.source_id] = r

    for cit in citations:
        sid = cit.source_id or "Source"
        doc = cit.document_name or "Document"
        pages = f"p. {cit.start_page}" if cit.start_page == cit.end_page else f"pp. {cit.start_page}-{cit.end_page}"
        expander_title = f"[{sid}] {doc} — {pages}"

        with st.expander(expander_title, expanded=False):
            col_meta, col_badges = st.columns([1, 1])

            with col_meta:
                st.markdown(f"**Document:** `{doc}`")
                st.markdown(f"**Physical Page(s):** `{pages}`")
                st.markdown(f"**Chunk ID:** `{cit.matched_chunk_id or 'Unmapped'}`")

            with col_badges:
                # 3-Layer Verification Badges
                st.markdown(
                    f"• **Syntax (Layer 1):** "
                    f"{'Valid ✅' if cit.is_syntax_valid else 'Malformed ❌'}"
                )
                st.markdown(
                    f"• **Metadata Linkage (Layer 2):** "
                    f"{'Mapped to ContextReference ✅' if cit.is_metadata_valid else 'Hallucinated Source ❌'}"
                )
                st.markdown(
                    f"• **Evidence Support (Layer 3):** "
                    f"{'Claim terms confirmed in chunk ✅' if cit.is_evidence_supported else 'Unverified ⚠️'}"
                )

            # Look up excerpt from context references or formatted context block
            ref = ref_map.get(cit.matched_chunk_id) or (ref_map.get(cit.source_id) if cit.source_id else None)
            excerpt = None
            if hasattr(ref, "text") and getattr(ref, "text", None):
                excerpt = getattr(ref, "text")
            elif context and context.text and ref and ref.source_id:
                for block in context.text.split("\n\n"):
                    if block.startswith(f"[{ref.source_id}]"):
                        content_lines = [
                            line[2:].strip()
                            for line in block.splitlines()
                            if line.startswith("> ") and not line.startswith("> [Document Content")
                        ]
                        if content_lines:
                            excerpt = " ".join(content_lines)
                        else:
                            excerpt = block
                        break

            if excerpt:
                st.markdown("**Evidence Chunk Excerpt:**")
                st.caption(f"> {excerpt}")
            elif cit.raw_text:
                st.caption(f"Raw Citation String: `{cit.raw_text}`")


def render_telemetry_expander(telemetry: Dict[str, Any]) -> None:
    """Render telemetry diagnostics: latencies, token throughput, and RAM."""
    if not telemetry:
        return

    with st.expander("⏱️ Performance Telemetry & Hardware Diagnostics", expanded=False):
        c1, c2, c3, c4 = st.columns(4)

        retrieval_ms = telemetry.get("retrieval_latency_ms", 0.0)
        c1.metric("Retrieval Latency", f"{retrieval_ms:.1f} ms")

        gen_ms = telemetry.get("generation_latency_ms", 0.0)
        c2.metric("Generation Latency", f"{gen_ms:.1f} ms")

        tps = telemetry.get("tokens_per_sec", 0.0)
        c3.metric("Throughput", f"{tps:.2f} tok/s")

        total_ms = telemetry.get("total_latency_ms", 0.0)
        c4.metric("Total Latency", f"{total_ms:.1f} ms")

        p_tok = telemetry.get("prompt_tokens", 0)
        c_tok = telemetry.get("completion_tokens", 0)
        fast_path = telemetry.get("fast_path", False)
        hallucinated = telemetry.get("hallucinated_citations", [])

        st.caption(
            f"**Tokens:** Prompt: `{p_tok}` | Completion: `{c_tok}` | "
            f"**Fast-Path:** `{fast_path}` | "
            f"**Hallucinated Citations:** `{len(hallucinated)}`"
        )
