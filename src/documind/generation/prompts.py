"""Grounded prompt engineering templates and context isolation contracts."""

from typing import Dict, List, Optional
from documind.retrieval.schemas import FormattedContext

SYSTEM_PROMPT = (
    "You are DocuMind, an accurate, citation-grounded assistant.\n"
    "Your task is to answer the user's question SOLELY using the facts provided in the Context below.\n\n"
    "RULES:\n"
    "1. Do NOT use outside knowledge or make assumptions not directly stated in the Context.\n"
    "2. Every claim or factual statement you make MUST include an inline citation citing the source identifier [S1], [S2] or [Source: <filename>, p. <page>].\n"
    "3. Example format: 'Chunk overlap is usually 10 to 20 percent of chunk size [S1].'\n"
    "4. If the Context is empty, or does not contain sufficient information to answer the question with certainty, you MUST respond ONLY with:\n"
    "\"I cannot answer this question based on the provided document context.\"\n"
    "Do not guess, apologize, or provide tangential commentary.\n"
    "5. Do NOT copy, reproduce, or cite context headers, chunk IDs, '### Source:', or 'Ref ID:' lines in your response."
)

FEW_SHOT_USER = (
    "Context:\n"
    "[S1] ### Source: Solar-System-Guide.pdf | Physical Page(s): p. 1 | Ref ID: ref_solar_01\n"
    "> The Earth orbits the Sun once every 365.25 days.\n\n"
    "Question: How long does Earth take to orbit the Sun?\n\n"
    "Answer:"
)
FEW_SHOT_ASSISTANT = "The Earth orbits the Sun once every 365.25 days [S1]."


def build_grounded_prompt(
    query: str,
    context: FormattedContext,
    system_prompt: Optional[str] = None,
    include_exemplar: bool = False,
) -> List[Dict[str, str]]:
    """Assemble standard chat messages payload with untrusted context encapsulation."""
    sys_content = system_prompt or SYSTEM_PROMPT

    if not context.text or not context.text.strip():
        user_content = (
            f"Context:\n[No relevant document context found]\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
    else:
        user_content = (
            f"Context:\n{context.text}\n\n"
            f"Question: {query}\n\n"
            f"Remember to cite every claim using [S1], [S2] etc.\n"
            f"Answer:"
        )

    if include_exemplar and context.text and context.text.strip():
        return [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": FEW_SHOT_USER},
            {"role": "assistant", "content": FEW_SHOT_ASSISTANT},
            {"role": "user", "content": user_content},
        ]

    return [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": user_content},
    ]
