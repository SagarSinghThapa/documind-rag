"""Test 2-message prompt format for Qwen2.5-0.5B."""

import os
import sys
import torch

sys.path.insert(0, os.path.abspath("src"))
from documind.generation.llm import TransformersLLM

def test():
    torch.set_num_threads(8)
    llm = TransformersLLM()
    llm._ensure_loaded()
    
    sys_prompt = (
        "You are DocuMind, an accurate, citation-grounded assistant.\n"
        "Your task is to answer the user's question SOLELY using the facts provided in the Context below.\n\n"
        "RULES:\n"
        "1. Do NOT use outside knowledge or make assumptions not directly stated in the Context.\n"
        "2. Every claim or factual statement you make MUST include an inline citation citing the source identifier like [S1], [S2] or [Source: <filename>, p. <page>].\n"
        "3. Example format: 'Chunk overlap is usually 10 to 20 percent of chunk size [S1].'\n"
        "4. If the Context is empty, or does not contain sufficient information to answer the question with certainty, you MUST respond ONLY with:\n"
        "\"I cannot answer this question based on the provided document context.\"\n"
        "Do not guess, apologize, or provide tangential commentary."
    )
    
    user_prompt = (
        "Context:\n"
        "[S1] Source: BASWE-15-RAG-Concepts-Guide.pdf | Page(s): pp. 2-3 | ID: chk_01\n"
        "> Chunk overlap repeats text at chunk boundaries. The usual range is 10 to 20 percent of chunk size.\n\n"
        "Question: What is the typical recommended range for chunk overlap percentage?\n\n"
        "Answer:"
    )
    
    msgs = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    res = llm.generate(msgs, max_new_tokens=64)
    print("PROMPT TOKENS:", res.prompt_tokens)
    print("OUTPUT TOKENS:", res.completion_tokens)
    print("LATENCY (ms):", round(res.generation_latency_ms, 1))
    print("RAW ANSWER:\n", res.text)

if __name__ == "__main__":
    test()
