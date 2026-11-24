"""Diagnostic script for PyTorch CPU generation performance profiling."""

import os
import sys
import time
import torch

sys.path.insert(0, os.path.abspath("src"))
from documind.generation.llm import TransformersLLM

def benchmark_threads_and_prompt(threads, num_prompt_words, max_new_tokens):
    torch.set_num_threads(threads)
    llm = TransformersLLM()
    llm._ensure_loaded()
    
    prompt = ("This is a document sentence about technical RAG concepts. " * (num_prompt_words // 8)) + "\nQuestion: What is chunk overlap?\nAnswer:"
    inputs = llm._tokenizer(prompt, return_tensors="pt")
    prompt_tokens = inputs.input_ids.shape[1]
    
    t0 = time.perf_counter()
    with torch.inference_mode():
        # First measure TTFT (prefill only: generate 1 token)
        t_prefill_start = time.perf_counter()
        out_prefill = llm._model.generate(
            **inputs,
            max_new_tokens=1,
            use_cache=True,
            pad_token_id=llm._tokenizer.eos_token_id,
            do_sample=False,
        )
        prefill_time = time.perf_counter() - t_prefill_start
        
        # Now generate full tokens
        t_gen_start = time.perf_counter()
        outputs = llm._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=llm._tokenizer.eos_token_id,
            do_sample=False,
        )
        total_gen_time = time.perf_counter() - t_gen_start
        
    completion_tokens = outputs.shape[1] - prompt_tokens
    decode_time = max(0.001, total_gen_time - prefill_time)
    decode_tps = (completion_tokens - 1) / decode_time if completion_tokens > 1 else 0.0
    overall_tps = completion_tokens / total_gen_time
    
    print(f"Threads={threads} | PromptTokens={prompt_tokens} | NewTokens={completion_tokens} | Prefill={prefill_time*1000:.1f}ms | TotalTime={total_gen_time:.2f}s | DecodeTPS={decode_tps:.2f} | OverallTPS={overall_tps:.2f}")

if __name__ == "__main__":
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Initial torch threads: {torch.get_num_threads()}")
    
    print("\n--- Test 1: Thread count comparison with 400 token prompt (typical concise context) ---")
    for th in [2, 4, 6, 8, 10]:
        benchmark_threads_and_prompt(threads=th, num_prompt_words=200, max_new_tokens=30)
        
    print("\n--- Test 2: Prompt size impact with optimal threads (threads=4) ---")
    for words in [100, 200, 400, 800]:
        benchmark_threads_and_prompt(threads=4, num_prompt_words=words, max_new_tokens=30)
