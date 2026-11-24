"""Pluggable Local LLM abstraction, Transformers engine, and deterministic mock."""

import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Union

from documind.config import GenerationConfig
from documind.generation.schemas import GenerationResult


class LLMError(Exception):
    """Base exception for LLM inference errors."""
    pass


class ModelNotFoundError(LLMError):
    """Raised when model weights are missing locally and offline runtime is enforced."""
    pass


class BaseLLM(ABC):
    """Abstract interface for local language models in DocuMind."""

    @abstractmethod
    def generate(
        self,
        messages: Union[str, List[Dict[str, str]]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> GenerationResult:
        """Generate response text synchronously."""
        pass

    @abstractmethod
    def stream_generate(
        self,
        messages: Union[str, List[Dict[str, str]]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Iterator[str]:
        """Stream generated response tokens sequentially."""
        pass


class TransformersLLM(BaseLLM):
    """Local causal language model runner powered by Hugging Face transformers on CPU.
    
    Guarantees:
    - Enforces offline loading by default (local_files_only=True).
    - Requires explicit model download via scripts/download_llm.py.
    - Deterministic decoding when temperature=0.0.
    """

    def __init__(self, config: Optional[GenerationConfig] = None) -> None:
        self._config = config or GenerationConfig()
        self._model_name = self._config.model_name
        self._device = self._config.device
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self):
        """Lazy load model and tokenizer with strict offline protection."""
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise LLMError("transformers or torch is not installed.") from exc

        local_only = self._config.local_files_only and not self._config.allow_download
        cache_folder = self._config.cache_folder
        if cache_folder is None and os.path.exists("models/llm"):
            cache_folder = "models/llm"

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                local_files_only=local_only,
                cache_dir=cache_folder,
            )
            model_kwargs: Dict[str, Any] = {
                "local_files_only": local_only,
                "cache_dir": cache_folder,
            }
            if self._device.startswith("cuda"):
                model_kwargs["device_map"] = self._device

            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                **model_kwargs,
            )
            if not self._device.startswith("cuda"):
                self._model.to(self._device)
                num_threads = self._config.num_threads or min(8, os.cpu_count() or 4)
                torch.set_num_threads(num_threads)
            self._model.eval()
        except Exception as exc:
            err_msg = str(exc).lower()
            if "offline" in err_msg or "not found" in err_msg or "local_files_only" in err_msg or "entry not found" in err_msg:
                raise ModelNotFoundError(
                    f"Model '{self._model_name}' was not found in the local cache, and offline mode is enforced. "
                    f"Please run the explicit provisioning script first: 'python scripts/download_llm.py --model-name {self._model_name}'"
                ) from exc
            raise LLMError(f"Failed to load LLM '{self._model_name}': {exc}") from exc

    def _format_input(self, messages: Union[str, List[Dict[str, str]]]) -> str:
        """Format input messages using model chat template or raw text."""
        self._ensure_loaded()
        if isinstance(messages, str):
            return messages
        
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        
        # Fallback formatting if template not available
        lines = []
        for m in messages:
            role = m.get("role", "user").capitalize()
            content = m.get("content", "")
            lines.append(f"{role}: {content}")
        lines.append("Assistant:")
        return "\n\n".join(lines)

    def generate(
        self,
        messages: Union[str, List[Dict[str, str]]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> GenerationResult:
        """Generate response text synchronously on CPU."""
        import torch

        self._ensure_loaded()
        prompt_text = self._format_input(messages)

        inputs = self._tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[1])

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self._tokenizer.eos_token_id,
            "use_cache": True,
        }

        if temperature <= 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = self._model.generate(**inputs, **gen_kwargs)
        duration_sec = time.perf_counter() - t0
        latency_ms = duration_sec * 1000.0

        # Decode newly generated tokens only
        output_ids = outputs[0][prompt_tokens:]
        completion_tokens = len(output_ids)
        response_text = self._tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        tokens_per_sec = (completion_tokens / duration_sec) if duration_sec > 0 else 0.0

        return GenerationResult(
            text=response_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            generation_latency_ms=latency_ms,
            tokens_per_sec=tokens_per_sec,
        )

    def stream_generate(
        self,
        messages: Union[str, List[Dict[str, str]]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Iterator[str]:
        """Stream response tokens using TextIteratorStreamer."""
        import threading
        import torch
        from transformers import TextIteratorStreamer

        self._ensure_loaded()
        prompt_text = self._format_input(messages)

        inputs = self._tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs: Dict[str, Any] = {
            "inputs": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
            "max_new_tokens": max_new_tokens,
            "streamer": streamer,
            "pad_token_id": self._tokenizer.eos_token_id,
        }

        if temperature <= 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        thread = threading.Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text
        thread.join()


class MockLLM(BaseLLM):
    """Deterministic mock LLM for fast unit testing and offline verification."""

    def __init__(
        self,
        responses: Optional[Dict[str, str]] = None,
        default_response: Optional[str] = None,
        response_text: Optional[str] = None,
    ) -> None:
        self.responses = responses or {}
        if response_text is not None:
            self.default_response = response_text
        elif default_response is not None:
            self.default_response = default_response
        else:
            self.default_response = "This is a grounded answer [Source: test.pdf, p. 1]."
        self.calls: List[Union[str, List[Dict[str, str]]]] = []

    def generate(
        self,
        messages: Union[str, List[Dict[str, str]]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> GenerationResult:
        self.calls.append(messages)

        query_key = ""
        if isinstance(messages, str):
            query_key = messages
        elif isinstance(messages, list):
            for m in reversed(messages):
                if m.get("role") == "user":
                    query_key = m.get("content", "")
                    break

        response_text = self.default_response
        for k, v in self.responses.items():
            if k.lower() in query_key.lower():
                response_text = v
                break

        completion_tokens = len(response_text.split())
        return GenerationResult(
            text=response_text,
            prompt_tokens=50,
            completion_tokens=completion_tokens,
            generation_latency_ms=10.0,
            tokens_per_sec=100.0,
        )

    def stream_generate(
        self,
        messages: Union[str, List[Dict[str, str]]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Iterator[str]:
        res = self.generate(messages, max_new_tokens, temperature, top_p)
        words = res.text.split(" ")
        for i, w in enumerate(words):
            yield w if i == len(words) - 1 else w + " "
