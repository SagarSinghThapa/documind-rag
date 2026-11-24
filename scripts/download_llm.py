"""Standalone CLI to provision local LLM weights explicitly before offline inference."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download and cache local LLM weights for DocuMind.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face model identifier (default: Qwen/Qwen2.5-0.5B-Instruct)",
    )
    parser.add_argument(
        "--cache-folder",
        type=str,
        default=None,
        help="Optional destination cache directory for model weights",
    )
    args = parser.parse_args()

    print(f"=== DocuMind LLM Provisioning ===")
    print(f"Target model: {args.model_name}")
    print(f"Cache folder: {args.cache_folder or 'Hugging Face default cache'}")
    print("Initiating explicit download with allow_download=True...")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("[1/2] Downloading tokenizer...")
        AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_folder, local_files_only=False)

        print("[2/2] Downloading model weights...")
        AutoModelForCausalLM.from_pretrained(
            args.model_name, cache_dir=args.cache_folder, local_files_only=False
        )
        print(f"SUCCESS: Model '{args.model_name}' successfully provisioned and ready for offline execution.")
    except Exception as exc:
        print(f"ERROR: Failed to provision model '{args.model_name}': {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
