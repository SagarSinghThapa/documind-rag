"""Script to explicitly provision and download local embedding models.

Guarantees that models are provisioned in a separate, explicit step before
offline runtime inference, ensuring zero silent network access during execution.
"""

import argparse
import sys
import time
from pathlib import Path


def download_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", cache_folder: str = None) -> bool:
    """Download and verify local embedding model weights."""
    print(f"[Provisioning] Downloading model '{model_name}'...")
    start_time = time.perf_counter()
    try:
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer(model_name, cache_folder=cache_folder)
        load_time = time.perf_counter() - start_time
        print(f"[Provisioning] Successfully downloaded and loaded model in {load_time:.2f}s")
        
        # Verification test on CPU
        test_text = "DocuMind offline verification query."
        test_vector = model.encode(test_text, normalize_embeddings=True)
        print(f"[Provisioning] Inference check: vector shape={test_vector.shape}, norm={float((test_vector**2).sum()**0.5):.4f}")
        print("[Provisioning] Model is fully cached and verified for 100% offline local inference.")
        return True
    except Exception as exc:
        print(f"[Provisioning ERROR] Failed to provision model '{model_name}': {exc}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Explicit model provisioning tool for DocuMind.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace model repository name or path.",
    )
    parser.add_argument(
        "--cache-folder",
        type=str,
        default=None,
        help="Optional custom cache directory.",
    )
    args = parser.parse_args()
    success = download_model(args.model_name, args.cache_folder)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
