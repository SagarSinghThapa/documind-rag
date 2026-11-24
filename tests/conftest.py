"""Pytest configuration and environment fixtures for DocuMind test suite."""

import os
import sys

# Ensure KMP duplicate lib is allowed on Windows before OpenMP runtimes load
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Pre-import faiss at pytest session start to bind OpenMP once per process
import faiss
