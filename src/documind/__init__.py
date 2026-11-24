"""DocuMind: Modular, General-Purpose Retrieval-Augmented Generation (RAG) System."""

import os
# Suppress Windows OpenMP conflict between PyTorch (libiomp5md.dll) and FAISS OpenBLAS (vcomp140.dll)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.1.0"
