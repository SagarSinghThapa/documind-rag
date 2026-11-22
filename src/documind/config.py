"""Configuration constants and default parameters for DocuMind."""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class IngestionConfig:
    """Configuration parameters for document ingestion and chunking.
    
    Measurement unit for chunking is characters. At typical English densities,
    1,000 characters corresponds to approximately 200-250 tokens.
    """
    chunk_size: int = 1000
    chunk_overlap: int = 150
    chunking_strategy: str = "recursive"  # 'recursive' or 'fixed'
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])
    max_file_size_bytes: int = 50 * 1024 * 1024  # 50 MB
    supported_extensions: List[str] = field(default_factory=lambda: [".pdf"])
    extract_empty_pages: bool = False  # Whether to emit empty PageDocument objects
