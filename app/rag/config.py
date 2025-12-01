from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RAGConfig:
    # Paths
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    index_dir: Path = Path("data/indexes")

    # Index naming
    index_name: str = "upsc_faiss_index"

    # Embeddings
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Chunking
    chunk_size: int = 800        # characters per chunk
    chunk_overlap: int = 200     # overlapping characters

    # Retrieval
    top_k: int = 5

    @property
    def index_path(self) -> Path:
        return self.index_dir / f"{self.index_name}.bin"

    @property
    def docstore_path(self) -> Path:
        return self.index_dir / f"{self.index_name}_docstore.json"
