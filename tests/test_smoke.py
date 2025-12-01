from __future__ import annotations

from app.rag.config import RAGConfig
from app.rag.retrieval import chunk_text


def test_config_defaults():
    config = RAGConfig()
    assert config.chunk_size > 0
    assert config.chunk_overlap >= 0
    assert config.top_k > 0
    assert config.embedding_model_name
    assert config.llm_model_name


def test_chunk_text_basic():
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = chunk_text(text, chunk_size=10, chunk_overlap=2)

    # Should produce more than one chunk
    assert len(chunks) >= 2

    # Combined chunks should roughly cover the text (with overlap)
    joined = "".join(chunks)
    for ch in "abcxyz":
        assert ch in joined
