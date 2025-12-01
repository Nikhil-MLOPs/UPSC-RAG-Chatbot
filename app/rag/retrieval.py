from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from app.rag.config import RAGConfig


@dataclass
class DocumentChunk:
    """Represents a single chunk of text used for retrieval."""
    id: str
    subject: str
    source_doc_id: str
    text: str


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Simple character-based chunking with overlap.
    """
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        # Move start with overlap
        start = end - chunk_overlap

    return chunks


def load_processed_as_chunks(config: RAGConfig) -> List[DocumentChunk]:
    """
    Walk through data/processed/<subject>/*.txt and create chunks.
    """
    processed_dir = config.processed_dir
    chunks: List[DocumentChunk] = []

    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed dir not found: {processed_dir.resolve()}")

    for subject_dir in processed_dir.iterdir():
        if not subject_dir.is_dir():
            continue

        subject = subject_dir.name

        for txt_path in subject_dir.glob("*.txt"):
            source_doc_id = txt_path.stem  # e.g. geography_Geo19-2-Climatology-1
            raw_text = txt_path.read_text(encoding="utf-8", errors="ignore")

            text_chunks = chunk_text(
                raw_text,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )

            for i, ch in enumerate(text_chunks):
                chunk_id = f"{source_doc_id}_chunk_{i}"
                chunks.append(
                    DocumentChunk(
                        id=chunk_id,
                        subject=subject,
                        source_doc_id=source_doc_id,
                        text=ch,
                    )
                )

    return chunks


def build_faiss_index(config: RAGConfig) -> None:
    """
    Build FAISS index from processed text chunks and save index + docstore.
    """
    config.index_dir.mkdir(parents=True, exist_ok=True)

    print("[INDEX] Loading processed text and creating chunks...")
    chunks = load_processed_as_chunks(config)
    print(f"[INDEX] Total chunks: {len(chunks)}")

    if not chunks:
        print("[INDEX] No chunks found. Did you run ingestion first?")
        return

    texts = [c.text for c in chunks]

    print("[INDEX] Loading embedding model:", config.embedding_model_name)
    model = SentenceTransformer(config.embedding_model_name)

    print("[INDEX] Computing embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # for cosine similarity with inner product
    )

    embeddings = embeddings.astype("float32")
    dim = embeddings.shape[1]

    print(f"[INDEX] Building FAISS index with dimension {dim}...")
    index = faiss.IndexFlatIP(dim)  # inner product (cosine if normalized)
    index.add(embeddings)

    print(f"[INDEX] Saving FAISS index to {config.index_path}")
    faiss.write_index(index, str(config.index_path))

    docstore: List[Dict[str, Any]] = [asdict(c) for c in chunks]

    print(f"[INDEX] Saving docstore metadata to {config.docstore_path}")
    with config.docstore_path.open("w", encoding="utf-8") as f:
        json.dump(docstore, f, ensure_ascii=False)

    print("[INDEX] Done building index.")


def load_index_and_model(config: RAGConfig) -> Tuple[faiss.Index, List[Dict[str, Any]], SentenceTransformer]:
    """
    Load FAISS index, docstore, and embedding model for querying.
    """
    if not config.index_path.exists():
        raise FileNotFoundError(f"Index not found at {config.index_path}. Build it first with --build.")

    if not config.docstore_path.exists():
        raise FileNotFoundError(f"Docstore not found at {config.docstore_path}. Build index first.")

    print(f"[LOAD] Loading FAISS index from {config.index_path}")
    index = faiss.read_index(str(config.index_path))

    print(f"[LOAD] Loading docstore from {config.docstore_path}")
    with config.docstore_path.open("r", encoding="utf-8") as f:
        docstore: List[Dict[str, Any]] = json.load(f)

    print("[LOAD] Loading embedding model for queries...")
    model = SentenceTransformer(config.embedding_model_name)

    return index, docstore, model


def search(
    query: str,
    config: RAGConfig,
    top_k: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Search the FAISS index for a given query and return top_k chunks with scores.
    """
    if top_k is None:
        top_k = config.top_k

    index, docstore, model = load_index_and_model(config)

    print(f"[SEARCH] Encoding query: {query}")
    q_emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    print(f"[SEARCH] Querying top_k={top_k}...")
    scores, indices = index.search(q_emb, top_k)

    results: List[Dict[str, Any]] = []
    for score, idx in zip(scores[0], indices[0]):
        meta = docstore[int(idx)]
        meta_with_score = {
            **meta,
            "score": float(score),
        }
        results.append(meta_with_score)

    return results


def interactive_cli(config: RAGConfig) -> None:
    """
    Simple interactive CLI to test retrieval.
    """
    print("=== UPSC RAG Retrieval CLI ===")
    print("Type your question (or 'exit' to quit)")

    while True:
        query = input("\nQ: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        results = search(query, config=config, top_k=config.top_k)
        print("\nTop chunks:")
        for i, r in enumerate(results, start=1):
            print(f"\n[{i}] subject={r['subject']}, source={r['source_doc_id']}, score={r['score']:.4f}")
            print("-" * 80)
            # Show a limited preview
            preview = r["text"]
            if len(preview) > 500:
                preview = preview[:500] + "..."
            print(preview)


def main():
    parser = argparse.ArgumentParser(description="Build and query FAISS index for UPSC RAG.")
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build the FAISS index from processed text.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single query to run against the index.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive retrieval CLI.",
    )

    args = parser.parse_args()
    config = RAGConfig()

    if args.build:
        build_faiss_index(config)
        return

    if args.query:
        results = search(args.query, config=config, top_k=config.top_k)
        print(f"\nTop {len(results)} chunks for query: {args.query!r}")
        for i, r in enumerate(results, start=1):
            print(f"\n[{i}] subject={r['subject']}, source={r['source_doc_id']}, score={r['score']:.4f}")
            print("-" * 80)
            preview = r["text"]
            if len(preview) > 500:
                preview = preview[:500] + "..."
            print(preview)
        return

    if args.interactive:
        interactive_cli(config)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
