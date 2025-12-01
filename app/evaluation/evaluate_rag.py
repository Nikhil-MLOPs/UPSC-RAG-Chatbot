from __future__ import annotations

import argparse
import json
import csv
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import mlflow

from app.rag.config import RAGConfig
from app.rag.pipeline import RAGPipeline


def load_eval_set(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Eval file not found: {path.resolve()}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Eval file must contain a JSON list of {id, question, reference_answer} objects")

    return data


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def run_evaluation(
    eval_path: Path,
    limit: int | None = None,
    config: RAGConfig | None = None,
) -> Dict[str, Any]:
    config = config or RAGConfig()
    eval_examples = load_eval_set(eval_path)

    if limit is not None:
        eval_examples = eval_examples[:limit]

    print(f"[EVAL] Loaded {len(eval_examples)} examples from {eval_path}")

    # Initialize pipeline once (loads index, embed model, LLM)
    pipeline = RAGPipeline(config=config)
    embed_model = pipeline.embed_model  # reuse the same embedding model

    results: List[Dict[str, Any]] = []
    cosine_scores: List[float] = []

    for ex in eval_examples:
        qid = ex.get("id", "")
        question = ex["question"]
        reference_answer = ex["reference_answer"]

        print(f"\n[EVAL] Evaluating {qid or question[:50]}...")

        rag_output = pipeline.answer(question)
        generated_answer = rag_output["answer"]

        # Embed reference and generated answers
        ref_emb = embed_model.encode([reference_answer], convert_to_numpy=True, normalize_embeddings=True)[0]
        gen_emb = embed_model.encode([generated_answer], convert_to_numpy=True, normalize_embeddings=True)[0]

        cos = cosine_similarity(ref_emb, gen_emb)
        cosine_scores.append(cos)

        result_row = {
            "id": qid,
            "question": question,
            "reference_answer": reference_answer,
            "generated_answer": generated_answer,
            "cosine_similarity": cos,
        }
        results.append(result_row)

    mean_cosine = float(np.mean(cosine_scores)) if cosine_scores else 0.0

    summary = {
        "num_examples": len(eval_examples),
        "mean_cosine_similarity": mean_cosine,
        "results": results,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate UPSC RAG pipeline with MLflow tracking.")
    parser.add_argument(
        "--eval-file",
        type=str,
        default="data/eval/upsc_eval_set.json",
        help="Path to evaluation JSON file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples (for quick tests).",
    )
    args = parser.parse_args()

    eval_path = Path(args.eval_file)
    config = RAGConfig()

    mlflow.set_experiment("upsc_rag_evaluation")

    with mlflow.start_run():
        # Log key configuration parameters
        mlflow.log_param("embedding_model_name", config.embedding_model_name)
        mlflow.log_param("chunk_size", config.chunk_size)
        mlflow.log_param("chunk_overlap", config.chunk_overlap)
        mlflow.log_param("top_k", config.top_k)
        mlflow.log_param("llm_model_name", config.llm_model_name)
        mlflow.log_param("max_new_tokens", config.max_new_tokens)
        mlflow.log_param("temperature", config.temperature)
        mlflow.log_param("top_p", config.top_p)
        mlflow.log_param("eval_file", str(eval_path))
        if args.limit is not None:
            mlflow.log_param("eval_limit", args.limit)

        summary = run_evaluation(
            eval_path=eval_path,
            limit=args.limit,
            config=config,
        )

        mean_cos = summary["mean_cosine_similarity"]
        mlflow.log_metric("mean_answer_ref_cosine", mean_cos)

        print(f"\n[EVAL] Mean cosine similarity between generated and reference answers: {mean_cos:.4f}")

        # Save per-question results as a CSV artifact
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        csv_path = artifacts_dir / "upsc_rag_eval_results.csv"

        fieldnames = ["id", "question", "reference_answer", "generated_answer", "cosine_similarity"]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary["results"]:
                writer.writerow(row)

        mlflow.log_artifact(str(csv_path))


if __name__ == "__main__":
    main()
