from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.rag.config import RAGConfig
from app.rag.retrieval import load_index_and_model


@dataclass
class RetrievedChunk:
    subject: str
    source_doc_id: str
    text: str
    score: float


class RAGPipeline:
    """
    End-to-end RAG pipeline:
      - embed query
      - retrieve relevant chunks from FAISS
      - build a UPSC-friendly prompt
      - generate answer with small open-source LLM
    """

    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RAGConfig()

        # Device: CPU by default (designed for machines like i5 + 8GB RAM)
        self.device = torch.device("cpu")

        # Load retrieval components (index, docstore, embedding model)
        print("[RAG] Loading index, docstore and embedding model...")
        self.index, self.docstore, self.embed_model = load_index_and_model(self.config)

        # Load LLM + tokenizer
        print(f"[RAG] Loading LLM: {self.config.llm_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.llm_model_name,
            torch_dtype=torch.float32,  # safe for CPU
        ).to(self.device)

        # Some chat models need a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def retrieve(self, query: str, top_k: int | None = None) -> List[RetrievedChunk]:
        """
        Use FAISS index + embedding model to get top_k relevant chunks.
        """
        if top_k is None:
            top_k = self.config.top_k

        print(f"[RAG] Encoding query for retrieval: {query}")
        q_emb = self.embed_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(q_emb, top_k)

        chunks: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            meta = self.docstore[int(idx)]
            chunks.append(
                RetrievedChunk(
                    subject=meta["subject"],
                    source_doc_id=meta["source_doc_id"],
                    text=meta["text"],
                    score=float(score),
                )
            )

        return chunks

    def build_prompt(self, query: str, chunks: List[RetrievedChunk]) -> str:
        """
        Build a UPSC-oriented prompt with retrieved context.
        """
        context_parts = []
        for i, ch in enumerate(chunks, start=1):
            context_parts.append(
                f"[{i}] (subject={ch.subject}, source={ch.source_doc_id})\n{ch.text}"
            )

        context = "\n\n".join(context_parts)

        system_instructions = (
            "You are an AI assistant helping a student prepare for the UPSC exam. "
            "You answer ONLY using the provided context from UPSC-related notes and books. "
            "If the answer is not clearly present in the context, say you are not sure "
            "instead of hallucinating.\n\n"
            "Answer in clear, concise points. Where helpful, use short paragraphs or bullet points."
        )

        prompt = (
            f"{system_instructions}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        return prompt

    def generate_answer(self, prompt: str) -> str:
        """
        Generate an answer from the LLM given a prompt.
        """
        print("[RAG] Generating answer from LLM...")
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,  # prevent huge prompts on CPU
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Simple way to cut off the prompt from the generated text if it echoes
        if generated.startswith(prompt):
            generated = generated[len(prompt):].strip()

        return generated.strip()

    def answer(self, query: str) -> Dict[str, Any]:
        """
        Full RAG call: retrieve + generate.
        Returns both answer and retrieved chunks for inspection.
        """
        chunks = self.retrieve(query)
        prompt = self.build_prompt(query, chunks)
        answer = self.generate_answer(prompt)

        return {
            "question": query,
            "answer": answer,
            "chunks": [ch.__dict__ for ch in chunks],
        }


def cli():
    parser = argparse.ArgumentParser(description="UPSC RAG Chatbot CLI")
    parser.add_argument(
        "--question",
        type=str,
        help="Single question to ask the RAG pipeline.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive chat mode.",
    )
    args = parser.parse_args()

    config = RAGConfig()
    pipeline = RAGPipeline(config=config)

    if args.question:
        result = pipeline.answer(args.question)
        print("\nQ:", result["question"])
        print("\nAnswer:\n", result["answer"])
        print("\n--- Retrieved Chunks (debug) ---")
        for i, ch in enumerate(result["chunks"], start=1):
            print(f"\n[{i}] subject={ch['subject']}, source={ch['source_doc_id']}, score={ch['score']:.4f}")
            preview = ch["text"]
            if len(preview) > 400:
                preview = preview[:400] + "..."
            print(preview)
        return

    if args.interactive:
        print("=== UPSC RAG Chatbot (CLI) ===")
        print("Type your question (or 'exit' to quit)")
        while True:
            q = input("\nYou: ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                break
            result = pipeline.answer(q)
            print("\nBot:\n", result["answer"])
        return

    parser.print_help()


if __name__ == "__main__":
    cli()
