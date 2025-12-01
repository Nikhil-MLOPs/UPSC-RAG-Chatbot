from __future__ import annotations

from typing import List, Tuple, Generator

import gradio as gr

from app.rag.config import RAGConfig
from app.rag.pipeline import RAGPipeline

# Global singleton to avoid reloading models on every request
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        config = RAGConfig()
        _pipeline = RAGPipeline(config=config)
    return _pipeline


def generate_streaming_answer(
    message: str,
    history: List[Tuple[str, str]],
) -> Generator[str, None, None]:
    """
    Gradio ChatInterface handler.
    Yields partial responses to simulate streaming.
    """
    pipeline = get_pipeline()

    # We only use the latest message for retrieval; history is just for display.
    result = pipeline.answer(message)
    answer = result["answer"]

    # Build a tiny sources summary
    sources = []
    for ch in result["chunks"]:
        tag = f"{ch['subject']}|{ch['source_doc_id']}"
        if tag not in sources:
            sources.append(tag)
    if sources:
        sources_text = "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources[:5])
        full_answer = answer.strip() + sources_text
    else:
        full_answer = answer.strip()

    # Stream out the answer in small pieces
    step = 40  # characters per update
    for i in range(step, len(full_answer) + step, step):
        yield full_answer[:i]


def create_interface() -> gr.Blocks:
    """
    Create and return the Gradio interface for the UPSC RAG Chatbot.
    """
    description = (
        "Ask UPSC-style questions (History, Geography, Polity, Economy, Science & Tech, Environment, etc.).\n"
        "The chatbot answers *only* from the curated UPSC PDFs ingested into its RAG pipeline."
    )

    chat = gr.ChatInterface(
        fn=generate_streaming_answer,
        title="UPSC RAG Chatbot",
        description=description,
    )


    # Wrap in Blocks for future custom controls if needed
    with gr.Blocks() as demo:
        gr.Markdown("# UPSC RAG Chatbot")
        gr.Markdown(
            "A 0-cost, CPU-friendly Retrieval-Augmented Generation (RAG) chatbot built on open-source models.\n"
            "It uses your UPSC PDFs as its only source of truth."
        )
        chat.render()

    return demo
