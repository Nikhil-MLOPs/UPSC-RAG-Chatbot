# UPSC RAG Chatbot

A fully open-source, 0-cost **Retrieval-Augmented Generation (RAG)** chatbot for UPSC preparation.
It ingests UPSC PDFs (History, Geography, Polity, Economy, Science & Tech, Environment, etc.) and answers questions only from those sources.

## Key Goals

- 💸 **0 Cost** – No OpenAI, no paid APIs. All models are open-source and CPU-friendly.
- 🧠 **RAG Architecture** – Embedding-based retrieval + small open-source LLM.
- 🧪 **MLOps Ready** – Uses **DVC** for data/versioning and **MLflow** for experiment tracking.
- 📦 **Production-ish** – Dockerized, with **CI/CD (GitHub Actions)**, and deployed on **Hugging Face Spaces**.
- 💻 **Runs on CPU** – Designed for machines like i5 + 8 GB RAM.

## High-Level Architecture (Planned)

1. **Data Layer (with DVC)**  
   - PDFs in `data/raw/`  
   - Text extraction & chunking to `data/processed/`  
   - Embedding index in `data/indexes/` (FAISS/Chroma)

2. **RAG Core**
   - Embeddings via `sentence-transformers` (e.g. MiniLM)
   - Vector store for retrieval
   - Open LLM (TinyLlama/Qwen/etc) for answer generation

3. **Evaluation (with MLflow)**
   - UPSC-style question set
   - Metrics & experiment tracking (chunk size, top_k, models)

4. **Serving**
   - Gradio chat UI with streaming-style responses
   - Dockerized app
   - Hugging Face Spaces deployment

## Tech Stack

- **Language**: Python
- **RAG**: `transformers`, `sentence-transformers`, `faiss-cpu`
- **UI**: `gradio`
- **MLOps**: `dvc`, `mlflow`
- **Infra**: Docker, GitHub Actions, Hugging Face Spaces