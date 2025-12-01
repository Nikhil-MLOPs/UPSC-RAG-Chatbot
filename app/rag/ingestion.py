from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from pypdf import PdfReader


RAW_DIR_DEFAULT = Path("data/raw")
PROCESSED_DIR_DEFAULT = Path("data/processed")


@dataclass
class Document:
    """Represents a single PDF document and its extracted text."""
    id: str
    subject: str
    source_path: Path
    text: str


def clean_text(text: str) -> str:
    """Basic cleaning: normalize whitespace and remove junk characters."""
    # Replace multiple whitespace with single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract raw text from a PDF file using pypdf."""
    reader = PdfReader(str(pdf_path), strict=False)

    pages_text: List[str] = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages_text.append(page_text)

    full_text = "\n".join(pages_text)
    return clean_text(full_text)


def load_pdfs(
    raw_dir: Path = RAW_DIR_DEFAULT,
    subjects: Optional[List[str]] = None,
) -> List[Document]:
    """
    Walk through data/raw and load PDFs into Document objects.

    Folder structure:
      data/raw/<subject>/*.pdf
    """
    documents: List[Document] = []

    if subjects is None:
        # Subject dirs are subfolders of raw_dir
        subjects = [
            d.name
            for d in raw_dir.iterdir()
            if d.is_dir()
        ]

    for subject in subjects:
        subject_dir = raw_dir / subject
        if not subject_dir.exists():
            continue

        for pdf_path in subject_dir.glob("*.pdf"):
            doc_id = f"{subject}_{pdf_path.stem}"
            print(f"[INGEST] Processing {pdf_path} -> id={doc_id}")
            text = extract_text_from_pdf(pdf_path)

            documents.append(
                Document(
                    id=doc_id,
                    subject=subject,
                    source_path=pdf_path,
                    text=text,
                )
            )

    return documents


def save_documents_as_text(
    documents: List[Document],
    processed_dir: Path = PROCESSED_DIR_DEFAULT,
) -> None:
    """
    Save each Document's text into data/processed/<subject>/<id>.txt.
    """
    for doc in documents:
        subject_dir = processed_dir / doc.subject
        subject_dir.mkdir(parents=True, exist_ok=True)

        out_path = subject_dir / f"{doc.id}.txt"
        print(f"[SAVE] Writing text to {out_path}")
        out_path.write_text(doc.text, encoding="utf-8")


def run_ingestion(
    raw_dir: Path = RAW_DIR_DEFAULT,
    processed_dir: Path = PROCESSED_DIR_DEFAULT,
) -> None:
    """
    End-to-end ingestion: load PDFs from raw_dir and save cleaned text into processed_dir.
    """
    print(f"[INGEST] Raw dir: {raw_dir.resolve()}")
    print(f"[INGEST] Processed dir: {processed_dir.resolve()}")

    docs = load_pdfs(raw_dir=raw_dir)
    print(f"[INGEST] Loaded {len(docs)} document(s).")

    save_documents_as_text(docs, processed_dir=processed_dir)
    print("[INGEST] Done.")


if __name__ == "__main__":
    run_ingestion()
