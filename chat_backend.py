from __future__ import annotations

from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parent
UPLOAD_ROOT = PROJECT_ROOT / "uploaded_pdfs"

MODE_OPTIONS = {
    "Open-Source": "open_source",
    "OpenAI": "openai",
}


def save_uploaded_pdfs(pdf_docs, mode: str) -> Path:
    mode_dir = UPLOAD_ROOT / mode
    if mode_dir.exists():
        shutil.rmtree(mode_dir)
    mode_dir.mkdir(parents=True, exist_ok=True)

    for pdf in pdf_docs:
        destination = mode_dir / pdf.name
        destination.write_bytes(pdf.getbuffer())

    return mode_dir


def process_pdfs(mode: str, pdf_docs) -> Path:
    pdf_folder = save_uploaded_pdfs(pdf_docs, mode)

    if mode == "open_source":
        from preprocessing_open import (
            extract_pdfs,
            get_text_chunks,
            create_vector_store,
        )
    elif mode == "openai":
        from preprocessing import (
            extract_pdfs,
            get_text_chunks,
            create_vector_store,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    pages = extract_pdfs(str(pdf_folder))
    if not pages:
        raise ValueError("No text could be extracted from the uploaded PDFs.")

    chunks = get_text_chunks(pages)
    if not chunks:
        raise ValueError("No chunks were created from the uploaded PDFs.")

    create_vector_store(chunks)
    return pdf_folder


def build_chain(mode: str):
    if mode == "open_source":
        from drive_function_open import build_chain as build_open_source_chain

        return build_open_source_chain()

    if mode == "openai":
        from drive_function import (
            OPENAI_MODEL,
            create_conversation_chain,
            load_vector_store,
        )
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
        vector_store = load_vector_store()
        return create_conversation_chain(llm, vector_store)

    raise ValueError(f"Unsupported mode: {mode}")
