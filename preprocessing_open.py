import os
import glob
import argparse
import shutil
from pathlib import Path

from pypdf import PdfReader
from dotenv import load_dotenv

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PDF_FOLDER = str(PROJECT_ROOT / "DSCI560_Lab9")
COLLECTION_NAME = "Ads_cookbook"
PERSIST_DIRECTORY = PROJECT_ROOT / "chroma_db_open"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
BATCH_SIZE = 300
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def extract_pdfs(pdf_folder: str) -> list[dict]:
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{pdf_folder}'.")
        return []

    pages = []
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"Extracting: {filename}")
        reader = PdfReader(pdf_path)

        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(
                    {
                        "filename": filename,
                        "page_num": page_num,
                        "content": text,
                    }
                )

    print(f"Extracted {len(pages)} pages from {len(pdf_files)} PDF(s).")
    return pages


def get_text_chunks(pages: list[dict]) -> list[dict]:
    """
    Split each page into chunks so we preserve filename/page metadata.
    """
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    chunks = []
    for page in pages:
        page_chunks = splitter.split_text(page["content"])
        for idx, chunk_text in enumerate(page_chunks):
            chunk_text = chunk_text.strip()
            if chunk_text:
                chunks.append(
                    {
                        "content": chunk_text,
                        "metadata": {
                            "filename": page["filename"],
                            "page_num": page["page_num"],
                            "chunk_num": idx + 1,
                        },
                    }
                )

    print(
        f"Created {len(chunks)} chunks "
        f"(target size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})."
    )
    return chunks


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
    )


def reset_persist_directory(persist_directory: str) -> None:
    path = Path(persist_directory)
    if path.exists():
        shutil.rmtree(path)


def create_vector_store(chunks: list[dict]) -> None:
    if not chunks:
        print("No chunks to upload.")
        return

    embeddings = get_embeddings()

    # Start fresh each time so reruns do not duplicate data.
    reset_persist_directory(PERSIST_DIRECTORY)

    texts = [item["content"] for item in chunks]
    metadatas = [item["metadata"] for item in chunks]

    persist_directory = str(PERSIST_DIRECTORY)
    print(f"Saving vectors locally in '{persist_directory}' ...")

    vectorstore = None
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        batch_metadatas = metadatas[i : i + BATCH_SIZE]

        if i == 0:
            vectorstore = Chroma.from_texts(
                texts=batch_texts,
                embedding=embeddings,
                metadatas=batch_metadatas,
                collection_name=COLLECTION_NAME,
                persist_directory=persist_directory,
            )
        else:
            vectorstore.add_texts(
                texts=batch_texts,
                metadatas=batch_metadatas,
            )

        print(f"Uploaded {min(i + BATCH_SIZE, len(texts))}/{len(texts)} chunks")

    # Persist to disk
    vectorstore.persist()

    print(
        f"Done. {len(texts)} chunks saved to Chroma "
        f"(collection: '{COLLECTION_NAME}', directory: '{persist_directory}')."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf-folder",
        default=DEFAULT_PDF_FOLDER,
        help=f"Folder containing PDF files (default: {DEFAULT_PDF_FOLDER})",
    )
    args = parser.parse_args()

    pdf_folder = args.pdf_folder
    if not os.path.isdir(pdf_folder):
        raise FileNotFoundError(f"PDF folder not found: {pdf_folder}")

    print("\nExtracting PDFs...")
    pages = extract_pdfs(pdf_folder)
    if not pages:
        return

    print("\nChunking text...")
    chunks = get_text_chunks(pages)

    print("\nCreating vector store...")
    create_vector_store(chunks)

    print("\nPreprocessing complete.\n")


if __name__ == "__main__":
    main()
