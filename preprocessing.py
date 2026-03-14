"""
preprocess.py
=============
Steps:
  a) PDF Text Extraction  → in-memory list (no relational DB)
  b) Text Chunking        → LangChain CharacterTextSplitter (chunk_size=500)
  c) Vector Datastore     → OpenAI Embeddings + Chroma Cloud

Environment variables required
-------------------------------
  OPENAI_API_KEY    your OpenAI API key
  CHROMA_API_KEY    your Chroma Cloud API key
  CHROMA_TENANT     your Chroma Cloud tenant name
  CHROMA_DATABASE   your Chroma Cloud database name

Usage
-----
  python preprocess.py
  python preprocess.py --pdf-folder ./my_pdfs
"""

import os
import glob
import argparse
from dotenv import load_dotenv

from pypdf import PdfReader

from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

import chromadb

load_dotenv()
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_PDF_FOLDER  = "./DSCI560_Lab9"
COLLECTION_NAME     = "Ads_cookbook"
CHUNK_SIZE          = 500
CHUNK_OVERLAP       = 50


# ===========================================================================
# a) PDF TEXT EXTRACTION
# ===========================================================================

def extract_pdfs(pdf_folder: str) -> list[dict]:
    """
    Iterate every PDF in pdf_folder, extract text page-by-page,
    and return a list of dicts: {filename, page_num, content}.
    """
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    if not pdf_files:
        print(f"[WARNING] No PDF files found in '{pdf_folder}'.")
        return []

    pages = []
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"  Extracting: {filename}")
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({
                    "filename": filename,
                    "page_num": page_num,
                    "content":  text,
                })

    print(f"  ✔ Extracted {len(pages)} pages from {len(pdf_files)} PDF(s).")
    return pages


# ===========================================================================
# b) GET TEXT CHUNKS
# ===========================================================================

def get_text_chunks(pages: list[dict]) -> list[str]:
    """
    Concatenate all page texts, then split into chunks of ~500 characters.

    CharacterTextSplitter parameters
    ---------------------------------
    separator      : split on newlines first so we don't break mid-sentence
    chunk_size     : target character count per chunk (500)
    chunk_overlap  : characters shared between adjacent chunks (50)
    length_function: len — counts characters, not tokens
    """
    full_text = "\n\n".join(p["content"] for p in pages)

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_text(full_text)
    print(f"  ✔ Created {len(chunks)} chunks (size≤{CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
    return chunks


# ===========================================================================
# c) CREATE VECTOR DATASTORE  (Chroma Cloud)
# ===========================================================================

def get_chroma_cloud_client() -> chromadb.CloudClient:
    """Build a Chroma Cloud client from environment variables."""

    client = chromadb.CloudClient(
    api_key='ck-5dgKymXwyiqkzbwh67P8EcvqBoWEeVxa4GYjticy7ofC',
    tenant='b2a6f32d-669f-4c1a-8525-c857a7d1e59e',
    database='DSCI560_Lab9'
    )
    return client

def create_vector_store(chunks: list[str]) -> None:
    """
    Embed each chunk with OpenAI text-embedding-ada-002 and upload to
    Chroma Cloud.
    """
    embeddings    = OpenAIEmbeddings(model="text-embedding-ada-002")
    chroma_client = get_chroma_cloud_client()

    for i in range(0, int(len(chunks)/300)):
        Chroma.from_texts(
            texts=chunks[i*300: (i+1)*300],
            embedding=embeddings,
            client=chroma_client,
            collection_name=COLLECTION_NAME,
        )
        print(f"  ✔ {len(chunks)} chunks uploaded to Chroma Cloud "
            f"(collection: '{COLLECTION_NAME}').")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess PDFs → Chroma Cloud")
    parser.add_argument(
        "--pdf-folder",
        default=DEFAULT_PDF_FOLDER,
        help=f"Folder containing PDF files (default: {DEFAULT_PDF_FOLDER})",
    )
    args = parser.parse_args()

    for var in ("OPENAI_API_KEY", "CHROMA_API_KEY", "CHROMA_TENANT", "CHROMA_DATABASE"):
        if not os.getenv(var):
            raise EnvironmentError(f"Environment variable '{var}' is not set.")

    print("\n=== Step a) Extracting PDFs ===")
    pages = extract_pdfs(args.pdf_folder)
    if not pages:
        return

    print("\n=== Step b) Chunking text ===")
    chunks = get_text_chunks(pages)

    print("\n=== Step c) Uploading to Chroma Cloud ===")
    create_vector_store(chunks)

    print("\n✅ Preprocessing complete. You can now run drive_function.py.\n")


if __name__ == "__main__":
    main()