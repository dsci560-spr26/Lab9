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

DEFAULT_PDF_FOLDER  = "./DSCI560_Lab9"
COLLECTION_NAME     = "Ads_cookbook"
CHUNK_SIZE          = 500
CHUNK_OVERLAP       = 50

def extract_pdfs(pdf_folder: str) -> list[dict]:
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{pdf_folder}'.")
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

    print(f" Extracted {len(pages)} pages from {len(pdf_files)} PDF(s).")
    return pages

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
    print(f"Created {len(chunks)} chunks (size≤{CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
    return chunks

def get_chroma_cloud_client() -> chromadb.CloudClient:

    client = chromadb.CloudClient(
    api_key='ck-5dgKymXwyiqkzbwh67P8EcvqBoWEeVxa4GYjticy7ofC',
    tenant='b2a6f32d-669f-4c1a-8525-c857a7d1e59e',
    database='DSCI560_Lab9'
    )
    return client

def create_vector_store(chunks: list[str]) -> None:
    embeddings    = OpenAIEmbeddings(model="text-embedding-ada-002")
    chroma_client = get_chroma_cloud_client()

    for i in range(0, int(len(chunks)/300)):
        Chroma.from_texts(
            texts=chunks[i*300: (i+1)*300],
            embedding=embeddings,
            client=chroma_client,
            collection_name=COLLECTION_NAME,
        )
    print(f" {len(chunks)} chunks uploaded to Chroma Cloud "
        f"(collection: '{COLLECTION_NAME}').")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf-folder",
        default=DEFAULT_PDF_FOLDER,
        help=f"Folder containing PDF files (default: {DEFAULT_PDF_FOLDER})",
    )
    args = parser.parse_args()

    for var in ("OPENAI_API_KEY", "CHROMA_API_KEY", "CHROMA_TENANT", "CHROMA_DATABASE"):
        if not os.getenv(var):
            raise EnvironmentError(f"Environment variable '{var}' is not set.")

    print("\nExtracting PDFs...")
    pages = extract_pdfs(args.pdf_folder)
    if not pages:
        return

    print("\nChunking text...")
    chunks = get_text_chunks(pages)

    print("\nUploading to Chroma Cloud...")
    create_vector_store(chunks)

    print("\nPreprocessing complete. You can now run drive_function.py.\n")

if __name__ == "__main__":
    main()

"""
Usage
-----
  python preprocess.py
  python preprocess.py --pdf-folder ./my_pdfs
"""
