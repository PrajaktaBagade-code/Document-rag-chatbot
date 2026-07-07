"""
ingest.py
---------
Everything needed to turn a raw PDF into a searchable FAISS vector index:

    PDF file  ->  page-level Documents  ->  overlapping text chunks
              ->  HuggingFace embeddings ->  FAISS index (persisted to disk)

The index is cached on disk keyed by a hash of the file's bytes, so
re-uploading the same PDF is instant on the second run.
"""

import hashlib
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def _file_hash(file_path: str) -> str:
    """Content-based hash so identical PDFs re-use the same cached index."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()[:16]


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Loads a free, local sentence-transformers model for embeddings.
    Runs on CPU — no API key, no cost, ~80MB download on first run.
    """
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # cosine sim via dot product
    )


def load_and_split(pdf_path: str):
    """Load a PDF page-by-page and split into overlapping chunks."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()  # one Document per page, with page metadata

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # tries paragraph -> sentence -> word
    )
    chunks = splitter.split_documents(pages)

    # tag each chunk with the source filename for citation display later
    filename = os.path.basename(pdf_path)
    for chunk in chunks:
        chunk.metadata["source_file"] = filename

    return chunks


def build_or_load_vectorstore(pdf_path: str, force_rebuild: bool = False) -> FAISS:
    """
    Returns a FAISS vectorstore for the given PDF.
    Rebuilds only if no cached index exists for this exact file content.
    """
    embeddings = get_embeddings()
    doc_id = _file_hash(pdf_path)
    index_dir = os.path.join(config.VECTORSTORE_DIR, doc_id)

    if os.path.exists(index_dir) and not force_rebuild:
        return FAISS.load_local(
            index_dir, embeddings, allow_dangerous_deserialization=True
        )

    chunks = load_and_split(pdf_path)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_dir)
    return vectorstore
