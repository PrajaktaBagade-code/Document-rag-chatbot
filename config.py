"""
config.py
---------
Central place for every tunable setting in DocuChat.
Keeping these in one file makes it trivial to swap embedding models,
change chunking strategy, or point at a different LLM backend without
touching the pipeline logic itself.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # pulls OPENAI_API_KEY / HUGGINGFACEHUB_API_TOKEN from a .env file

# ---------------------------------------------------------------------------
# LLM backend: "openai" or "mistral" (free, via HuggingFace Inference API)
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mistral")  # default to the free option

OPENAI_MODEL = "gpt-4o-mini"
MISTRAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ---------------------------------------------------------------------------
# Embeddings — free, local, runs on CPU
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Chunking strategy
# ---------------------------------------------------------------------------
CHUNK_SIZE = 1000        # characters per chunk
CHUNK_OVERLAP = 150      # overlap so context isn't lost at chunk boundaries

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K = 4                # number of chunks retrieved per question

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
VECTORSTORE_DIR = "vectorstore"   # FAISS index persisted here, keyed by doc hash
UPLOAD_DIR = "uploaded_pdfs"

os.makedirs(VECTORSTORE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
