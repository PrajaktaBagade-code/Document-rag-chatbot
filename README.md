# DocuChat — RAG-Powered PDF Question-Answering Chatbot

Upload any PDF, ask questions in plain English, get grounded answers with page-level citations.

## Project Structur

```
docuchat/
├── app.py           # Streamlit UI — upload, chat, source display
├── ingest.py        # PDF loading, chunking, embeddings, FAISS index
├── rag_chain.py     # Retrieval + prompt + LLM chain (LCEL)
├── config.py        # All tunable settings in one place
├── requirements.txt
└── .env.example     # Copy to .env and fill in your keys
```
## Setup

```bash
pip install -r requirements.txt
cp .env.example .env     # then edit .env with your provider + API key
streamlit run app.py
```

- **Free path**: set `LLM_PROVIDER=mistral` and add a free `HUGGINGFACEHUB_API_TOKEN` from huggingface.co/settings/tokens.
- **OpenAI path**: set `LLM_PROVIDER=openai` and add `OPENAI_API_KEY`.
- Embeddings always run locally and for free (`sentence-transformers/all-MiniLM-L6-v2`), no key needed.

## How It Works (RAG Pipeline)

**1. Ingestion (`ingest.py`)**
- `PyPDFLoader` reads the PDF page-by-page, preserving page numbers as metadata.
- `RecursiveCharacterTextSplitter` breaks pages into ~1000-character chunks with 150-character overlap, splitting on paragraph → sentence → word boundaries so ideas aren't cut mid-thought.
- Each chunk is embedded with a local HuggingFace sentence-transformer (384-dim vectors, normalized for cosine similarity).
- Vectors are stored in a **FAISS** index, saved to disk keyed by a hash of the file's bytes — re-uploading the same PDF skips re-embedding entirely.

**2. Retrieval + Generation (`rag_chain.py`)**
- A **question condenser** step rewrites follow-up questions ("what about the second one?") into standalone questions using chat history, so retrieval doesn't lose context across turns.
- The standalone question is embedded and used to pull the top-K (default 4) most similar chunks from FAISS.
- Retrieved chunks are formatted into a numbered context block and injected into a strict prompt that instructs the LLM to answer **only** from the provided excerpts and say "I don't know" rather than hallucinate.
- The LLM (GPT-4o-mini or free Mistral-7B-Instruct) generates the final answer.

**3. UI (`app.py`)**
- Streamlit handles file upload, a persistent chat interface (`st.session_state`), and an expandable "Sources used" panel under each answer showing the exact page and excerpt the answer was grounded in.

## Data Flow Diagram

```
 PDF Upload
     │
     ▼
 PyPDFLoader (page-level Documents)
     │
     ▼
 RecursiveCharacterTextSplitter (overlapping chunks)
     │
     ▼
 HuggingFace Embeddings (MiniLM, local, free)
     │
     ▼
 FAISS Vector Store  ◄──── cached to disk by file hash
     │
     ▼
 User Question ──► Condense w/ chat history ──► Standalone Question
     │
     ▼
 Similarity Search (top-K chunks)
     │
     ▼
 Prompt = [Context chunks] + [Chat history] + [Question]
     │
     ▼
 LLM (GPT-4o-mini / Mistral-7B) ──► Answer + Source citations
     │
     ▼
 Streamlit Chat UI
```

## Why This Project Matters for AI Engineer Interviews

This single project demonstrates the full modern GenAI stack end-to-end:

- **Document processing**: parsing unstructured PDFs into structured, chunkable text.
- **Chunking strategy**: balancing chunk size/overlap for retrieval quality vs. context window cost.
- **Embeddings**: using open-source sentence-transformers instead of paying for every vectorization call.
- **Vector databases**: FAISS indexing, persistence, and similarity search.
- **RAG architecture**: separating retrieval from generation, and grounding LLM output to reduce hallucination.
- **Conversational memory**: handling multi-turn follow-ups via question condensing.
- **LLM orchestration**: LangChain's LCEL for composable, swappable chains (OpenAI ↔ free Mistral with one config change).
- **Product thinking**: citations, caching, and a usable chat UI — not just a notebook demo.

## Extending It

- Swap FAISS for a hosted vector DB (Pinecone, Weaviate, Qdrant) for multi-user, multi-document production use.
- Add re-ranking (e.g., Cohere Rerank) after initial retrieval to improve precision on longer documents.
- Support multi-file corpora by namespacing FAISS indices per collection instead of per single file.
- Stream tokens instead of waiting for the full answer, for a more responsive feel.
- Add evaluation (e.g., RAGAS) to measure faithfulness and answer relevancy over a test question set.
