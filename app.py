"""
app.py 
------
Streamlit front-end for DocuChat.

Run with:  streamlit run app.py
"""

import os

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

import config
from ingest import build_or_load_vectorstore
from rag_chain import build_rag_chain

st.set_page_config(page_title="DocuChat", page_icon="📄", layout="wide")
st.title("📄 DocuChat — Ask Questions About Any PDF")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list[HumanMessage | AIMessage]
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

# ---------------------------------------------------------------------------
# Sidebar: upload + settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Upload a PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    st.header("2. LLM backend")
    st.caption(f"Currently configured: **{config.LLM_PROVIDER}** (edit .env to change)")

    process_clicked = st.button("Process document", type="primary", disabled=uploaded_file is None)

    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Process uploaded PDF -> build/load vectorstore -> build RAG chain
# ---------------------------------------------------------------------------
if uploaded_file and process_clicked:
    save_path = os.path.join(config.UPLOAD_DIR, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Reading, chunking, and embedding your PDF..."):
        vectorstore = build_or_load_vectorstore(save_path)
        st.session_state.rag_chain = build_rag_chain(vectorstore)
        st.session_state.processed_file = uploaded_file.name
        st.session_state.chat_history = []

    st.success(f"'{uploaded_file.name}' is ready. Ask away!")

# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------
if st.session_state.processed_file:
    st.caption(f"Currently chatting with: **{st.session_state.processed_file}**")

for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

question = st.chat_input(
    "Ask something about the document..."
    if st.session_state.rag_chain
    else "Upload and process a PDF first"
)

if question:
    if not st.session_state.rag_chain:
        st.warning("Please upload and process a PDF before asking questions.")
    else:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.rag_chain(
                    {
                        "question": question,
                        "chat_history": st.session_state.chat_history,
                    }
                )
                st.markdown(result["answer"])

                with st.expander("📚 Sources used"):
                    for i, doc in enumerate(result["source_documents"], start=1):
                        page = doc.metadata.get("page", "?")
                        st.markdown(f"**[{i}] Page {page}**")
                        st.caption(doc.page_content[:300] + "...")

        st.session_state.chat_history.append(HumanMessage(content=question))
        st.session_state.chat_history.append(AIMessage(content=result["answer"]))
