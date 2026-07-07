"""
rag_chain.py
------------
Wires the retriever to an LLM using LangChain's modern LCEL (LangChain
Expression Language) syntax, with:

  - Question condensing: rewrites a follow-up question ("what about page 2?")
    into a standalone question using chat history, so retrieval doesn't miss
    context from earlier turns.
  - A strict prompt that forces grounded, "I don't know" honest answers.
  - Source documents returned alongside the answer for citation display.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableBranch

import config

# ---------------------------------------------------------------------------
# LLM factory — swap providers via config.LLM_PROVIDER, no other code changes
# ---------------------------------------------------------------------------
def get_llm():
    if config.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.2,
            api_key=config.OPENAI_API_KEY,
        )
    else:
        # Free tier: Mistral-7B-Instruct via HuggingFace Inference API
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

        endpoint = HuggingFaceEndpoint(
            repo_id=config.MISTRAL_MODEL,
            huggingfacehub_api_token=config.HUGGINGFACEHUB_API_TOKEN,
            temperature=0.2,
            max_new_tokens=512,
        )
        return ChatHuggingFace(llm=endpoint)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's latest message as a standalone question that "
            "makes sense without the chat history. Do not answer it, only "
            "rewrite it. If it is already standalone, return it unchanged.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are DocuChat, an assistant that answers questions ONLY using "
            "the provided document excerpts below. Rules:\n"
            "1. If the excerpts don't contain the answer, say you don't know — "
            "never invent facts.\n"
            "2. Keep answers concise and directly grounded in the excerpts.\n"
            "3. Quote sparingly; prefer summarizing in your own words.\n\n"
            "Document excerpts:\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)


def _format_docs(docs) -> str:
    """Turns retrieved chunks into a numbered context block for the prompt."""
    formatted = []
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source_file", "document")
        formatted.append(f"[{i}] (source: {source}, page {page})\n{doc.page_content}")
    return "\n\n".join(formatted)


def build_rag_chain(vectorstore):
    """
    Returns a callable chain: invoke({"question": ..., "chat_history": [...]})
    -> {"answer": str, "source_documents": [Document, ...]}
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": config.TOP_K})
    llm = get_llm()

    # Step 1: condense question using history (skip if no history yet)
    condense_chain = CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()

    has_history = lambda x: len(x.get("chat_history", [])) > 0
    standalone_question = RunnableBranch(
        (has_history, condense_chain),
        RunnablePassthrough() | (lambda x: x["question"]),
    )

    def run(inputs: dict) -> dict:
        question = standalone_question.invoke(inputs)
        docs = retriever.invoke(question)
        context = _format_docs(docs)

        answer_chain = ANSWER_PROMPT | llm | StrOutputParser()
        answer = answer_chain.invoke(
            {
                "context": context,
                "question": question,
                "chat_history": inputs.get("chat_history", []),
            }
        )
        return {"answer": answer, "source_documents": docs}

    return run
