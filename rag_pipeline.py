import os
import re
import streamlit as st
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# ── Prompt ─────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """Answer ONLY from the context below.

Also provide timestamps if available.

If not found, say:
"I don't know. This information is not available in the video transcript."

Context:
{context}

Question: {question}

Answer:
"""

PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


def format_docs(docs):
    output = []
    for doc in docs:
        timestamp = doc.metadata.get("start", "N/A")
        content = doc.page_content
        output.append(f"[Time: {timestamp}] {content}")
    return "\n\n".join(output)


# ── Cache embeddings + vector DB ───────────────────────
@st.cache_resource(show_spinner=False)
def get_vectorstore(video_id, docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    db_path = f"faiss_store/{video_id}"

    if os.path.exists(db_path):
        return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(db_path)
    return vectorstore


# ── Extract video ID ───────────────────────────────────
def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([\w-]+)", url)
    return match.group(1)


# ── Main Pipeline ──────────────────────────────────────
def load_youtube_video(url: str):

    video_id = extract_video_id(url)

    loader = YoutubeLoader.from_youtube_url(
        url,
        add_video_info=False,
        language=["en"],
    )

    documents = loader.load()

    if not documents:
        raise ValueError("No transcript found.")

    video_title = documents[0].metadata.get("title", f"Video ({video_id})")

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(documents)

    # Vectorstore (cached + persistent)
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10}
    )

    # LLM (FIXED MODEL)
    llm = ChatGroq(
        model="llama-3.1-8b-instant",  # ✅ updated working model
        temperature=0,
        api_key=st.secrets["GROQ_API_KEY"],
    )

    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    return chain, video_title, video_id