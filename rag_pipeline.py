import re
import warnings

warnings.filterwarnings("ignore")

import streamlit as st
import requests
import yt_dlp

from langchain_community.document_loaders import YoutubeLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# ── Prompt ─────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """Answer ONLY from the context below.

If the answer is not found, say:
"I don't know. This information is not available in the video transcript."

Context:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


# ── Helpers ────────────────────────────────────────────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def extract_video_id(url):
    patterns = [
        r"v=([a-zA-Z0-9_-]+)",
        r"youtu\.be/([a-zA-Z0-9_-]+)"
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return "unknown"


# 🔥 yt-dlp fallback (MAIN FIX)
def get_transcript_yt_dlp(url):
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "subtitlesformat": "json3",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

        subs = info.get("subtitles") or info.get("automatic_captions")
        if not subs:
            raise ValueError("No subtitles found via yt-dlp")

        en_subs = subs.get("en") or list(subs.values())[0]
        sub_url = en_subs[0]["url"]

        data = requests.get(sub_url).json()

        transcript = ""
        for event in data.get("events", []):
            for seg in event.get("segs", []):
                transcript += seg.get("utf8", "") + " "

        return transcript.strip(), info.get("title", "YouTube Video")


# ── MAIN FUNCTION ──────────────────────────────────────
def load_youtube_video(url: str):
    try:
        video_id = extract_video_id(url)

        # 1️⃣ Try normal loader
        try:
            loader = YoutubeLoader.from_youtube_url(
                url,
                add_video_info=False,
                language=["en"],
            )
            documents = loader.load()

            if not documents:
                raise ValueError("Empty transcript")

            video_title = documents[0].metadata.get("title", f"Video ({video_id})")

        # 2️⃣ Fallback to yt-dlp
        except Exception:
            transcript, video_title = get_transcript_yt_dlp(url)

            documents = [Document(page_content=transcript)]

        # 3️⃣ Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
        )
        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ValueError("Transcript empty after splitting")

        # 4️⃣ Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # 5️⃣ Vector store
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # 6️⃣ LLM
        if "GROQ_API_KEY" not in st.secrets:
            raise ValueError("GROQ_API_KEY missing")

        llm = ChatGroq(
            model="llama-3.1-8b-instant",   # ✅ correct working model
            temperature=0,
            api_key=st.secrets["GROQ_API_KEY"],
        )

        # 7️⃣ Chain
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | PROMPT
            | llm
            | StrOutputParser()
        )

        # ✅ MUST MATCH app.py
        return chain, video_title, video_id

    except Exception as e:
        raise RuntimeError(f"❌ Error processing video: {str(e)}")