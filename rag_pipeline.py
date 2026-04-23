import os
import warnings
import requests

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp


# ── Prompt ─────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """Answer ONLY from the context below.

If the answer is not found in the context, say:
"I don't know. This information is not available in the video transcript."

Context:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


# ── Helper: Extract video ID ────────────────────────────
def extract_video_id(url):
    if "youtu.be" in url:
        return url.split("/")[-1]
    return url.split("v=")[-1].split("&")[0]


# ── Transcript Fetcher (MAIN FIX 🔥) ─────────────────────
def get_transcript(url):
    video_id = extract_video_id(url)

    # ✅ Method 1: youtube_transcript_api
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t["text"] for t in transcript])
    except:
        pass

    # 🔁 Method 2: yt-dlp fallback (more reliable)
    try:
        ydl_opts = {
            "writesubtitles": True,
            "writeautomaticsub": True,
            "skip_download": True,
            "subtitleslangs": ["en"],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            subtitles = info.get("subtitles") or info.get("automatic_captions")

            if not subtitles or "en" not in subtitles:
                raise Exception("No English subtitles found")

            sub_url = subtitles["en"][0]["url"]

            data = requests.get(sub_url).text
            return data

    except:
        raise ValueError(
            "❌ Could not fetch transcript.\n\n"
            "Possible reasons:\n"
            "- Video has no captions\n"
            "- YouTube blocked request\n"
            "- Try another video (TED Talk recommended)"
        )


# ── Format docs ─────────────────────────────────────────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ── Main Pipeline ───────────────────────────────────────
def load_youtube_video(url: str):
    try:
        # 🔥 Use new transcript system
        text = get_transcript(url)

        if not text:
            raise ValueError("Transcript is empty.")

        documents = [Document(page_content=text)]
        video_title = f"YouTube Video ({extract_video_id(url)})"

        # Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
        )
        chunks = splitter.split_documents(documents)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )

        # Vector DB
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # LLM (UPDATED MODEL ✅)
        if "GROQ_API_KEY" not in st.secrets:
            raise ValueError("GROQ_API_KEY missing in secrets")

        llm = ChatGroq(
            model="llama-3.1-8b-instant",  # 🔥 FIXED MODEL
            temperature=0,
            api_key=st.secrets["GROQ_API_KEY"],
        )

        # Chain
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | PROMPT
            | llm
            | StrOutputParser()
        )

        return chain, video_title

    except Exception as e:
        raise RuntimeError(f"Error processing video: {str(e)}")