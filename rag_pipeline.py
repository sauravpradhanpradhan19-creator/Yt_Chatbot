import re
import warnings
import requests

warnings.filterwarnings("ignore")

import streamlit as st
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


def extract_video_id(url: str) -> str:
    patterns = [
        r"(?:v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return "unknown"


# ── Method 1: youtube-transcript-api ───────────────────
def get_transcript_api(video_id: str):
    """Direct transcript fetch — works locally, often blocked on cloud."""
    from youtube_transcript_api import YouTubeTranscriptApi

    transcript_list = YouTubeTranscriptApi.get_transcript(
        video_id, languages=["en", "en-US", "en-GB"]
    )
    text = " ".join(t["text"] for t in transcript_list)
    return text


# ── Method 2: yt-dlp subtitle extraction ───────────────
def get_transcript_ytdlp(url: str):
    """
    Uses yt-dlp to grab auto-generated subtitles.
    Works on Streamlit Cloud even when transcript-api is blocked.
    """
    import yt_dlp

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "writesubtitles": False,
        "writeautomaticsub": False,
        "subtitleslangs": ["en"],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    title = info.get("title", "YouTube Video")

    # Prefer manual subs, fall back to auto-captions
    subs = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}

    en_entries = (
        subs.get("en")
        or subs.get("en-US")
        or auto.get("en")
        or auto.get("en-US")
        or None
    )

    if not en_entries:
        # Try first available language
        for lang_data in (list(subs.values()) + list(auto.values())):
            if lang_data:
                en_entries = lang_data
                break

    if not en_entries:
        raise ValueError("No subtitles or auto-captions found for this video.")

    # Pick json3 format if available, else first format
    sub_url = None
    for entry in en_entries:
        if entry.get("ext") == "json3":
            sub_url = entry["url"]
            break
    if not sub_url:
        sub_url = en_entries[0]["url"]

    resp = requests.get(sub_url, timeout=15)
    resp.raise_for_status()

    # Parse json3
    data = resp.json()
    parts = []
    for event in data.get("events", []):
        for seg in event.get("segs", []):
            txt = seg.get("utf8", "").strip()
            if txt and txt != "\n":
                parts.append(txt)

    transcript = " ".join(parts).strip()
    if not transcript:
        raise ValueError("Subtitles were found but appear to be empty.")

    return transcript, title


# ── MAIN FUNCTION ──────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_youtube_video(url: str):
    """
    Returns (chain, title, video_id).
    Tries youtube-transcript-api first, then yt-dlp as fallback.
    """
    video_id = extract_video_id(url)
    transcript = None
    title = f"Video ({video_id})"
    errors = []

    # — Attempt 1: youtube-transcript-api —
    try:
        transcript = get_transcript_api(video_id)
        title = f"Video ({video_id})"   # API doesn't return title
    except Exception as e1:
        errors.append(f"transcript-api: {e1}")

    # — Attempt 2: yt-dlp —
    if not transcript:
        try:
            transcript, title = get_transcript_ytdlp(url)
        except Exception as e2:
            errors.append(f"yt-dlp: {e2}")

    if not transcript:
        raise RuntimeError(
            "Could not retrieve transcript.\n" + "\n".join(errors)
        )

    # — Split —
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    docs = [Document(page_content=transcript)]
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError("Transcript is empty after splitting.")

    # — Embeddings (cached on disk by sentence-transformers) —
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    # — Vector store —
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # — LLM —
    if "GROQ_API_KEY" not in st.secrets:
        raise ValueError("GROQ_API_KEY is missing from Streamlit secrets.")

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=st.secrets["GROQ_API_KEY"],
    )

    # — Chain —
    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    return chain, title, video_id