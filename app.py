import streamlit as st
from rag_pipeline import load_youtube_video
import re

st.set_page_config(page_title="YT ChatBot", page_icon="🎬", layout="wide")

# ── Custom UI Styling ──────────────────────────────────
st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
}

/* Center container */
.block-container {
    max-width: 900px;
    margin: auto;
    padding-top: 2rem;
}

/* Header */
.title {
    text-align: center;
    font-size: 36px;
    font-weight: 700;
    color: white;
}

.subtitle {
    text-align: center;
    color: #9ca3af;
    margin-bottom: 20px;
}

/* Input box */
.stTextInput input {
    border-radius: 10px;
}

/* Chat bubbles */
.chat {
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 10px;
    max-width: 75%;
}

.user {
    background: #2563eb;
    color: white;
    margin-left: auto;
}

.bot {
    background: #1f2937;
    color: white;
    margin-right: auto;
}

/* Chat container */
.chat-container {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 15px;
    margin-top: 20px;
}

/* Button */
.stButton button {
    border-radius: 10px;
    background: #2563eb;
    color: white;
}

/* Sticky chat input */
[data-testid="stChatInput"] {
    position: fixed;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    width: 60%;
}

</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────
st.markdown('<div class="title">🎬 YouTube AI ChatBot</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Chat with any YouTube video using AI</div>', unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "video_loaded" not in st.session_state:
    st.session_state.video_loaded = False
if "video_title" not in st.session_state:
    st.session_state.video_title = ""
if "video_id" not in st.session_state:
    st.session_state.video_id = ""

# ── Input Section ──────────────────────────────────────
col1, col2 = st.columns([5, 1])

with col1:
    yt_url = st.text_input("", placeholder="Paste YouTube link here...")

with col2:
    load_btn = st.button("Load")

# ── Load Video ─────────────────────────────────────────
if load_btn and yt_url:
    yt_pattern = r"(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+"

    if not re.match(yt_pattern, yt_url):
        st.error("Invalid URL")
    else:
        with st.spinner("Processing video..."):
            try:
                chain, title, vid = load_youtube_video(yt_url)

                st.session_state.rag_chain = chain
                st.session_state.video_loaded = True
                st.session_state.video_title = title
                st.session_state.video_id = vid
                st.session_state.chat_history = []

            except Exception as e:
                st.error(str(e))

# ── Video Section ──────────────────────────────────────
if st.session_state.video_loaded:
    st.markdown("### 🎥 Video")

    thumbnail = f"https://img.youtube.com/vi/{st.session_state.video_id}/0.jpg"
    st.image(thumbnail, use_container_width=True)
    st.success(st.session_state.video_title)

# ── Chat Section ───────────────────────────────────────
if st.session_state.video_loaded:

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        role_class = "user" if msg["role"] == "user" else "bot"
        icon = "🧑" if msg["role"] == "user" else "🤖"

        st.markdown(f"""
        <div class="chat {role_class}">
        {icon} {msg["content"]}
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    query = st.chat_input("Ask about the video...")

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.spinner("🤖 Thinking..."):
            answer = st.session_state.rag_chain.invoke(query)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

else:
    st.info("Paste a YouTube link to start 🚀")