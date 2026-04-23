import streamlit as st
from rag_pipeline import load_youtube_video
import re

st.set_page_config(page_title="YT ChatBot", page_icon="🎬", layout="wide")

# ── UI Styling ─────────────────────────────────────────
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}

.block-container {
    max-width: 900px;
    margin: auto;
    padding-top: 2rem;
}

.title {
    text-align: center;
    font-size: 36px;
    font-weight: 700;
}

.subtitle {
    text-align: center;
    color: #9ca3af;
    margin-bottom: 20px;
}

/* FIX BUTTON VISIBILITY */
.stButton button {
    background-color: #2563eb !important;
    color: white !important;
    border-radius: 8px;
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
    margin-left: auto;
}

.bot {
    background: #1f2937;
    margin-right: auto;
}

.chat-container {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 15px;
    margin-top: 20px;
}

/* Sticky input */
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
st.markdown('<div class="subtitle">Chat with any YouTube video</div>', unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────
for key in ["rag_chain", "chat_history", "video_loaded", "video_title", "video_id"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "chat_history" else None

# ── Input ──────────────────────────────────────────────
col1, col2 = st.columns([5, 1])

with col1:
    yt_url = st.text_input("", placeholder="Paste YouTube link...")

with col2:
    load_btn = st.button("Load")

# ── Load Video ─────────────────────────────────────────
if load_btn and yt_url:
    if not re.match(r"(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+", yt_url):
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
    st.image(thumbnail)
    st.success(st.session_state.video_title)

# ── Chat ───────────────────────────────────────────────
if st.session_state.video_loaded:

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        role = "user" if msg["role"] == "user" else "bot"
        icon = "🧑" if role == "user" else "🤖"

        st.markdown(f"""
        <div class="chat {role}">
        {icon} {msg["content"]}
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    query = st.chat_input("Ask about the video...")

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.spinner("Thinking..."):
            try:
                answer = st.session_state.rag_chain.invoke(query)
            except Exception as e:
                answer = f"Error: {str(e)}"

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

else:
    st.info("Paste a YouTube link to start 🚀")