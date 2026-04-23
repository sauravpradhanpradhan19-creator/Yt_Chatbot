# 🎬 YT ChatBot — RAG-based YouTube Q&A

Ask questions about any YouTube video using its transcript. Built with LangChain, Groq, HuggingFace Embeddings, FAISS, and Streamlit.

---
---

## 🎥 Demo Video

<video src="demo.mp4" controls width="700"></video>

---

## 🏗️ Architecture

```
YouTube URL
    │
    ▼
YoutubeLoader (LangChain)
    │  fetches transcript + metadata
    ▼
RecursiveCharacterTextSplitter
    │  chunk_size=1000, overlap=200
    ▼
HuggingFace Embeddings (all-MiniLM-L6-v2)
    │  384-dim vectors, free, runs on CPU
    ▼
FAISS Vector Store
    │  in-memory, fast similarity search
    ▼
MMR Retriever (k=4 chunks)
    │  fetches most relevant transcript chunks
    ▼
Groq LLM (llama3-8b-8192)
    │  temperature=0, strict RAG prompt
    ▼
Answer (or "I don't know" if not in transcript)
```

---

## 🚀 Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/yt-chatbot.git
cd yt-chatbot
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your Groq API Key
Get your free API key from https://console.groq.com

Create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "gsk_your_key_here"
```

> ⚠️ Never commit this file. It's already in `.gitignore`.

### 5. Run the app
```bash
streamlit run app.py
```

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push your project to a **public GitHub repo**
   ```bash
   git init
   git add .
   git commit -m "initial commit"
   git remote add origin https://github.com/yourusername/yt-chatbot.git
   git push -u origin main
   ```
   > Do NOT push `.streamlit/secrets.toml` — it's gitignored.

2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**

3. Connect your GitHub repo → select `app.py` as the main file

4. Click **Advanced settings → Secrets** and paste:
   ```
   GROQ_API_KEY = "gsk_your_key_here"
   ```

5. Click **Deploy** — your app will be live in ~2 minutes ✅

---

## 📁 Project Structure

```
yt-chatbot/
├── app.py                    # Streamlit UI
├── rag_pipeline.py           # Full RAG pipeline logic
├── requirements.txt          # Python dependencies
├── .gitignore
├── .streamlit/
│   ├── config.toml           # Streamlit theme config
│   └── secrets.toml.example  # Template (don't commit real one)
└── README.md
```

---

## ⚙️ Tuning Guide

| Parameter | Default | When to change |
|---|---|---|
| `chunk_size` | 1000 | Increase to 1500 for long videos |
| `chunk_overlap` | 200 | Increase to 300 for better context continuity |
| `k` (retriever) | 4 | Increase to 6-8 for complex multi-part questions |
| `temperature` | 0 | Keep at 0 for factual Q&A |
| `model_name` | llama3-8b-8192 | Switch to llama3-70b for harder questions |

---

## 🔑 API Keys Needed

| Service | Key | Cost |
|---|---|---|
| Groq | `GROQ_API_KEY` | Free tier available |
| HuggingFace Embeddings | None | Completely free |
| FAISS | None | Completely free |

---

## ⚠️ Known Limitations

- Videos must have **English captions** (auto-generated or manual)
- Private, age-restricted, or region-locked videos won't work
- Very long videos (2h+) may take 30-60s to index
- FAISS is in-memory — reloads on each session (use Chroma for persistence)

---

## 🛠️ Upgrade Path

| When you're ready to scale... | Switch to |
|---|---|
| Persist vectors across sessions | Chroma or Qdrant |
| Handle millions of chunks | Pinecone |
| Better answers on hard questions | llama3-70b-8192 |
| Better embeddings | BAAI/bge-large-en-v1.5 |
| Add chat history memory | ConversationalRetrievalChain |
| Multilingual videos | add more language codes to YoutubeLoader |
