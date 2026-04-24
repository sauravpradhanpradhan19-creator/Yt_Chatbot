🎬 YT ChatBot — Chat with Any YouTube Video
Ever watched a long YouTube video and wished you could just ask it questions? That's exactly what this does.
Paste any YouTube link, and you can have a full conversation with the video — ask for summaries, specific details, timestamps, anything. Built with LangChain, Groq's blazing-fast LLaMA 3, HuggingFace Embeddings, FAISS, and Streamlit.

🚀 Try it Live
👉 ytchatbot-hxtjukurb37mjraxajjsra.streamlit.app
No setup needed — just paste a YouTube URL and start chatting.

🎥 Demo

Paste a YouTube link → click Load → ask anything about the video

Examples of what you can ask:

"Summarize this video in 3 bullet points"
"What did he say about X?"
"What are the main takeaways?"
"Explain the part about Y in simple terms"


🏗️ How It Works
YouTube URL
    │
    ▼
Transcript Fetcher
    │  tries youtube-transcript-api first
    │  falls back to yt-dlp if blocked
    ▼
RecursiveCharacterTextSplitter
    │  chunk_size=800, overlap=100
    ▼
HuggingFace Embeddings (all-MiniLM-L6-v2)
    │  384-dim vectors, free, runs on CPU
    ▼
FAISS Vector Store
    │  in-memory, fast similarity search
    ▼
Retriever (k=3 most relevant chunks)
    │
    ▼
Groq LLM (llama-3.1-8b-instant)
    │  temperature=0, answers only from transcript
    ▼
Answer (or "I don't know" if not in transcript)
The app only answers from the video transcript — it won't make things up. If the answer isn't in the video, it tells you.

🛠️ Run It Locally
1. Clone the repo
bashgit clone https://github.com/yourusername/yt-chatbot.git
cd yt-chatbot
2. Create a virtual environment
bashpython -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
3. Install dependencies
bashpip install -r requirements.txt
4. Add your Groq API Key
Get a free key from console.groq.com — it takes 30 seconds.
Create .streamlit/secrets.toml:
tomlGROQ_API_KEY = "gsk_your_key_here"

⚠️ Never commit this file. Add .streamlit/secrets.toml to your .gitignore.

5. Start the app
bashstreamlit run app.py
Open http://localhost:8501 and you're good to go.

☁️ Deploy Your Own (Free)
Want to host your own version? It's free on Streamlit Cloud.

Push your code to a public GitHub repo:

bash   git init
   git add .
   git commit -m "initial commit"
   git remote add origin https://github.com/yourusername/yt-chatbot.git
   git push -u origin main

Go to share.streamlit.io → New app
Connect your GitHub repo → select app.py as the entry point
Go to Advanced settings → Secrets and add:

   GROQ_API_KEY = "gsk_your_key_here"

Hit Deploy — live in ~2 minutes ✅


📁 Project Structure
yt-chatbot/
├── app.py                    # Streamlit UI
├── rag_pipeline.py           # RAG pipeline (transcript → answer)
├── requirements.txt          # Python dependencies
├── .gitignore
├── .streamlit/
│   ├── config.toml           # Theme config
│   └── secrets.toml          # Your API keys (never commit this)
└── README.md

⚙️ Tuning Parameters
ParameterDefaultWhen to changechunk_size800Increase to 1200+ for long videoschunk_overlap100Increase for better context continuityk (retriever)3Increase to 5-6 for complex questionstemperature0Keep at 0 for factual answersmodelllama-3.1-8b-instantSwitch to llama-3.1-70b for harder questions

🔑 API Keys
ServiceKey RequiredCostGroqGROQ_API_KEYFree tier availableHuggingFace EmbeddingsNoneCompletely freeFAISSNoneCompletely free

⚠️ Known Limitations

Videos need English captions (auto-generated is fine)
Private, age-restricted, or region-locked videos won't work
Very long videos (2h+) may take 30–60s to load
The vector store is in-memory — it resets when the session ends
Streamlit Cloud IPs are sometimes blocked by YouTube, which is why we use a yt-dlp fallback


🔮 What's Next
FeatureHowPersistent vector storageSwitch FAISS → Chroma or QdrantMulti-language supportAdd more language codes to the transcript fetcherBetter answersUpgrade to llama-3.1-70bBetter embeddingsTry BAAI/bge-large-en-v1.5Chat memoryAdd ConversationBufferMemory to the chainHandle millions of chunksMove to Pinecone

🤝 Contributing
Found a bug or want to add a feature? PRs are welcome. Open an issue first if it's a big change.
