# 🧬 CuraRAG  
### Local Offline RAG System for Medical & Technical Video Understanding

CuraRAG is a **local Retrieval-Augmented Generation (RAG)** system designed to convert long medical or technical videos into searchable, queryable knowledge — all running **fully offline**, using **local LLMs** and a local embedding + vector storage workflow.

It is built specifically for students, researchers, and developers who want to convert educational videos into an interactive knowledge base **without cloud APIs**.

---

# ✨ Features

### 🎥 Video-to-Text Pipeline (Local)
- Extract audio from video  
- Transcribe using locally available models (e.g., Whisper or local ASR engines)  
- Save time-stamped transcripts  

### 🧩 Semantic Chunking
- Split the transcript into small, meaningful chunks  
- Maintain timestamps for grounding  

### 🔍 Embedding + Vector Index (Local)
- Embeddings generated using local embedding models  
- Stored inside a vector index (filesystem-based, simple DB, or any vector store — user’s choice)  
- No cloud, no external services  

### 🤖 Local LLM Question Answering
- User asks a question  
- System retrieves relevant chunks  
- Local LLM generates grounded answers  
- **No internet required**  

### 🛡 No Hallucinations
- Answers include **citation timestamps**  
- Only retrieved chunks are used for generation  

---

# 🛠 Technology Stack (Offline)

| Component | Options (Your choice) |
|----------|------------------------|
| Transcription | Local Whisper / Local ASR model |
| Embeddings | Local embedding models (SentenceTransformers, Instructor, etc.) |
| Vector Store | Any local vector DB (FAISS / Chroma / custom simple index) |
| LLM | Local model (Deepseek, Llama, Mistral, etc.) |
| Runtime | Python scripts

---

# ⚙ How CuraRAG Works

