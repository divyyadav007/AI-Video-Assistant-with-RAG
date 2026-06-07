# 🎥 AI Video Assistant with RAG

Transform any YouTube video, meeting recording, lecture, or audio file into searchable insights using AI-powered transcription, summarization, and Retrieval-Augmented Generation (RAG).

---

## 🚀 Overview

AI Video Assistant is an end-to-end GenAI application that converts video and audio content into actionable intelligence.

The system automatically:

* Transcribes videos using Whisper AI or Sarvam AI
* Generates concise AI-powered summaries
* Extracts action items, key decisions, and open questions
* Creates a searchable knowledge base using ChromaDB
* Enables conversational Q&A over video content using RAG

Built to demonstrate practical applications of:

* Retrieval-Augmented Generation (RAG)
* Vector Databases
* LLM Orchestration
* Speech-to-Text Pipelines
* Prompt Engineering
* AI-Powered Knowledge Retrieval

<img width="1919" height="916" alt="Screenshot 2026-06-07 203924" src="https://github.com/user-attachments/assets/e7c39a2e-3d03-4c76-b162-baffdfb050fc" />

---

## ✨ Features

### 🎙 Intelligent Transcription

* YouTube URL support
* Local audio/video file upload
* Automatic audio extraction
* Long-audio chunking support
* English transcription using Whisper
* Hindi/Hinglish transcription using Sarvam AI

### 📝 AI Meeting Intelligence

Generate:

* Executive Summary
* Action Items
* Key Decisions
* Open Questions
* Meeting Title

### 🔍 Retrieval-Augmented Generation

Chat directly with your video:

* Context-aware answers
* Semantic search
* ChromaDB vector storage
* Transcript-based retrieval
<img width="1919" height="917" alt="Screenshot 2026-06-07 203913" src="https://github.com/user-attachments/assets/d09463de-4372-4c46-ae24-f07368ee7d6c" />

### 📄 Export Options

* PDF Report Export
* TXT Export
* Transcript Download

---

## 🏗 Architecture

```text
YouTube URL / Video File
            │
            ▼
 Audio Extraction & Processing
            │
            ▼
      Audio Chunking
            │
            ▼
      Transcription Layer
   (Whisper / Sarvam AI)
            │
            ▼
      Full Transcript
            │
 ┌──────────┼──────────┐
 ▼          ▼          ▼
Summary  Decisions  Actions
            │
            ▼
      Vector Embeddings
     (MiniLM Embeddings)
            │
            ▼
         ChromaDB
            │
            ▼
      Retrieval Engine
            │
            ▼
         Mistral AI
            │
            ▼
      Conversational QA
```

---

## 🛠 Tech Stack

| Category         | Technology          |
| ---------------- | ------------------- |
| Language         | Python              |
| UI               | Streamlit           |
| LLM              | Mistral AI          |
| STT              | OpenAI Whisper      |
| Hindi STT        | Sarvam AI           |
| Framework        | LangChain           |
| Vector Database  | ChromaDB            |
| Embeddings       | all-MiniLM-L6-v2    |
| RAG              | LangChain Retrieval |
| PDF Export       | ReportLab           |
| Audio Processing | pydub + yt-dlp      |

---

## 📂 Project Structure

```text
AI-Video-Assistant/
│
├── app.py
├── main.py
├── requirements.txt
├── .env
│
├── core/
│   ├── transcriber.py
│   ├── summarizer.py
│   ├── extractor.py
│   ├── vector_store.py
│   └── rag_engine.py
│
├── utils/
│   └── audio_processor.py
│
├── downloads/
├── vectorDB/
└── README.md
```

---

## ⚡ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/AI-Video-Assistant-with-RAG.git

cd AI-Video-Assistant-with-RAG
```

### Create Virtual Environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux / Mac:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file:

```env
MISTRAL_API_KEY=your_key
SARVAM_API_KEY=your_key
```

---

## ▶ Run Application

### Streamlit UI

```bash
streamlit run app.py
```

### CLI Version

```bash
python main.py
```

---

## 💬 Example Questions

After processing a video, users can ask:

* What are the key takeaways?
* What decisions were made?
* What tasks were assigned?
* Summarize the entire discussion.
* What action items are pending?
* Explain this topic in simple terms.
<img width="1917" height="917" alt="Screenshot 2026-06-07 203900" src="https://github.com/user-attachments/assets/b6b634e2-4296-4cd4-b29e-6fe578966e43" />

---

## 🎯 Learning Outcomes

This project demonstrates:

* Retrieval-Augmented Generation (RAG)
* LangChain LCEL
* Vector Search
* Semantic Retrieval
* Whisper Integration
* Prompt Engineering
* ChromaDB
* Mistral AI
* Streamlit Application Development

---

## 🚧 Future Improvements

* Multi-language support
* Speaker diarization
* Real-time meeting transcription
* Cloud deployment
* Authentication system
* Team collaboration workspace
* Conversation memory
* Advanced analytics dashboard

---

## ⭐ Why This Project Matters

Most meeting recordings and lectures are difficult to revisit and search.

This project transforms passive video content into an interactive knowledge assistant that enables users to:

* Understand faster
* Search smarter
* Extract insights automatically
* Chat with their content using AI

---

## 📜 License

MIT License

---

## 👨‍💻 Author

Built by Yadav Ji as a hands-on GenAI, RAG, and LLM Engineering project.
