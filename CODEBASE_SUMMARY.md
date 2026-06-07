AI Video Assistant with RAG — Codebase Summary

Overview
- Purpose: Transcribe audio/video (YouTube or local files), extract meeting-style artifacts (summary, action items, decisions, questions), and provide a retrieval-augmented chat interface over the transcript using a Chroma vector store.
- Primary entrypoint: `main.py` (CLI). Consider using `streamlit_app.py` for the new web UI.

Key modules
- `core/vector_store.py` — builds and loads a ChromaDB vector store from transcript chunks.
- `core/rag_engine.py` — creates an LLM + prompt pipeline for RAG-based chat.
- `core/transcriber.py` — wrappers around Whisper/Sarvam transcription logic.
- `core/summarizer.py` — generates summary and title.
- `core/extractor.py` — rule-based or LLM-based extraction of action items, decisions, questions.
- `utils/audio_processor.py` — handles YouTube/video download, audio extraction, and chunking.

Notes for future you
- Environment: Use a Python virtualenv and set required API keys via `.env` (e.g., `MISTRAL_API_KEY`, HF_TOKEN).
- Vector DB: Chroma persists to `vectorDB/` by default.
- Embeddings: `HuggingFaceEmbeddings` may show deprecation warnings; consider migrating to `langchain-huggingface`.
- Streamlit UI: `streamlit_app.py` provides a production-styled dashboard (dark mode). Run with `streamlit run streamlit_app.py`.

Troubleshooting
- If transcription stalls, ensure Whisper models are installed and CPU/GPU resources are sufficient.
- If Chroma init fails, check `chromadb` version compatibility and `persist_directory` permissions.

Quick commands
- Run CLI: `python main.py`
- Run UI: `streamlit run streamlit_app.py`

That's it — leave this file as an entry point for future maintenance.
