"""Chroma vector store helpers for transcript retrieval.

This module turns transcript chunks into embeddings, stores them in Chroma,
and exposes a retriever for the RAG chat chain.
"""

import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


CHROMA_DIR = "vectorDB"
COLLECTION_NAME= "meeting_transcript"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def get_embeddings():
    # Keep embedding config in one place so it is easy to replace later.
    return HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL,
        model_kwargs = {"device" : "cpu"}
    )

def build_vector_store(transcript: str)->Chroma:
    # Split transcript, wrap each chunk as a Document, and persist to Chroma.
    print("Building vector store")


    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )

    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata = {'chunk_index': i})
        for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    return vector_store


def load_vector_store()-> Chroma:
    # Re-open the persisted Chroma collection for later chat sessions.
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name = COLLECTION_NAME,
        persist_directory = CHROMA_DIR,
        embedding_function = embeddings,

    )
    return vector_store

def get_retriever(vector_store:Chroma, k :int = 4):
    # Standard similarity retriever used by the RAG chain.
    return vector_store.as_retriever(
        search_type = 'similarity',
        search_kwargs = {"k":k}
    )