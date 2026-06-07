"""RAG chain construction and question-answer helpers.

This module builds the prompt, retriever, and LLM pipeline used for
chatting with the transcript through ChromaDB-backed retrieval.
"""

import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_retriever

def get_llm():
    # Keep the model centralized so every RAG prompt uses the same settings.
    return ChatMistralAI(model= "mistral-small-latest", mistral_api_key = os.getenv("MISTRAL_API_KEY"),temperature=0.2)


def format_docs(docs):
    # Convert retrieved documents into a compact prompt-friendly block.
    return "\n\n".join([doc.page_content for doc in docs])


def build_rag_chain(transcript : str):
    # Build a fresh vector store for the current transcript and connect it to the LLM.
    vector_store = build_vector_store(transcript)

    retriever = get_retriever(vector_store, k = 5)

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant. Answer the user's question based ONLY on the meeting transcript context provided below.

            If the answer is not found in the context, say:
            "I could not find this information in the meeting transcript."

        Always be concise and precise. If quoting someone, mention it clearly.

        Context from meeting transcript:{context}"""
        ),
        (
            "human","{question}"
        )
    ])
     
    ##full LCEL pipeline


    rag_chain = (

        {
            "context" : retriever|RunnableLambda(format_docs),
            "question" : RunnablePassthrough()
        } | prompt | llm | StrOutputParser()
    )
    return rag_chain

def load_rag_chain():
    # Load an existing persisted Chroma collection if one is already available.
    vector_store = load_vector_store()
    retriever = get_retriever()

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant. Answer the user's question based ONLY on the meeting transcript context provided below.

            If the answer is not found in the context, say:
            "I could not find this information in the meeting transcript."

        Always be concise and precise. If quoting someone, mention it clearly.

        Context from meeting transcript:{context}"""
        ),
        (
            "human","{question}"
        )
    ])
    rag_chain = (

        {
            "context" : retriever|RunnableLambda(format_docs),
            "question" : RunnablePassthrough()
        } | prompt | llm | StrOutputParser()
    )
    return rag_chain

def ask_question(rag_chain, question : str)->str :
    # Thin logging wrapper around the chain invoke call for CLI debugging.
    print(f"Question: {question}")
    answer = rag_chain.invoke(question)
    print(f"Answer : {answer}")
    return answer
