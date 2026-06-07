"""Transcript summarization utilities.

This file creates compact meeting summaries and a short title from the
transcribed text using a map-reduce style LangChain workflow.
"""

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

import os


def get_llm():
    # Centralized summarization model config.
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3
    )


def split_transcript(transcript: str) -> list:
    # Split long transcripts into chunks before summarizing them.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    # Two-stage summary: summarize chunks first, then combine them.
    llm = get_llm()

    # Map Step
    map_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Summarize this portion of a meeting transcript concisely."
            ),
            (
                "human",
                "{text}"
            )
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    chunk_summaries = [
        map_chain.invoke({"text": chunk})
        for chunk in chunks
    ]

    combined = "\n\n".join(chunk_summaries)

    # Reduce Step
    combined_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert meeting summarizer. Combine these partial summaries into one final professional meeting summary in bullet points."
            ),
            (
                "human",
                "{text}"
            )
        ]
    )

    combined_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | combined_prompt
        | llm
        | StrOutputParser()
    )

    return combined_chain.invoke(combined)


def generate_title(transcript: str) -> str:
    # Ask the model for a short meeting title suitable for UI display.
    llm = get_llm()

    title_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Based on the meeting transcript, generate a short professional meeting title (maximum 8 words). Return only the title."
            ),
            (
                "human",
                "{text}"
            )
        ]
    )

    title_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | title_prompt
        | llm
        | StrOutputParser()
    )

    return title_chain.invoke(transcript[:2000])