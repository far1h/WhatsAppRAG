import json
import os
from pathlib import Path

from chromadb import PersistentClient
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential

from whatsapp_rag.model_config import (
    DASHSCOPE_COMPATIBLE_BASE_URL,
    answer_model,
    chat_api_base_url,
    chat_api_key,
    query_model,
)


load_dotenv(override=True)

DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")

collection_name = "docs"
embedding_provider = os.getenv(
    "EMBEDDING_PROVIDER",
    "dashscope" if os.getenv("DASHSCOPE_API_KEY") and not os.getenv("OPENAI_API_KEY") else "openai",
).lower()
embedding_model = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-v4" if embedding_provider == "dashscope" else "text-embedding-3-large",
)
wait = wait_exponential(multiplier=1, min=10, max=240)

chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_or_create_collection(collection_name)

RETRIEVAL_K = 20
FINAL_K = 10

SYSTEM_PROMPT = """
You answer questions about the user's WhatsApp chat history.

Use only the retrieved chat excerpts as evidence. Be specific about who said what and when when the evidence supports it. Pay close attention to dates, times, message order, people, plans, locations, and follow-up context.

If the excerpts do not contain enough evidence, say that you do not know from the retrieved chat context. Do not invent private feelings, motives, events, or relationship conclusions that are not directly supported by the messages.

Retrieved chat context:
{context}

Answer the user's question accurately and concisely.
"""


def make_embedding_client() -> OpenAI:
    """Create an OpenAI-compatible embedding client."""
    if embedding_provider == "dashscope":
        return OpenAI(
            api_key=os.getenv("EMBEDDING_API_KEY") or os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("EMBEDDING_API_BASE")
            or os.getenv("DASHSCOPE_API_BASE")
            or DASHSCOPE_COMPATIBLE_BASE_URL,
        )

    kwargs = {}
    if os.getenv("EMBEDDING_API_KEY"):
        kwargs["api_key"] = os.getenv("EMBEDDING_API_KEY")
    if os.getenv("EMBEDDING_API_BASE"):
        kwargs["base_url"] = os.getenv("EMBEDDING_API_BASE")
    return OpenAI(**kwargs)


openai = make_embedding_client()
chat_client = OpenAI(api_key=chat_api_key(), base_url=chat_api_base_url())


class Result(BaseModel):
    page_content: str
    metadata: dict


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="Chunk ids ranked from most relevant to least relevant."
    )


@retry(wait=wait)
def rerank(question: str, chunks: list[Result]) -> list[Result]:
    system_prompt = """
You are reranking WhatsApp conversation chunks for a chat-history RAG system.

Rank chunks by usefulness for answering the user's question. Prefer chunks that match the requested person, topic, date/time reference, plan, location, or conversation sequence. Include every chunk id exactly once.

Return JSON only in this shape:
{"order": [1, 2, 3]}
"""
    user_prompt = (
        f"Question:\n{question}\n\n"
        "Rank all chunks from most relevant to least relevant.\n\n"
    )
    for index, chunk in enumerate(chunks):
        metadata = chunk.metadata
        user_prompt += (
            f"# CHUNK ID: {index + 1}\n"
            f"Date: {metadata.get('date')}\n"
            f"Participants: {metadata.get('participants')}\n"
            f"Start: {metadata.get('start_time')}\n"
            f"End: {metadata.get('end_time')}\n\n"
            f"{chunk.page_content}\n\n"
        )

    response = chat_client.chat.completions.create(
        model=query_model(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    reply = response.choices[0].message.content
    order = parse_rank_order(reply).order
    return [chunks[i - 1] for i in order if 1 <= i <= len(chunks)]


def parse_rank_order(reply: str) -> RankOrder:
    """Parse reranker JSON, tolerating fenced JSON replies."""
    content = reply.strip()
    if content.startswith("```"):
        content = content.removeprefix("```json").removeprefix("```").strip()
        content = content.removesuffix("```").strip()
    return RankOrder.model_validate(json.loads(content))


def make_rag_messages(question: str, history: list[dict], chunks: list[Result]) -> list[dict]:
    context = "\n\n".join(format_context_chunk(chunk) for chunk in chunks)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )


def format_context_chunk(chunk: Result) -> str:
    metadata = chunk.metadata
    return (
        f"Source: {metadata.get('source')}\n"
        f"Participants: {metadata.get('participants')}\n"
        f"Date: {metadata.get('date')}\n"
        f"Start: {metadata.get('start_time')}\n"
        f"End: {metadata.get('end_time')}\n"
        f"{chunk.page_content}"
    )


@retry(wait=wait)
def rewrite_query(question: str, history: list[dict] | None = None) -> str:
    """Rewrite the user question into a compact WhatsApp search query."""
    history = history or []
    message = f"""
Rewrite the user's question as a short semantic search query for WhatsApp chat retrieval.

Preserve names, nicknames, dates, relative time references, places, plans, events, and keywords. If the conversation history clarifies the current question, include only the useful details. Do not answer the question.

Conversation with the assistant so far:
{history}

Current question:
{question}

Return only the search query.
"""
    response = chat_client.chat.completions.create(
        model=query_model(),
        messages=[{"role": "system", "content": message}],
    )
    return response.choices[0].message.content.strip()


def merge_chunks(chunks: list[Result], reranked: list[Result]) -> list[Result]:
    merged = chunks[:]
    existing = [chunk.page_content for chunk in chunks]
    for chunk in reranked:
        if chunk.page_content not in existing:
            merged.append(chunk)
    return merged


def fetch_context_unranked(question: str) -> list[Result]:
    if collection.count() == 0:
        raise RuntimeError(
            "Vector database is empty. Run `uv run python -m whatsapp_rag.ingest` "
            "before asking questions."
        )

    query = openai.embeddings.create(model=embedding_model, input=[question]).data[0].embedding
    results = collection.query(query_embeddings=[query], n_results=RETRIEVAL_K)
    chunks = []
    for result in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(Result(page_content=result[0], metadata=result[1]))
    return chunks


def fetch_context(original_question: str, history: list[dict] | None = None) -> list[Result]:
    rewritten_question = rewrite_query(original_question, history)
    chunks1 = fetch_context_unranked(original_question)
    chunks2 = fetch_context_unranked(rewritten_question)
    chunks = merge_chunks(chunks1, chunks2)
    reranked = rerank(original_question, chunks)
    return reranked[:FINAL_K]


@retry(wait=wait)
def answer_question(question: str, history: list[dict] | None = None) -> tuple[str, list[Result]]:
    """Answer a question using the indexed WhatsApp chat context."""
    history = history or []
    chunks = fetch_context(question, history)
    messages = make_rag_messages(question, history, chunks)
    response = chat_client.chat.completions.create(
        model=answer_model(),
        messages=messages,
    )
    return response.choices[0].message.content, chunks
