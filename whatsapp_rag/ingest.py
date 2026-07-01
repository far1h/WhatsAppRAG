import json
import os
from pathlib import Path

from chromadb import PersistentClient
from dotenv import load_dotenv
from litellm import completion
from openai import OpenAI
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential
from tqdm import tqdm

from whatsapp_rag.whatsapp import create_conversation_chunks, parse_whatsapp_export


load_dotenv(override=True)

DASHSCOPE_COMPATIBLE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
CHAT_MODEL = os.getenv("CHAT_MODEL", "dashscope/qwen-plus")
DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"
CHAT_EXPORT_PATH = KNOWLEDGE_BASE_PATH / "chat"

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


class Result(BaseModel):
    page_content: str
    metadata: dict


class ChunkSummary(BaseModel):
    headline: str = Field(
        description="A short retrieval heading for this conversation chunk."
    )
    summary: str = Field(
        description="A concise summary of the topics, people, dates, plans, and useful details."
    )


def fetch_documents() -> list[dict]:
    """Load WhatsApp exports and split them into conversation documents."""
    paths = find_chat_exports()
    documents = []

    for path in paths:
        messages = parse_whatsapp_export(path)
        for chunk in create_conversation_chunks(messages):
            documents.append(
                {
                    "type": "chat",
                    "source": path.as_posix(),
                    "text": chunk.conversation_text,
                    "metadata": {
                        "source": path.as_posix(),
                        "type": "chat",
                        "participants": ", ".join(chunk.participants),
                        "start_time": chunk.start_time.isoformat(),
                        "end_time": chunk.end_time.isoformat(),
                        "date": chunk.start_time.date().isoformat(),
                        "message_count": chunk.message_count,
                    },
                }
            )

    print(f"Loaded {len(documents)} conversation chunks from {len(paths)} chat export(s)")
    return documents


def find_chat_exports() -> list[Path]:
    """Find local WhatsApp export files for ingestion."""
    paths = []
    if CHAT_EXPORT_PATH.exists():
        paths.extend(CHAT_EXPORT_PATH.rglob("*.txt"))
        paths.extend(CHAT_EXPORT_PATH.rglob("*.md"))

    fallback = Path(__file__).parent.parent / "_chat.txt"
    if not paths and fallback.exists():
        paths.append(fallback)

    return sorted(set(paths))


def make_prompt(document: dict) -> str:
    return f"""
Summarize this WhatsApp conversation chunk for a retrieval system.

Focus on:
- Main topics discussed
- Plans, meetings, trips, dates, times, places, or logistics
- Important facts, decisions, promises, questions, or conclusions
- Relationship or emotional context only when it is directly supported by the text
- Searchable names, keywords, and temporal details

Rules:
- Do not invent facts.
- Preserve uncertainty when the text is unclear.
- Keep the summary compact and useful for semantic search.
- Return valid JSON only with these exact keys: "headline", "summary".

Conversation metadata:
- Source: {document["source"]}
- Participants: {document["metadata"]["participants"]}
- Date: {document["metadata"]["date"]}
- Start: {document["metadata"]["start_time"]}
- End: {document["metadata"]["end_time"]}
- Message count: {document["metadata"]["message_count"]}

Conversation:
{document["text"]}
"""


def make_messages(document: dict) -> list[dict]:
    return [{"role": "user", "content": make_prompt(document)}]


@retry(wait=wait)
def process_document(document: dict) -> Result:
    response = completion(model=CHAT_MODEL, messages=make_messages(document))
    reply = response.choices[0].message.content
    summary = parse_chunk_summary(reply)
    text = f"{summary.headline}\n\n{summary.summary}\n\n{document['text']}"
    return Result(page_content=text, metadata=document["metadata"])


def parse_chunk_summary(reply: str) -> ChunkSummary:
    """Parse a model JSON summary, tolerating fenced JSON replies."""
    content = reply.strip()
    if content.startswith("```"):
        content = content.removeprefix("```json").removeprefix("```").strip()
        content = content.removesuffix("```").strip()

    data = json.loads(content)
    return ChunkSummary.model_validate(data)


def create_chunks(documents: list[dict]) -> list[Result]:
    """Create summarized searchable chunks from parsed conversations."""
    chunks = []
    for document in tqdm(documents):
        chunks.append(process_document(document))
    return chunks


def create_embeddings(chunks: list[Result]) -> None:
    openai = make_embedding_client()
    chroma = PersistentClient(path=DB_NAME)
    if collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)

    texts = [chunk.page_content for chunk in chunks]
    emb = openai.embeddings.create(model=embedding_model, input=texts).data
    vectors = [e.embedding for e in emb]

    collection = chroma.get_or_create_collection(collection_name)

    ids = [str(i) for i in range(len(chunks))]
    metas = [chunk.metadata for chunk in chunks]

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    print(f"Vectorstore created with {collection.count()} documents")


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


if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")
