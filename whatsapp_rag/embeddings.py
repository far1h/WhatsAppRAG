from litellm import embedding

from whatsapp_rag.model_config import (
    embedding_model,
)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with Gemini embeddings."""
    if not texts:
        return []

    response = embedding(
        model=embedding_model(),
        input=texts,
    )
    return [item["embedding"] for item in response["data"]]


def embed_query(text: str) -> list[float]:
    """Embed a single retrieval query with Gemini embeddings."""
    return embed_texts([text])[0]
