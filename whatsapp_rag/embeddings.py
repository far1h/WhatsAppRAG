from litellm import embedding

from whatsapp_rag.model_config import (
    embedding_model,
)


EMBEDDING_BATCH_SIZE = 100


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts with Gemini embeddings in API-sized batches."""
    if not texts:
        return []

    vectors = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        response = embedding(
            model=embedding_model(),
            input=batch,
        )
        vectors.extend(item["embedding"] for item in response["data"])
    return vectors


def embed_query(text: str) -> list[float]:
    """Embed a single retrieval query with Gemini embeddings."""
    return embed_texts([text])[0]
