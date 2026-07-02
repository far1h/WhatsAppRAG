import os


DEFAULT_SUMMARY_MODEL = "gemini/gemini-2.5-flash"
DEFAULT_QUERY_MODEL = "gemini/gemini-2.5-flash"
DEFAULT_ANSWER_MODEL = "gemini/gemini-3.5"
DEFAULT_EMBEDDING_MODEL = "gemini/gemini-embedding-2"


def summary_model() -> str:
    """Return the small model used for ingest-time chunk summaries."""
    return os.getenv("SUMMARY_MODEL", DEFAULT_SUMMARY_MODEL)


def query_model() -> str:
    """Return the small model used for query rewriting and reranking."""
    return os.getenv("QUERY_MODEL", DEFAULT_QUERY_MODEL)


def answer_model() -> str:
    """Return the stronger model used for final answers."""
    return os.getenv("ANSWER_MODEL") or os.getenv("CHAT_MODEL") or DEFAULT_ANSWER_MODEL


def embedding_model() -> str:
    """Return the Gemini embedding model used for indexing and retrieval."""
    return os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
