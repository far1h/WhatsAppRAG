import os


DASHSCOPE_COMPATIBLE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def normalize_model_name(model: str) -> str:
    """Return a model name accepted by the OpenAI-compatible DashScope API."""
    return model.removeprefix("dashscope/")


def summary_model() -> str:
    """Return the model used for ingest-time chunk summaries."""
    return normalize_model_name(os.getenv("SUMMARY_MODEL", "qwen3.6-flash"))


def query_model() -> str:
    """Return the model used for query rewriting and lightweight reranking."""
    return normalize_model_name(os.getenv("QUERY_MODEL", summary_model()))


def answer_model() -> str:
    """Return the higher-quality model used for final answers."""
    return normalize_model_name(
        os.getenv("ANSWER_MODEL") or os.getenv("CHAT_MODEL") or "qwen3.7-plus"
    )


def chat_api_base_url() -> str:
    """Return the OpenAI-compatible chat endpoint base URL."""
    return (
        os.getenv("CHAT_API_BASE")
        or os.getenv("DASHSCOPE_API_BASE")
        or DASHSCOPE_COMPATIBLE_BASE_URL
    )


def chat_api_key() -> str | None:
    """Return the API key for the OpenAI-compatible chat client."""
    return os.getenv("CHAT_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
