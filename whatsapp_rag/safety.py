import re


SENSITIVE_TERM_RE = re.compile(
    r"\b("
    r"nsfw|"
    r"explicit|"
    r"fuck(?:ing|ed|er)?|"
    r"shit|"
    r"bitch(?:es)?|"
    r"asshole|"
    r"dick|"
    r"pussy|"
    r"horny|"
    r"sex(?:ual)?|"
    r"sexy|"
    r"nude(?:s)?|"
    r"porn|"
    r"cum|"
    r"boob(?:s)?"
    r")\b",
    re.IGNORECASE,
)

DATA_INSPECTION_MARKERS = (
    "datainspectionfailed",
    "data_inspection_failed",
    "inappropriate content",
)


def sanitize_for_model(text: str) -> str:
    """Return a safer cloud-facing copy of a chat chunk."""
    return SENSITIVE_TERM_RE.sub("[sensitive term]", text)


def is_data_inspection_error(error: Exception) -> bool:
    """Return whether a provider rejected content during safety inspection."""
    message = str(error).lower()
    return any(marker in message for marker in DATA_INSPECTION_MARKERS)


def should_retry_model_error(error: Exception) -> bool:
    """Return whether a model error looks transient enough to retry."""
    return not is_data_inspection_error(error)


def make_fallback_summary(metadata: dict) -> dict:
    """Create a searchable summary without sending the chunk back to a model."""
    participants = metadata.get("participants") or "unknown participants"
    date = metadata.get("date") or "unknown date"
    start_time = metadata.get("start_time") or "unknown start"
    end_time = metadata.get("end_time") or "unknown end"
    return {
        "headline": f"Conversation on {date} with {participants}",
        "summary": (
            "Cloud model summary skipped because provider content inspection "
            f"rejected this chunk. Use the sanitized conversation text and "
            f"metadata for retrieval. Time range: {start_time} to {end_time}."
        ),
    }
