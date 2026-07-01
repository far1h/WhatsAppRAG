import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


MESSAGE_RE = re.compile(
    r"^[\ufeff\u200e\u200f]*"
    r"\[(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s*"
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?)\]\s*"
    r"(?P<sender>[^:]+):\s*"
    r"(?P<content>.*)$"
)


@dataclass(frozen=True)
class WhatsAppMessage:
    timestamp: datetime
    sender: str
    content: str

    @property
    def date(self):
        return self.timestamp.date()


@dataclass(frozen=True)
class ConversationChunk:
    id: str
    participants: list[str]
    start_time: datetime
    end_time: datetime
    conversation_text: str
    message_count: int


def parse_whatsapp_export(path: Path) -> list[WhatsAppMessage]:
    """Parse a WhatsApp text export from disk."""
    return parse_whatsapp_export_text(path.read_text(encoding="utf-8"))


def parse_whatsapp_export_text(text: str) -> list[WhatsAppMessage]:
    """Parse WhatsApp export text into timestamped messages."""
    messages: list[WhatsAppMessage] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = MESSAGE_RE.match(line)
        if match:
            timestamp = parse_timestamp(match.group("date"), match.group("time"))
            messages.append(
                WhatsAppMessage(
                    timestamp=timestamp,
                    sender=match.group("sender").strip(),
                    content=match.group("content").strip(),
                )
            )
        elif messages:
            previous = messages[-1]
            messages[-1] = WhatsAppMessage(
                timestamp=previous.timestamp,
                sender=previous.sender,
                content=f"{previous.content}\n{line}",
            )

    return messages


def parse_timestamp(date_text: str, time_text: str) -> datetime:
    """Parse common WhatsApp export date formats."""
    if time_text.count(":") == 1:
        time_text = f"{time_text}:00"

    year_formats = ["%y", "%Y"]
    date_orders = ["%m/%d", "%d/%m"]
    attempted = []

    for date_order in date_orders:
        for year_format in year_formats:
            date_format = f"{date_order}/{year_format} %H:%M:%S"
            attempted.append(date_format)
            try:
                return datetime.strptime(f"{date_text} {time_text}", date_format)
            except ValueError:
                continue

    raise ValueError(
        f"Unsupported WhatsApp timestamp '{date_text} {time_text}'. "
        f"Tried: {', '.join(attempted)}"
    )


def create_conversation_chunks(
    messages: list[WhatsAppMessage], gap_threshold_minutes: int = 30
) -> list[ConversationChunk]:
    """Group messages into conversation chunks split by idle time gaps."""
    chunks: list[ConversationChunk] = []
    current_chunk: list[WhatsAppMessage] = []
    gap_threshold = timedelta(minutes=gap_threshold_minutes)

    for message in messages:
        if not current_chunk:
            current_chunk.append(message)
            continue

        if message.timestamp - current_chunk[-1].timestamp > gap_threshold:
            chunks.append(process_chunk(current_chunk))
            current_chunk = [message]
        else:
            current_chunk.append(message)

    if current_chunk:
        chunks.append(process_chunk(current_chunk))

    return chunks


def process_chunk(messages: list[WhatsAppMessage]) -> ConversationChunk:
    """Convert a sequence of messages into one searchable conversation chunk."""
    start_time = messages[0].timestamp
    end_time = messages[-1].timestamp
    participants = unique_in_order(message.sender for message in messages)
    conversation_text = "\n".join(format_message(message) for message in messages)

    return ConversationChunk(
        id=f"chat_{start_time.isoformat()}",
        participants=participants,
        start_time=start_time,
        end_time=end_time,
        conversation_text=conversation_text,
        message_count=len(messages),
    )


def unique_in_order(values) -> list[str]:
    """Return unique values while preserving first-seen order."""
    seen = set()
    ordered = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def format_message(message: WhatsAppMessage) -> str:
    """Format one message with timestamp context for retrieval."""
    timestamp = message.timestamp.strftime("%Y-%m-%d %H:%M")
    return f"{message.sender} ({timestamp}): {message.content}"
