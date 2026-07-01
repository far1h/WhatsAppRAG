# WhatsApp Chunking

This project uses conversation-based chunking for WhatsApp exports.

The idea comes from the WhatsApp RAG pattern in the reference article: do not paste the entire chat into a model, and do not split the export into arbitrary fixed-size text blocks. WhatsApp messages are conversational, so the useful unit is usually a short session of messages that happened close together in time.

## Why Not One Huge Prompt

Putting the whole export into one prompt does not scale:

- Long chat histories can exceed model context limits.
- Important details get buried in unrelated messages.
- The model may summarize or infer too broadly.
- Every question would require processing the whole chat again.

Instead, ingestion turns the export into searchable conversation chunks and stores those chunks in Chroma.

## Parsing

Parsing lives in `whatsapp_rag/whatsapp.py`.

Each WhatsApp message is parsed into:

- `timestamp`
- `sender`
- `content`

The parser expects lines shaped like:

```text
[4/8/25, 18:24:04] Kei: mau lari ga
```

Continuation lines are appended to the previous message. This matters because WhatsApp exports can wrap long messages across multiple lines.

## Conversation-Based Chunking

The chunker groups messages by time gaps.

Current rule:

```python
create_conversation_chunks(messages, gap_threshold_minutes=30)
```

If the next message is more than 30 minutes after the previous message, the current chunk closes and a new chunk starts. Otherwise, the message stays in the same conversation chunk.

This keeps natural conversation flow together. For example, a back-and-forth about running plans stays in one chunk, while a later topic about travel discounts becomes another chunk.

## Chunk Shape

Each chunk stores:

- `participants`
- `start_time`
- `end_time`
- `conversation_text`
- `message_count`

The text inside a chunk is formatted with sender and timestamp:

```text
Kei (2025-04-08 18:24): mau lari ga
mo (2025-04-08 18:40): gass
```

Including timestamps in the chunk text helps the model answer temporal questions like “when did this happen?” or “what did she say yesterday?”

## Summaries

In `whatsapp_rag/ingest.py`, each conversation chunk is summarized before embedding.

The summary prompt asks for:

- Main topics discussed
- Plans, meetings, trips, dates, times, places, or logistics
- Important facts, decisions, promises, questions, or conclusions
- Relationship or emotional context only when directly supported by the text
- Searchable names, keywords, and temporal details

The final embedded document is:

```text
headline

summary

original conversation text
```

This follows the reference article’s point that summaries improve retrieval because they add compact searchable context while preserving the original messages.

## Metadata

Each stored chunk includes metadata:

- `source`
- `type`
- `participants`
- `start_time`
- `end_time`
- `date`
- `message_count`

This makes the index easier to debug and leaves room for future filtered retrieval by person or date.

## Retrieval And Rerank

Answering lives in `whatsapp_rag/answer.py`.

The current flow is:

1. Rewrite the user question into a compact search query.
2. Retrieve chunks for the original question.
3. Retrieve chunks for the rewritten question.
4. Merge both candidate lists.
5. Rerank candidates with the chat model.
6. Answer using the top reranked chunks.

This is a simpler version of the reference article’s “hybrid search” idea. It does not implement a full ReAct agent yet, but it does combine semantic retrieval, query rewriting, metadata-rich context, and LLM reranking.

## Current Tradeoffs

The 30-minute threshold is simple and usually reasonable, but it is not perfect. A long slow conversation may split into several chunks, and a busy short period may merge multiple topics. That is acceptable for now because the summary and rerank steps help recover the most relevant chunks.

Future improvements could add metadata filtering by participant/date, a configurable gap threshold, or a small ReAct-style tool layer for multi-step questions.
