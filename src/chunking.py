"""Token-aware splits of build_retrieval_text() output.

bge silently drops tokens past 512. Chunking recovers the description tail without
exploding the index:

- Chunk 0: Title + Error Log + Description head (highest-signal fields together).
- Later chunks: Title + Description only — repeating a long Error Log on every
  window collapsed the body budget and produced ~28 vectors/bug.
- Hard cap MAX_CHUNKS_PER_BUG; the last chunk is the description *tail* so the end
  of a long report is never dropped.
"""

from __future__ import annotations

from typing import Protocol

from src.config import CHUNK_OVERLAP, CHUNK_TOKENS, MAX_CHUNKS_PER_BUG

_ERROR_SEP = ". Error Log: "
_DESC_SEP = ". Description: "


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...

    def decode(self, token_ids: list[int], skip_special_tokens: bool = ...) -> str: ...


def chunk_id(bug_id: str, chunk_index: int) -> str:
    return f"{bug_id}::{chunk_index}"


def parent_id_from_chunk_id(chunk_id_str: str) -> str:
    if "::" in chunk_id_str:
        return chunk_id_str.split("::", 1)[0]
    return chunk_id_str


def _parts(text: str) -> tuple[str, str, str]:
    """Return (title, error_log, description) from a composed retrieval string."""
    if not text.startswith("Title: "):
        return "", "", text

    rest = text[len("Title: ") :]
    title = ""
    error_log = ""
    description = ""

    if _ERROR_SEP in rest:
        title, rest = rest.split(_ERROR_SEP, 1)
        if _DESC_SEP in rest:
            error_log, description = rest.split(_DESC_SEP, 1)
        else:
            error_log = rest
    elif _DESC_SEP in rest:
        title, description = rest.split(_DESC_SEP, 1)
    else:
        title = rest

    return title.strip(), error_log.strip(), description.strip()


def _with_error(title: str, error_log: str, description: str) -> str:
    return (
        f"Title: {title}. Error Log: {error_log}. Description: {description}"
    )


def _title_only(title: str, description: str) -> str:
    return f"Title: {title}. Description: {description}"


def chunk_retrieval_text(
    text: str,
    tokenizer: Tokenizer,
    chunk_tokens: int = CHUNK_TOKENS,
    overlap: int = CHUNK_OVERLAP,
    max_chunks: int = MAX_CHUNKS_PER_BUG,
) -> list[str]:
    """Split a composed retrieval string into a small, bounded set of chunks."""
    text = text.strip()
    if not text:
        return []

    full_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(full_ids) <= chunk_tokens:
        return [text]

    title, error_log, description = _parts(text)
    if not description:
        truncated = tokenizer.decode(full_ids[:chunk_tokens], skip_special_tokens=True)
        return [truncated]

    max_chunks = max(1, max_chunks)
    body_ids = tokenizer.encode(description, add_special_tokens=False)

    head_prefix = _with_error(title, error_log, "")
    later_prefix = _title_only(title, "")
    head_budget = max(1, chunk_tokens - len(tokenizer.encode(head_prefix, add_special_tokens=False)))
    later_budget = max(
        1, chunk_tokens - len(tokenizer.encode(later_prefix, add_special_tokens=False))
    )

    first_end = min(head_budget, len(body_ids))
    chunks = [
        _with_error(
            title,
            error_log,
            tokenizer.decode(body_ids[:first_end], skip_special_tokens=True),
        )
    ]
    if first_end >= len(body_ids) or max_chunks == 1:
        return chunks

    remaining = body_ids[max(0, first_end - overlap) :]
    slots_left = max_chunks - 1

    # Always reserve the last slot for the true description tail.
    if slots_left == 1 or len(remaining) <= later_budget:
        tail = body_ids[-later_budget:]
        chunks.append(
            _title_only(title, tokenizer.decode(tail, skip_special_tokens=True))
        )
        return chunks

    middle_slots = slots_left - 1
    step = max(1, later_budget - overlap)
    start = 0
    for _ in range(middle_slots):
        if start >= len(remaining) - later_budget:
            break
        end = start + later_budget
        chunks.append(
            _title_only(
                title,
                tokenizer.decode(remaining[start:end], skip_special_tokens=True),
            )
        )
        start += step

    tail = body_ids[-later_budget:]
    tail_text = _title_only(title, tokenizer.decode(tail, skip_special_tokens=True))
    if chunks[-1] != tail_text:
        chunks.append(tail_text)

    return chunks[:max_chunks]
