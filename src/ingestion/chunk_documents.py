"""
Recursive, token-aware chunking of parsed ATT&CK docs.

Why recursive instead of pure sentence sliding-window (stage01 version):
  - Sentence splitting alone doesn't respect natural document structure
    (e.g. the enrichment block from parse_attack.py is a separate logical
    unit from the description, and shouldn't be split mid-line).
  - Recursive splitting tries the "nicest" separator first (paragraph
    breaks), and only falls through to finer-grained separators (line,
    sentence, word, then a hard token cut) for the pieces that are still
    too big. This mirrors LangChain's RecursiveCharacterTextSplitter, but
    reimplemented at the token level with no framework dependency, per
    the project's "lower-level components" constraint.

Why token-based sizing instead of word count:
  - The embedding model (BAAI/bge-small-en-v1.5) has a hard 512-token
    limit; word count is a poor proxy for token count for text full of
    ATT&CK IDs, acronyms, and punctuation-heavy security jargon, and can
    silently lead to truncated (and therefore under-represented) chunks.
  - We reuse the SAME tokenizer as the embedding model (imported from
    build_index.py) so "how many tokens will this actually cost at
    embedding time" is exact, not estimated.

Relationship enrichment handling:
  - The enrichment lines from parse_attack.py (`related_context`) are
    short (capped upstream) and repeated on EVERY chunk of a document,
    not just the first. This keeps each chunk self-contained: if a long
    technique description splits into 3 chunks, a query about mitigation
    should be able to hit any of them and still see "Mitigated by: ...".
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from src.data_models.chunk import Chunk

from transformers import AutoTokenizer

# from src.indexing.build_index import EMBED_MODEL_NAME
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

MODEL_MAX_TOKENS = 512       # bge-small-en-v1.5's hard sequence limit
SAFETY_BUFFER = 12           # room for special tokens (CLS/SEP) + slack
MAX_CHUNK_TOKENS = 200       # default/target budget for the splittable body
MIN_CHUNK_TOKENS = 50        # floor -- never shrink the body below this
OVERLAP_TOKENS = 40
MAX_RELATED_BLOCK_TOKENS = 200  # hard cap on the enrichment block itself;
                                  # a heavily-connected technique (many
                                  # relationship types, each near its own
                                  # MAX_RELATED_PER_LABEL cap) could
                                  # otherwise dominate the whole chunk
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]   # tried in order, coarse -> fine

_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)


def _token_len(text: str) -> int:
    return len(_tokenizer.encode(text, add_special_tokens=False))


def _hard_token_split(text: str, max_tokens: int) -> list[str]:
    """Last-resort fallback: slice by raw token ids. Used only for a
    single 'word' (e.g. a long URL or hash) that exceeds max_tokens even
    on its own -- rare, but silent truncation is worse than an ugly split.
    """
    token_ids = _tokenizer.encode(text, add_special_tokens=False)
    pieces = []
    for i in range(0, len(token_ids), max_tokens):
        piece_ids = token_ids[i:i + max_tokens]
        pieces.append(_tokenizer.decode(piece_ids))
    return pieces


def _recursive_split(text: str, separators: list[str], max_tokens: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if _token_len(text) <= max_tokens:
        return [text]

    if not separators:
        return _hard_token_split(text, max_tokens)

    sep, rest_separators = separators[0], separators[1:]
    parts = text.split(sep) if sep else list(text)

    # Re-merge parts greedily up to max_tokens; recurse on any part that's
    # still too big on its own (e.g. one giant paragraph with no periods).
    merged_chunks = []
    current_parts: list[str] = []
    current_len = 0

    def flush():
        if current_parts:
            merged_chunks.append(sep.join(current_parts))

    for part in parts:
        part = part.strip() if sep else part
        if not part:
            continue
        part_len = _token_len(part)

        if part_len > max_tokens:
            flush()
            current_parts, current_len = [], 0
            merged_chunks.extend(_recursive_split(part, rest_separators, max_tokens))
            continue

        added_len = part_len + (_token_len(sep) if current_parts else 0)
        if current_len + added_len > max_tokens and current_parts:
            flush()
            current_parts, current_len = [], 0

        current_parts.append(part)
        current_len += added_len

    flush()
    return merged_chunks


def _add_overlap(chunks: list[str], overlap_tokens: int) -> list[str]:
    """Prepend the tail of each chunk to the next one, so adjacent chunks
    share context and a fact split across a boundary isn't invisible to
    whichever chunk gets embedded and matched at retrieval time."""
    if overlap_tokens <= 0 or len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_ids = _tokenizer.encode(chunks[i - 1], add_special_tokens=False)
        tail_ids = prev_ids[-overlap_tokens:]
        tail_text = _tokenizer.decode(tail_ids)
        overlapped.append(f"{tail_text} {chunks[i]}")
    return overlapped


def _truncate_related_block(related_block: str) -> str:
    """Cap the enrichment block's token length by dropping trailing lines
    (each line is one relationship label, e.g. 'Used by: ...') rather than
    truncating mid-line, which would leave a dangling, misleading list."""
    if _token_len(related_block) <= MAX_RELATED_BLOCK_TOKENS:
        return related_block

    lines = related_block.split("\n")
    kept, total = [], 0
    for line in lines:
        line_len = _token_len(line)
        if total + line_len > MAX_RELATED_BLOCK_TOKENS:
            break
        kept.append(line)
        total += line_len
    return "\n".join(kept)


def _chunk_description(name: str, description: str, related_tokens: int) -> list[str]:
    body = f"{name}\n\n{description}" if description else name

    # Shrink the body budget dynamically so body + overlap + enrichment
    # never exceeds the embedding model's actual sequence limit.
    available = MODEL_MAX_TOKENS - SAFETY_BUFFER - related_tokens - OVERLAP_TOKENS
    effective_max = max(MIN_CHUNK_TOKENS, min(MAX_CHUNK_TOKENS, available))

    raw_chunks = _recursive_split(body, SEPARATORS, effective_max)
    return _add_overlap(raw_chunks, OVERLAP_TOKENS)


def chunk_docs(docs_path: str) -> list[Chunk]:
    chunks = []
    with open(docs_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)

            related_context = doc.get("related_context") or []
            related_block = _truncate_related_block("\n".join(related_context))
            related_tokens = _token_len(related_block) if related_block else 0

            body_chunks = _chunk_description(doc["name"], doc.get("description", ""), related_tokens)

            for i, body in enumerate(body_chunks):
                full_text = f"{body}\n\n{related_block}" if related_block else body
                chunks.append(Chunk(
                    chunk_id=f"{doc['attack_id']}::chunk{i}",
                    doc_id=doc["attack_id"],
                    attack_domain=doc["domain"],
                    attack_type=doc["attack_type"],
                    name=doc["name"],
                    text=full_text,
                    url=doc.get("url"),
                ))
    return chunks


def save_chunks(chunks: list[Chunk], out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c)) + "\n")


if __name__ == "__main__":
    chunks = chunk_docs("data/processed/attack_docs.jsonl")
    save_chunks(chunks, "data/processed/chunks.jsonl")

    token_counts = [_token_len(c.text) for c in chunks]
    print(f"Created {len(chunks)} chunks -> data/processed/chunks.jsonl")
    print(f"Token length: min={min(token_counts)}, max={max(token_counts)}, "
          f"avg={sum(token_counts) / len(token_counts):.1f}")