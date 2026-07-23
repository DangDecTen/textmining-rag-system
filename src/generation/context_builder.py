"""
Composed internally by a Generator (not a separate pipeline stage in the
Generator ABC -- see project discussion). Takes retrieval results and builds
the "Context:" block that goes into the prompt: orders by score, dedupes by
doc_id (defensive -- matters once BM25 + dense results might get merged
upstream), and truncates to a token budget so we don't silently overflow a
small local model's context window.

Needs the SAME tokenizer the Generator will use, so token counts are accurate
for that specific model, not an approximation.
"""
from __future__ import annotations
from src.data_models.data_models import RetrievalResult


class ContextBuilder:
    def __init__(self, tokenizer, max_context_tokens: int = 1500):
        self.tokenizer = tokenizer
        self.max_context_tokens = max_context_tokens

    def build(self, contexts: list[RetrievalResult]) -> str:
        """Returns a formatted context block."""
        blocks = []
        seen_doc_ids = set()
        token_count = 0

        for r in sorted(contexts, key=lambda x: -x.score):
            if r.document is None or r.doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(r.doc_id)

            block = f"[{len(blocks) + 1}] {r.document.text}"
            n_tokens = len(self.tokenizer.encode(block))

            if blocks and token_count + n_tokens > self.max_context_tokens:
                break
            blocks.append(block)
            token_count += n_tokens

        return "\n\n".join(blocks)
