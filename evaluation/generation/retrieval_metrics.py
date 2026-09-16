"""
Retrieval-quality metrics (Recall@k, MRR@k) computed against whatever
`doc_id` list the *generator actually saw* for a question -- i.e. the
retrieved (and, if enabled, reranked) list at the `top_k` used for that
generation call. Keeping k tied to the generation top_k (rather than a
separate, larger k) is deliberate: this eval is about explaining generation
quality via retrieval quality, so we want "did the doc the LLM needed
actually make it into its context" -- not retrieval quality at some other
cutoff the LLM never saw.

Both metrics assume `relevant_doc_ids` is small (AttackQA: always length 1
today, but the loop handles more without change since `QAExample` keeps it
as a list for forward compatibility -- see data_models.py).
"""
from __future__ import annotations


def recall_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: list[str]) -> float:
    """1.0 if any relevant doc_id appears anywhere in `retrieved_doc_ids`, else 0.0.

    `retrieved_doc_ids` is expected to already be truncated to k by the
    caller (i.e. pass the list the generator was actually given).
    """
    if not relevant_doc_ids:
        raise ValueError("relevant_doc_ids must be non-empty")
    relevant = set(relevant_doc_ids)
    return 1.0 if any(doc_id in relevant for doc_id in retrieved_doc_ids) else 0.0


def mrr_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: list[str]) -> float:
    """1/rank of the first relevant doc_id in `retrieved_doc_ids` (1-indexed), else 0.0."""
    if not relevant_doc_ids:
        raise ValueError("relevant_doc_ids must be non-empty")
    relevant = set(relevant_doc_ids)
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0
