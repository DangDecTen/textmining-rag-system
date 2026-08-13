"""
Shared naming convention for this package's output files.

A "run" is one retriever, optionally followed by one reranker. Both
`run_eval.py` (writes predictions) and `group_eval.py` (reads predictions,
writes grouped metrics) need to agree on the same filename for a given
(retriever, reranker) pair -- this is the one place that mapping is
defined, so they can't drift apart.
"""
from __future__ import annotations


def run_name(retriever: str, reranker: str | None = None) -> str:
    """'bm25' alone, or 'bm25+cross_encoder' when a reranker was applied.

    Deliberately distinct from the bare retriever name whenever a reranker
    is involved, so `bm25_dev_predictions.jsonl` (retrieval only) and
    `bm25+cross_encoder_dev_predictions.jsonl` (retrieval, then reranked)
    never collide or silently overwrite one another -- they're different
    experiments, not the same one re-run.
    """
    return retriever if not reranker else f"{retriever}+{reranker}"
