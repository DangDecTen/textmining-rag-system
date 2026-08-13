"""
Per-example retrieval metrics + aggregation.

Two levels of output, on purpose:

- `evaluate_example()` builds one flat, JSONL-serializable record per QA
  pair -- which docs were retrieved, at what rank/score, whether each
  configured k was a hit, and metadata about the relevant document(s)
  (length, subject_type, field). That's enough raw information to recompute
  ANY aggregate metric later, sliced by ANY field -- source, a document-
  length bucket, subject_type, whatever -- without rerunning the retriever.
  This is the record written to `predictions.jsonl`.

- `aggregate()` reduces a list of those records into overall + per-group
  summary rows (recall@k, MRR) -- the same shape as
  `src/eval/retrieval_eval.py`, just factored so it can run over an
  arbitrary (e.g. pre-filtered) list of records, not only a fresh retriever
  run. This is the "quick sanity check at eval time" written to
  `metrics.jsonl`; it is NOT a replacement for re-slicing
  `predictions.jsonl` yourself later with pandas or similar (see the
  package README for examples -- continuous fields like document length
  need bucketing, which only makes sense to do at analysis time, not baked
  into a single eval run).
"""
from __future__ import annotations

import statistics
from bisect import bisect_right
from collections import defaultdict
from typing import Any

from src.data_models.data_models import Document, QAExample, RetrievalResult


def _doc_length(doc: Document | None) -> dict[str, int | None]:
    if doc is None:
        return {"chars": None, "words": None}
    return {"chars": len(doc.text), "words": len(doc.text.split())}


def evaluate_example(
    qa: QAExample,
    retrieved: list[RetrievalResult],
    corpus_lookup: dict[str, Document],
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict[str, Any]:
    """Builds one JSONL-ready record for a single QA pair.

    `retrieved` should already be `retriever.search(qa.question, top_k=max(k_values))`
    -- this function only scores it, it doesn't call the retriever itself,
    so it can be unit tested with hand-built `RetrievalResult` lists.
    """
    relevant = set(qa.relevant_doc_ids)
    retrieved_ids = [r.doc_id for r in retrieved]

    rank_of_first_relevant = None
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            rank_of_first_relevant = rank
            break
    reciprocal_rank = 1.0 / rank_of_first_relevant if rank_of_first_relevant else 0.0

    relevant_docs_meta = [
        {
            "doc_id": doc_id,
            "subject_type": doc.subject_type if doc else None,
            "field": doc.field if doc else None,
            **_doc_length(doc),
        }
        for doc_id, doc in ((doc_id, corpus_lookup.get(doc_id)) for doc_id in qa.relevant_doc_ids)
    ]

    return {
        "qa_id": qa.qa_id,
        "question": qa.question,
        "source": qa.source,
        "human_question": qa.human_question,
        "human_answer": qa.human_answer,
        "num_relevant_docs": len(qa.relevant_doc_ids),
        "relevant_doc_ids": qa.relevant_doc_ids,
        "relevant_docs_meta": relevant_docs_meta,
        "retrieved": [
            {"rank": rank, "doc_id": r.doc_id, "score": r.score, "is_relevant": r.doc_id in relevant}
            for rank, r in enumerate(retrieved, start=1)
        ],
        "rank_of_first_relevant": rank_of_first_relevant,
        "reciprocal_rank": reciprocal_rank,
        "k_values": list(k_values),
        **{f"hit@{k}": int(any(doc_id in relevant for doc_id in retrieved_ids[:k])) for k in k_values},
    }


def aggregate(
    records: list[dict[str, Any]],
    k_values: tuple[int, ...] = (1, 5, 10),
    group_by: str | None = None,
) -> list[dict[str, Any]]:
    """Reduces per-example records (from `evaluate_example`, optionally
    pre-filtered) into JSONL-ready summary rows: one row for "overall", plus
    one row per distinct value of `records[i][group_by]` if `group_by` is
    given (e.g. `"source"`).

    For a continuous field (document length, question length, ...),
    `group_by` should point at a bucketed/categorical field -- see
    `add_quantile_buckets()` below -- not the raw numeric value, since
    grouping by raw value would produce one group per distinct number
    instead of a useful summary.

    Raises ValueError on an empty `records` list, same as the inline eval in
    src/eval/retrieval_eval.py -- an empty aggregate is a caller bug, not a
    valid all-zeros result.
    """
    if not records:
        raise ValueError("records is empty -- nothing to aggregate.")

    def _row(group_name: str, group_records: list[dict[str, Any]]) -> dict[str, Any]:
        def _mean(records: list[dict], key: str) -> float:
            return sum(r[key] for r in records) / len(records)

        row = {
            "group_by": group_by,
            "group": group_name,
            "n": len(group_records),
            "mrr": _mean(group_records, 'reciprocal_rank'),
        }
        
        for k in k_values:
            row[f"recall@{k}"] = _mean(group_records, f'hit@{k}')
        
        row['avg_retrieve_ms'] = _mean(group_records, 'retrieve_time_ms')
        row['avg_rerank_ms'] = _mean(group_records, 'rerank_time_ms')
        row['avg_ms'] = _mean(group_records, 'total_time_ms')
        return row

    rows = [_row("overall", records)]

    if group_by:
        groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for r in records:
            groups[r[group_by]].append(r)
        for group_name, group_records in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            rows.append(_row(group_name, group_records))

    return rows


def add_quantile_buckets(
    records: list[dict[str, Any]],
    field: str,
    new_field: str,
    n_bins: int = 4,
    labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Mutates `records` in place, adding `new_field` with a quantile-based
    category (e.g. "q1", "q2", "q3", "q4" for `n_bins=4`, i.e. quartiles) 
    computed from the distribution of `records[i][field]` across the WHOLE
    input list.

    This is what turns a continuous field like document/question length
    into something `aggregate(..., group_by=new_field)` can group on --
    equal-sized (or close to it, with tied values) buckets rather than one
    group per distinct length.

    `labels` defaults to `["q1", ..., f"q{n_bins}"]`, ordered from smallest
    to largest values. Pass e.g. `["short", "medium", "long", "very_long"]`
    for more readable output.

    Bucket edges are computed once, from `records` as given -- call this
    BEFORE filtering to a subset (e.g. by source) if you want buckets
    comparable across that subset's siblings; call it AFTER filtering if you
    want buckets relative to just that subset.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    labels = labels or [f"q{i + 1}" for i in range(n_bins)]
    if len(labels) != n_bins:
        raise ValueError(f"Got {len(labels)} labels for n_bins={n_bins}; need exactly {n_bins}.")

    values = sorted(r[field] for r in records)
    distinct = sorted(set(values))

    if len(distinct) < n_bins:
        # Not enough distinct values to form n_bins groups (e.g. a tiny or
        # highly-duplicated sample) -- fall back to as many buckets as there
        # are distinct values rather than raising or silently collapsing
        # everything into one bucket.
        edges = distinct[:-1]
        labels = labels[: len(distinct)]
    else:
        edges = statistics.quantiles(values, n=n_bins, method="inclusive")

    for r in records:
        idx = bisect_right(edges, r[field])
        r[new_field] = labels[idx]

    return records

