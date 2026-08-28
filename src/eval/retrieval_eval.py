"""
Retrieval evaluation for AttackQA: since every QAExample has known ground-
truth doc_id(s), this is standard single-relevant-passage retrieval eval
(structurally like MS MARCO passage ranking).

Metrics:
- Recall@k: did a relevant doc appear anywhere in the top-k?
- MRR: mean reciprocal rank of the first relevant doc (rewards ranking it
  higher, not just present).

Both are reported overall AND per `source`, since source categories vary
a lot in size (see ingestion stage notes) -- a good aggregate score can hide
a category that retrieves badly.
"""
from __future__ import annotations

from collections import defaultdict

from src.data_models.data_models import QAExample
from src.retrieval.base import Retriever


def evaluate_retriever(
    retriever: Retriever,
    qa_examples: list[QAExample],
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict:
    max_k = max(k_values)

    per_source_hits = defaultdict(lambda: defaultdict(int))
    per_source_total = defaultdict(int)
    per_source_rr = defaultdict(list)
    overall_hits = defaultdict(int)
    overall_rr = []

    for qa in qa_examples:
        retrieved = retriever.search(qa.question, top_k=max_k)
        retrieved_ids = [r.doc_id for r in retrieved]
        relevant = set(qa.relevant_doc_ids)

        rr = 0.0
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant:
                rr = 1.0 / rank
                break
        overall_rr.append(rr)
        per_source_rr[qa.source].append(rr)
        per_source_total[qa.source] += 1

        for k in k_values:
            hit = int(any(doc_id in relevant for doc_id in retrieved_ids[:k]))
            overall_hits[k] += hit
            per_source_hits[qa.source][k] += hit

    n = len(qa_examples)
    if n == 0:
        raise ValueError("qa_examples is empty -- nothing to evaluate.")

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    report = {
        "n": n,
        "overall": {
            "mrr": _mean(overall_rr),
            **{f"recall@{k}": overall_hits[k] / n for k in k_values},
        },
        "by_source": {
            src: {
                "n": total,
                "mrr": _mean(per_source_rr[src]),
                **{f"recall@{k}": per_source_hits[src][k] / total for k in k_values},
            }
            for src, total in per_source_total.items()
        },
    }
    return report


def print_report(report: dict) -> None:
    print(f"n={report['n']}")
    print("Overall:")
    for metric, value in report["overall"].items():
        print(f"  {metric}: {value:.3f}")
    print("By source:")
    for src, metrics in sorted(report["by_source"].items(), key=lambda kv: -kv[1]["n"]):
        line = ", ".join(f"{m}={v:.3f}" for m, v in metrics.items() if m != "n")
        print(f"  {src} (n={metrics['n']}): {line}")
