"""
Evaluation runner for ColBERTv2 Late-Interaction Retriever.

Usage:
    python -m src.run_colbert_eval --split dev --limit 200
    python -m src.run_colbert_eval --split test --limit 200
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Literal

from src.data_models.io import load_corpus_lookup, load_qa_examples
from src.factory import get_retriever
from src.eval.retrieval_eval import print_report


def evaluate_colbert_with_progress(
    retriever,
    qa_examples: list,
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict:
    max_k = max(k_values)
    per_source_hits = defaultdict(lambda: defaultdict(int))
    per_source_total = defaultdict(int)
    per_source_rr = defaultdict(list)
    overall_hits = defaultdict(int)
    overall_rr = []
    latencies = []

    total_n = len(qa_examples)
    start_all = time.time()

    for idx, qa in enumerate(qa_examples, start=1):
        t0 = time.time()
        retrieved = retriever.search(qa.question, top_k=max_k)
        latency = (time.time() - t0) * 1000  # ms
        latencies.append(latency)

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

        if idx % 20 == 0 or idx == total_n:
            current_mrr = sum(overall_rr) / len(overall_rr)
            current_r1 = overall_hits[1] / idx
            avg_ms = sum(latencies) / len(latencies)
            print(
                f"[{idx}/{total_n}] Running ColBERT Evaluation -> Current MRR: {current_mrr:.4f} | Recall@1: {current_r1:.4f} | Avg Latency: {avg_ms:.1f}ms",
                flush=True,
            )

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    report = {
        "n": total_n,
        "overall": {
            "mrr": _mean(overall_rr),
            **{f"recall@{k}": overall_hits[k] / total_n for k in k_values},
            "avg_latency_ms": _mean(latencies),
            "total_eval_time_sec": time.time() - start_all,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ColBERT Late-Interaction Retriever")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, default=200, help="Number of QA examples to evaluate (default: 200)")
    parser.add_argument("--candidate-k", type=int, default=25, help="Number of candidates for Stage 1")
    parser.add_argument("--export", action="store_true", default=True)
    args = parser.parse_args()

    corpus_lookup = load_corpus_lookup()
    qa_examples = load_qa_examples(split=args.split)

    if args.limit is not None:
        qa_examples = qa_examples[: args.limit]

    print(f"Loaded {len(corpus_lookup)} documents, {len(qa_examples)} QA examples ({args.split})")
    print(f"Retriever: ColBERTv2 (Late-Interaction) | Candidate K: {args.candidate_k}", flush=True)

    retriever = get_retriever("colbert")
    retriever.candidate_k = args.candidate_k

    print(f"Evaluating ColBERT Late-Interaction Retriever on '{args.split}' split (n={len(qa_examples)})...", flush=True)
    report = evaluate_colbert_with_progress(retriever, qa_examples, k_values=(1, 5, 10))

    print("\n" + "=" * 50)
    print("COLBERT EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Samples: {report['n']}")
    print(f"MRR@10:    {report['overall']['mrr']:.4f}")
    print(f"Recall@1:  {report['overall']['recall@1']:.4f}")
    print(f"Recall@5:  {report['overall']['recall@5']:.4f}")
    print(f"Recall@10: {report['overall']['recall@10']:.4f}")
    print(f"Avg Latency: {report['overall']['avg_latency_ms']:.2f} ms")
    print("=" * 50)

    if args.export:
        out_dir = Path("analysis/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"colbert_{args.split}.json"
        with open(out_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved evaluation metrics to {out_file}", flush=True)


if __name__ == "__main__":
    main()
