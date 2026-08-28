"""
Evaluation runner for Cross-Encoder Re-ranking Retriever.

Usage:
    python -m src.run_rerank_eval --split dev
    python -m src.run_rerank_eval --split test
    python -m src.run_rerank_eval --split dev --limit 200 --candidate-k 25
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Literal

from src.data_models.io import load_corpus_lookup, load_qa_examples
from src.retrieval.retriever_factory import RetrieverFactory
from src.retrieval.cross_encoder_retriever import CrossEncoderRetriever
from src.eval.retrieval_eval import print_report


def evaluate_reranker_with_progress(
    retriever: CrossEncoderRetriever,
    qa_examples: list,
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict:
    max_k = max(k_values)
    per_source_hits = defaultdict(lambda: defaultdict(int))
    per_source_total = defaultdict(int)
    per_source_rr = defaultdict(list)
    overall_hits = defaultdict(int)
    overall_rr = []

    total_n = len(qa_examples)
    for idx, qa in enumerate(qa_examples, start=1):
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

        if idx % 20 == 0 or idx == total_n:
            current_mrr = sum(overall_rr) / len(overall_rr)
            current_r1 = overall_hits[1] / idx
            print(f"[{idx}/{total_n}] Running Evaluation -> Current MRR: {current_mrr:.3f} | Recall@1: {current_r1:.3f}", flush=True)

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    report = {
        "n": total_n,
        "overall": {
            "mrr": _mean(overall_rr),
            **{f"recall@{k}": overall_hits[k] / total_n for k in k_values},
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


def main(
    split: Literal["train", "dev", "test"],
    base_type: str = "hybrid",
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    candidate_k: int = 25,
    limit: int | None = None,
    export: bool = True,
) -> None:
    corpus_lookup = load_corpus_lookup()
    qa_examples = load_qa_examples(split=split)

    if limit is not None:
        qa_examples = qa_examples[:limit]

    print(f"Loaded {len(corpus_lookup)} documents, {len(qa_examples)} QA examples ({split})")
    print(f"Base Retriever: {base_type} | Cross-Encoder Model: {model_name} | Candidate K: {candidate_k}", flush=True)
    print()

    base_retriever = RetrieverFactory.create(base_type, corpus_lookup=corpus_lookup)

    reranker = CrossEncoderRetriever(
        base_retriever=base_retriever,
        model_name=model_name,
        candidate_k=candidate_k,
        corpus_lookup=corpus_lookup,
    )

    print(f"Evaluating Cross-Encoder Re-ranker on '{split}' split...", flush=True)
    report = evaluate_reranker_with_progress(reranker, qa_examples, k_values=(1, 5, 10))
    print_report(report)

    if export:
        out_dir = Path("analysis/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"rerank_{split}.json"
        with open(out_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved evaluation metrics to {out_file}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--base", default="hybrid", choices=["hybrid", "dense", "bm25"])
    parser.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--candidate-k", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None, help="Limit N QA examples for fast evaluation")
    args = parser.parse_args()

    main(
        split=args.split,
        base_type=args.base,
        model_name=args.model,
        candidate_k=args.candidate_k,
        limit=args.limit,
    )
