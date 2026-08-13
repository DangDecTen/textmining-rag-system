"""
Run a registered retriever -- optionally followed by a registered reranker
-- over a QA split and save results as JSONL.

Usage:
    python -m evaluation.retrieval.run_eval --retriever bm25 --split dev
    python -m evaluation.retrieval.run_eval --retriever dense --split test --k 1 5 10 20
    python -m evaluation.retrieval.run_eval --retriever bm25 --split dev --rerank
    python -m evaluation.retrieval.run_eval --retriever bm25 --split dev --rerank --reranker cross_encoder
    python -m evaluation.retrieval.run_eval --list

The retriever is built through `src.factory.get_retriever`, so any
retriever registered via `@register_retriever(...)` (see
src/retrieval/README.md) works here with no changes to this file -- just
pass its name. Same for `--reranker` and `src.factory.get_reranker` /
`@register_reranker(...)` (see src/reranking/README.md).

`--rerank` mirrors what `src.pipeline.Pipeline` actually does at query
time: retrieve a wider candidate set (`--candidate-k`, default
`settings.rerank_candidate_k`) than the requested k values, then rerank
down to `max(k_values)` before scoring. Without `--rerank`, this evaluates
the retriever alone -- exactly like before this flag existed.

Writes one file under --output-dir (default: evaluation/retrieval/results/),
named `{run_name}_{split}_predictions.jsonl`, where `run_name` is the bare
retriever name (e.g. `bm25`) or `{retriever}+{reranker}` when `--rerank` is
given (e.g. `bm25+cross_encoder`) -- see `naming.py`. This keeps a
retrieval-only run and a reranked run of the same retriever as two separate
files rather than one overwriting the other:

    predictions.jsonl   One row per QA example: question, source, which docs
                         were retrieved and at what rank/score, hit@k for
                         each requested k, and metadata (length, subject_type,
                         field) about the relevant document(s). Also carries
                         "retriever" and "reranker" (null if --rerank wasn't
                         given) so reranked and non-reranked runs stay
                         distinguishable after concatenating. This is the
                         file to filter and re-aggregate later -- by source,
                         by a document-length bucket, by subject_type,
                         whatever. See the package README for examples.
"""
from __future__ import annotations

import time
import argparse
import json
from pathlib import Path
from typing import Any

from src.config import settings
from src.data_models.io import load_corpus_lookup, load_qa_examples
from src.factory import available_rerankers, available_retrievers, get_reranker, get_retriever

from evaluation.retrieval.metrics import evaluate_example
from evaluation.retrieval.naming import run_name


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def run(
    retriever_name: str,
    split: str,
    k_values: tuple[int, ...],
    output_dir: str,
    reranker_name: str | None = None,
    candidate_k: int | None = None,
) -> None:
    corpus_lookup = load_corpus_lookup()
    qa_examples = load_qa_examples(split=split)
    print(f"Loaded {len(corpus_lookup)} documents, {len(qa_examples)} QA examples ({split})")

    try:
        retriever = get_retriever(retriever_name)
    except ValueError as e:
        raise SystemExit(f"{e}\nAvailable retrievers: {available_retrievers()}") from e
    except FileNotFoundError as e:
        raise SystemExit(
            f"Index for '{retriever_name}' not found on disk -- build it first "
            f"(see root README, 'Indexing & Retrieval'). {e}"
        ) from e

    max_k = max(k_values)

    reranker = None
    if reranker_name is not None:
        try:
            reranker = get_reranker(reranker_name)
        except ValueError as e:
            raise SystemExit(f"{e}\nAvailable rerankers: {available_rerankers()}") from e
        candidate_k = candidate_k or settings.rerank_candidate_k
        search_k = max(max_k, candidate_k)
        print(
            f"Retrieving with '{retriever_name}' (top {search_k}), "
            f"reranking with '{reranker_name}' to top {max_k}..."
        )
    else:
        search_k = max_k
        print(f"Retrieving with '{retriever_name}'...")

    predictions = []
    for qa in qa_examples:
        t0 = time.perf_counter()
        retrieved = retriever.search(qa.question, top_k=search_k)
        t2 = t1 = time.perf_counter()
        if reranker is not None:
            retrieved = reranker.rerank(qa.question, retrieved, top_k=max_k)
            t2 = time.perf_counter()

        record = evaluate_example(qa, retrieved, corpus_lookup, k_values=k_values)
        record["retrieve_time_ms"] = (t1 - t0) * 1000
        record["rerank_time_ms"] = (t2 - t1) * 1000
        record["total_time_ms"] = (t2 - t0) * 1000
        record["retriever"] = retriever_name
        record["reranker"] = reranker_name
        record["split"] = split
        predictions.append(record)

    def _mean(records: list[dict], key: str) -> float:
        return sum(r[key] for r in records) / len(records)

    overall_metrics = {}
    overall_metrics['n'] = len(predictions)
    overall_metrics['mrr'] = _mean(predictions, 'reciprocal_rank')
    for k in k_values:
        overall_metrics[f"recall@{k}"] = _mean(predictions, f'hit@{k}')
    overall_metrics['avg_retrieve_ms'] = _mean(predictions, 'retrieve_time_ms')
    overall_metrics['avg_rerank_ms'] = _mean(predictions, 'rerank_time_ms')
    overall_metrics['avg_ms'] = _mean(predictions, 'total_time_ms')

    run = run_name(retriever_name, reranker_name)
    output_dir_p = Path(output_dir)
    predictions_path = output_dir_p / f"{run}_{split}_predictions.jsonl"
    _write_jsonl(predictions, predictions_path)

    print(f"Wrote {len(predictions)} predictions -> {predictions_path}")
    print("Overall metrics:")
    print("| Retriever |", end="")
    for metric in overall_metrics.keys():
        print(f" {metric} |", end="")
    print(f"\n|{run} |", end="")
    for value in overall_metrics.values():
        if isinstance(value, int):
            print(f" {value}|", end="")
        else:
            print(f" {value:.3f}|", end="")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retriever", help="Retriever name, e.g. 'bm25' or 'dense' (see --list)")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10], help="k values for recall@k / hit@k")
    parser.add_argument("--output-dir", default="evaluation/retrieval/results")
    parser.add_argument(
        "--rerank", action="store_true", help="Rerank retrieved candidates before scoring (see --reranker)"
    )
    parser.add_argument(
        "--reranker",
        default=None,
        help="Reranker name, e.g. 'cross_encoder' (see --list). Implies --rerank. "
        "Defaults to settings.default_reranker if --rerank is given without this.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help="How many candidates to retrieve before reranking (default: settings.rerank_candidate_k). "
        "Ignored unless --rerank is given.",
    )
    parser.add_argument("--list", action="store_true", help="List available retrievers/rerankers and exit")
    args = parser.parse_args()

    if args.list:
        print(f"Available retrievers: {available_retrievers()}")
        print(f"Available rerankers: {available_rerankers()}")
        return

    if not args.retriever:
        parser.error("--retriever is required (or pass --list to see available options)")

    rerank = args.rerank or args.reranker is not None
    reranker_name = (args.reranker or settings.default_reranker) if rerank else None

    run(
        args.retriever,
        args.split,
        tuple(sorted(set(args.k))),
        args.output_dir,
        reranker_name=reranker_name,
        candidate_k=args.candidate_k,
    )


if __name__ == "__main__":
    main()
