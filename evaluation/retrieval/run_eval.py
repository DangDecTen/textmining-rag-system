"""
Run a registered retriever over a QA split and save results as JSONL.

Usage:
    python -m evaluation.retrieval.run_eval --retriever bm25 --split dev
    python -m evaluation.retrieval.run_eval --retriever dense --split test --k 1 5 10 20
    python -m evaluation.retrieval.run_eval --list

The retriever is built through `src.factory.get_retriever`, so any
retriever registered via `@register_retriever(...)` (see
src/retrieval/README.md) works here with no changes to this file -- just
pass its name.

Writes one file under --output-dir (default: evaluation/retrieval/results/),
named `{retriever}_{split}_predictions.jsonl`:

    predictions.jsonl   One row per QA example: question, source, which docs
                         were retrieved and at what rank/score, hit@k for
                         each requested k, and metadata (length, subject_type,
                         field) about the relevant document(s). This is the
                         file to filter and re-aggregate later -- by source,
                         by a document-length bucket, by subject_type,
                         whatever. See the package README for examples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.data_models.io import load_corpus_lookup, load_qa_examples
from src.factory import available_retrievers, get_retriever

from evaluation.retrieval.metrics import evaluate_example


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def run(retriever_name: str, split: str, k_values: tuple[int, ...], output_dir: str) -> None:
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
    print(f"Retrieving with '{retriever_name}'...")

    predictions = []
    for qa in qa_examples:
        retrieved = retriever.search(qa.question, top_k=max_k)
        record = evaluate_example(qa, retrieved, corpus_lookup, k_values=k_values)
        record["retriever"] = retriever_name
        record["split"] = split
        predictions.append(record)

    output_dir_p = Path(output_dir)
    predictions_path = output_dir_p / f"{retriever_name}_{split}_predictions.jsonl"
    _write_jsonl(predictions, predictions_path)

    print(f"Wrote {len(predictions)} predictions -> {predictions_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retriever", help="Retriever name, e.g. 'bm25' or 'dense' (see --list)")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10], help="k values for recall@k / hit@k")
    parser.add_argument("--output-dir", default="evaluation/retrieval/results")
    parser.add_argument("--list", action="store_true", help="List available retrievers and exit")
    args = parser.parse_args()

    if args.list:
        print(f"Available retrievers: {available_retrievers()}")
        return

    if not args.retriever:
        parser.error("--retriever is required (or pass --list to see available options)")

    run(args.retriever, args.split, tuple(sorted(set(args.k))), args.output_dir)


if __name__ == "__main__":
    main()
