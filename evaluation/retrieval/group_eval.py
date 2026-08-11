"""
Filter Q&A evaluation by metadata

Usage:
    python -m evaluation.retrieval.group_eval --retriever bm25 --split dev --group_by source
    python -m evaluation.retrieval.group_eval --retriever bm25 --split dev --group_by document_len
    python -m evaluation.retrieval.group_eval --retriever bm25 --split dev --group_by question_len --bins 5

Reads `evaluation/retrieval/results/{retriever}_{split}_predictions.jsonl`
(written by `evaluation.retrieval.run_eval`) and writes one file under
--output-dir (default: evaluation/retrieval/results/), named
`{retriever}_{split}_metrics_by_{group_by}.jsonl`:

    metrics_by_{group_by}.jsonl   One row per group ("overall" + one per
                                   group value) with recall@k / MRR.

`--group_by source` / `--group_by human_question` group by a field that's
already categorical in predictions.jsonl. `--group_by question_len` /
`--group_by document_len` group by *quantile buckets* computed here from
`question_chars` / the relevant document's `chars` -- e.g. with the default
--bins 4, each example falls into "q1" (shortest quarter of
questions/documents in this split) through "q4" (longest quarter), so group
sizes come out roughly equal regardless of the underlying length
distribution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.retrieval.metrics import add_quantile_buckets, aggregate

# Fields that are already categorical in predictions.jsonl -- group directly.
DIRECT_GROUP_FIELDS = {"source", "human_question", "human_answer"}
# Fields that need quantile bucketing before they're groupable.
QUANTILE_GROUP_FIELDS = {"question_len": "question_chars", "document_len": "document_chars"}


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _load_predictions(retriever_name: str, split: str) -> list[dict[str, Any]]:
    input_path = Path("evaluation/retrieval/results") / f"{retriever_name}_{split}_predictions.jsonl"
    if not input_path.exists():
        raise SystemExit(
            f"{input_path} not found. Run `python -m evaluation.retrieval.run_eval "
            f"--retriever {retriever_name} --split {split}` first."
        )

    predictions: list[dict[str, Any]] = []
    with open(input_path) as f:
        for line in f:
            record = json.loads(line)
            record["question_chars"] = len(record.get("question", ""))
            # relevant_docs_meta can be missing, empty, or hold a doc whose
            # `chars` is None (doc_id not found in the corpus lookup at eval
            # time) -- fall back to 0 in all of those cases so bucketing
            # never crashes on a partially-missing record.
            docs_meta = record.get("relevant_docs_meta") or [{}]
            record["document_chars"] = docs_meta[0].get("chars") or 0
            predictions.append(record)
    return predictions


def run(
    group_by: str,
    retriever_name: str,
    split: str,
    output_dir: str,
    k_values: tuple[int, ...],
    n_bins: int,
) -> None:
    predictions = _load_predictions(retriever_name, split)

    if group_by in QUANTILE_GROUP_FIELDS:
        source_field = QUANTILE_GROUP_FIELDS[group_by]
        add_quantile_buckets(predictions, field=source_field, new_field=group_by, n_bins=n_bins)
        edges_preview = sorted({r[group_by]: r[source_field] for r in predictions}.items())
        print(f"Quantile buckets for '{group_by}' (from '{source_field}', {n_bins} bins): {edges_preview}")

    metrics = aggregate(predictions, k_values=k_values, group_by=group_by)
    for row in metrics:
        row["retriever"] = retriever_name
        row["split"] = split

    output_dir_p = Path(output_dir)
    metrics_path = output_dir_p / f"{retriever_name}_{split}_metrics_by_{group_by}.jsonl"
    _write_jsonl(metrics, metrics_path)

    print(f"Wrote {len(metrics)} metric rows -> {metrics_path}")

    overall = next(row for row in metrics if row["group"] == "overall")
    summary = {k: v for k, v in overall.items() if k not in ("group_by", "group", "retriever", "split")}
    print(f"Overall: {summary}")
    for row in metrics:
        if row["group"] != "overall":
            group_summary = {k: v for k, v in row.items() if k not in ("group_by", "group", "retriever", "split")}
            print(f"  {row['group']}: {group_summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--group_by",
        default="source",
        choices=["source", "human_question", "question_len", "document_len"],
    )
    parser.add_argument("--retriever", required=True, help="Retriever name used for the earlier run_eval run")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10], help="k values for recall@k / hit@k")
    parser.add_argument("--output-dir", default="evaluation/retrieval/results")
    parser.add_argument(
        "--bins",
        type=int,
        default=4,
        help="Number of quantile buckets for --group_by question_len/document_len (default: 4, i.e. quartiles)",
    )
    args = parser.parse_args()

    run(args.group_by, args.retriever, args.split, args.output_dir, tuple(sorted(set(args.k))), args.bins)


if __name__ == "__main__":
    main()
