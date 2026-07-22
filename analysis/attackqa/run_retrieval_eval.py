"""Evaluate BM25 and dense retrievers on AttackQA QA splits."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from analysis.attackqa.common import attach_documents, doc_length_bucket, length_quantiles, load_jsonl, question_category, top_counts
from src.data_models.data_models import Document, QAExample
from src.indexing.bm25_index import BM25Index
from src.indexing.dense_index import DenseIndex
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
INDEX_DIR = REPO_ROOT / "data" / "index"
RESULTS_DIR = ROOT / "results"


@dataclass
class RetrievalSummary:
    retriever: str
    split: str
    num_queries: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float
    mean_rank: float
    median_rank: float
    hit_rate_at_1: float
    hit_rate_at_5: float
    hit_rate_at_10: float


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_corpus_lookup() -> dict[str, Document]:
    corpus_rows = load_jsonl(PROCESSED_DIR / "corpus.jsonl")
    return {row["doc_id"]: Document.model_validate(row) for row in corpus_rows}


def load_qa_split(split: str) -> list[QAExample]:
    return [QAExample.model_validate(row) for row in load_jsonl(PROCESSED_DIR / f"qa_{split}.jsonl")]


def load_retriever(name: str, corpus_lookup: dict[str, Document]):
    if name == "dense":
        index = DenseIndex.load(str(INDEX_DIR / "dense"))
        return DenseRetriever(index=index, corpus_lookup=corpus_lookup)
    if name == "bm25":
        index = BM25Index.load(str(INDEX_DIR / "bm25"))
        return BM25Retriever(index=index, corpus_lookup=corpus_lookup)
    raise ValueError(f"Unknown retriever: {name}")


def first_relevant_rank(retrieved_doc_ids: list[str], gold_doc_ids: list[str]) -> int | None:
    gold = set(gold_doc_ids)
    for idx, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in gold:
            return idx
    return None


def evaluate_split(split: str, retriever_name: str) -> tuple[RetrievalSummary, list[dict]]:
    corpus_lookup = load_corpus_lookup()
    qa_rows = load_qa_split(split)
    retriever = load_retriever(retriever_name, corpus_lookup)
    doc_lengths = [len(doc.text) for doc in corpus_lookup.values()]
    short_cutoff, long_cutoff = length_quantiles(doc_lengths)

    max_k = 10
    rows = []
    ranks = []
    hits_at_1 = hits_at_5 = hits_at_10 = 0
    mrr_terms = []

    for qa in qa_rows:
        retrieved = retriever.search(qa.question, top_k=max_k)
        retrieved_doc_ids = [result.doc_id for result in retrieved]
        rank = first_relevant_rank(retrieved_doc_ids, qa.relevant_doc_ids)
        ranks.append(rank if rank is not None else float("inf"))
        if rank is not None:
            if rank <= 1:
                hits_at_1 += 1
            if rank <= 5:
                hits_at_5 += 1
            if rank <= 10:
                hits_at_10 += 1
            mrr_terms.append(1.0 / rank)
        else:
            mrr_terms.append(0.0)

        doc_lookup = corpus_lookup[qa.relevant_doc_ids[0]]
        rows.append({
            "qa_id": qa.qa_id,
            "split": split,
            "retriever": retriever_name,
            "question": qa.question,
            "source": qa.source,
            "human_question": qa.human_question,
            "human_answer": qa.human_answer,
            "question_category": question_category(qa.question),
            "doc_length": len(doc_lookup.text),
            "doc_length_bucket": doc_length_bucket(len(doc_lookup.text), short_cutoff, long_cutoff),
            "gold_doc_id": qa.relevant_doc_ids[0],
            "gold_doc_subject_id": doc_lookup.subject_id,
            "gold_doc_subject_name": doc_lookup.subject_name,
            "gold_doc_subject_type": doc_lookup.subject_type,
            "rank": rank,
            "top1_doc_id": retrieved_doc_ids[0] if retrieved_doc_ids else None,
            "top1_score": retrieved[0].score if retrieved else None,
            "top3_doc_ids": json.dumps(retrieved_doc_ids[:3]),
            "top10_doc_ids": json.dumps(retrieved_doc_ids),
        })

    summary = RetrievalSummary(
        retriever=retriever_name,
        split=split,
        num_queries=len(qa_rows),
        recall_at_1=hits_at_1 / len(qa_rows),
        recall_at_5=hits_at_5 / len(qa_rows),
        recall_at_10=hits_at_10 / len(qa_rows),
        mrr_at_10=sum(mrr_terms) / len(mrr_terms),
        mean_rank=float(sum(r if r != float("inf") else 11 for r in ranks) / len(ranks)),
        median_rank=float(sorted(r if r != float("inf") else 11 for r in ranks)[len(ranks) // 2]),
        hit_rate_at_1=hits_at_1 / len(qa_rows),
        hit_rate_at_5=hits_at_5 / len(qa_rows),
        hit_rate_at_10=hits_at_10 / len(qa_rows),
    )
    return summary, rows


def slice_metrics(rows: list[dict], key: str) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)

    output = []
    for group_name, group_rows in sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True):
        total = len(group_rows)
        for cutoff in (1, 5, 10):
            pass
        ranks = [row["rank"] for row in group_rows]
        valid = [rank for rank in ranks if rank is not None]
        output.append({
            key: group_name,
            "count": total,
            "recall@1": sum(1 for rank in valid if rank <= 1) / total,
            "recall@5": sum(1 for rank in valid if rank <= 5) / total,
            "recall@10": sum(1 for rank in valid if rank <= 10) / total,
            "mrr@10": sum(0.0 if rank is None or rank > 10 else 1.0 / rank for rank in ranks) / total,
        })
    return output


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_bar_comparison(metrics: dict[tuple[str, str], RetrievalSummary], split: str) -> None:
    names = ["bm25", "dense"]
    values = {
        "recall@1": [metrics[(split, name)].recall_at_1 for name in names],
        "recall@5": [metrics[(split, name)].recall_at_5 for name in names],
        "recall@10": [metrics[(split, name)].recall_at_10 for name in names],
        "mrr@10": [metrics[(split, name)].mrr_at_10 for name in names],
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, metric in zip(axes, ("recall@10", "mrr@10")):
        axis.bar(names, values[metric], color=["#1f77b4", "#2ca02c"])
        axis.set_title(f"{split.upper()} {metric}")
        axis.set_ylim(0, 1)
        for idx, value in enumerate(values[metric]):
            axis.text(idx, value, f"{value:.3f}", ha="center", va="bottom")
    fig.suptitle(f"AttackQA Retriever Comparison on {split.upper()}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(RESULTS_DIR / f"comparison_{split}.png", dpi=180)
    plt.close(fig)


def write_report(summaries: list[RetrievalSummary], slice_tables: dict[tuple[str, str], list[dict]]) -> None:
    lines = [
        "# AttackQA Retriever Evaluation",
        "",
        "This report evaluates the saved BM25 and dense indexes on the AttackQA dev/test QA splits.",
        "",
        "## Overall Metrics",
        "",
        "| Split | Retriever | Queries | Recall@1 | Recall@5 | Recall@10 | MRR@10 | Mean Rank | Median Rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for summary in summaries:
        lines.append(
            f"| {summary.split} | {summary.retriever} | {summary.num_queries} | "
            f"{summary.recall_at_1:.3f} | {summary.recall_at_5:.3f} | {summary.recall_at_10:.3f} | "
            f"{summary.mrr_at_10:.3f} | {summary.mean_rank:.2f} | {summary.median_rank:.2f} |"
        )

    lines.extend([
        "",
        "## Slice Metrics",
        "",
    ])

    for (split, retriever_name), table in slice_tables.items():
        lines.extend([f"### {split.upper()} - {retriever_name}", "", "| Slice | Count | Recall@1 | Recall@5 | Recall@10 | MRR@10 |", "|---|---:|---:|---:|---:|---:|"])
        for row in table:
            slice_key = next(k for k in row.keys() if k not in {"count", "recall@1", "recall@5", "recall@10", "mrr@10"})
            lines.append(
                f"| {row[slice_key]} | {row['count']} | {row['recall@1']:.3f} | {row['recall@5']:.3f} | {row['recall@10']:.3f} | {row['mrr@10']:.3f} |"
            )
        lines.append("")

    lines.extend([
        "## Interpretation",
        "",
        "- `Recall@k` tells us whether the gold document appears in the top-k retrieved results.",
        "- `MRR@10` rewards the system for ranking the correct document earlier.",
        "- Slice tables show whether the retriever is systematically better on direct lookup questions, relation-heavy questions, or long-document cases.",
        "",
        "The full per-query outputs are saved beside the report for manual error analysis.",
    ])

    (RESULTS_DIR / "retrieval_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="*", default=["dev", "test"], help="QA splits to evaluate")
    parser.add_argument("--retrievers", nargs="*", default=["bm25", "dense"], help="Retrievers to evaluate")
    args = parser.parse_args()

    ensure_dirs()

    summaries = []
    slice_tables = {}
    metrics_by_split_retriever = {}

    for split in args.splits:
        for retriever_name in args.retrievers:
            summary, rows = evaluate_split(split, retriever_name)
            summaries.append(summary)
            metrics_by_split_retriever[(split, retriever_name)] = summary
            save_csv(rows, RESULTS_DIR / f"{split}_{retriever_name}_predictions.csv")
            slice_tables[(split, retriever_name)] = slice_metrics(rows, "question_category")
            save_csv(slice_metrics(rows, "question_category"), RESULTS_DIR / f"{split}_{retriever_name}_by_question_category.csv")
            save_csv(slice_metrics(rows, "doc_length_bucket"), RESULTS_DIR / f"{split}_{retriever_name}_by_doc_bucket.csv")

    for split in args.splits:
        save_bar_comparison(metrics_by_split_retriever, split)

    with open(RESULTS_DIR / "retrieval_summary.json", "w", encoding="utf-8") as handle:
        json.dump([asdict(summary) for summary in summaries], handle, indent=2)

    write_report(summaries, slice_tables)
    print(f"Wrote retriever evaluation artifacts to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
