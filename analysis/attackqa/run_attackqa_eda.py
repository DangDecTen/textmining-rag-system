"""Download AttackQA from Hugging Face and generate EDA artifacts.

Artifacts written by this script:
- analysis/attackqa/data/attackqa_train.jsonl
- analysis/attackqa/summary.json
- analysis/attackqa/analysis.md
- analysis/attackqa/figures/*.png
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt
from datasets import load_dataset


DATASET_NAME = "sambanovasystems/attackqa"
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"
RAW_SNAPSHOT = DATA_DIR / "attackqa_train.jsonl"
SUMMARY_PATH = ROOT / "summary.json"
REPORT_PATH = ROOT / "analysis.md"

QUESTION_LENGTH_BINS = 40
ANSWER_LENGTH_BINS = 40
DOCUMENT_LENGTH_BINS = 40
THOUGHT_LENGTH_BINS = 40
REFERENCE_COUNT_BINS = 8

ATTCK_ID_PATTERN = re.compile(r"^T\d+")
ATTCK_LIKE_PATTERN = re.compile(r"^T\d+(?:\.\d+)?$")
FIELD_CATEGORIES = {"description", "other"}


@dataclass
class AttackQAStats:
    row_count: int
    column_count: int
    columns: list[str]
    source_counts: dict[str, int]
    subject_type_counts: dict[str, int]
    subject_type_id_like_count: int
    subject_type_broad_count: int
    subject_type_unique_count: int
    human_question_counts: dict[str, int]
    human_answer_counts: dict[str, int]
    field_counts: dict[str, int]
    relation_name_counts: dict[str, int]
    references_stats: dict[str, float]
    question_length_stats: dict[str, float]
    answer_length_stats: dict[str, float]
    document_length_stats: dict[str, float]
    thought_length_stats: dict[str, float]
    top_subject_ids: list[tuple[str, int]]
    top_subject_names: list[tuple[str, int]]


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_attackqa_rows() -> list[dict]:
    dataset = load_dataset(DATASET_NAME, split="train")
    return dataset.to_list()


def save_raw_snapshot(rows: list[dict]) -> None:
    with open(RAW_SNAPSHOT, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def describe(values: list[int | float]) -> dict[str, float]:
    if not values:
        return {"min": math.nan, "median": math.nan, "mean": math.nan, "max": math.nan}
    return {
        "min": float(min(values)),
        "median": float(median(values)),
        "mean": float(mean(values)),
        "max": float(max(values)),
    }


def pct(part: int, whole: int) -> float:
    return 0.0 if whole == 0 else (part / whole) * 100.0


def top_counts(values: list[str], n: int = 10) -> list[tuple[str, int]]:
    return Counter(values).most_common(n)


def classify_subject_type(value: str) -> str:
    if ATTCK_LIKE_PATTERN.match(value):
        return "ATT&CK ID-like"
    if value in {"techniques", "software", "groups", "campaigns", "tactics"}:
        return value
    return "other"


def compute_stats(rows: list[dict]) -> AttackQAStats:
    source_counts = Counter(row["source"] for row in rows)
    subject_type_counts = Counter(row["subject_type"] for row in rows)
    human_question_counts = Counter(str(row["human_question"]) for row in rows)
    human_answer_counts = Counter(str(row["human_answer"]) for row in rows)
    field_counts = Counter((row["field"] or "None") for row in rows)
    relation_name_counts = Counter((row["relation_name"] or "None") for row in rows)

    references_count = [len(row["references"]) if row.get("references") else 0 for row in rows]
    question_lengths = [len(row["question"]) for row in rows]
    answer_lengths = [len(row["answer"]) for row in rows]
    document_lengths = [len(row["document"]) for row in rows]
    thought_lengths = [len(row["thought"]) for row in rows]

    subject_type_id_like_count = sum(1 for row in rows if ATTCK_ID_PATTERN.match(row["subject_type"]))
    subject_type_broad_count = sum(
        count for value, count in subject_type_counts.items() if value in {"techniques", "software", "groups", "campaigns", "tactics"}
    )

    return AttackQAStats(
        row_count=len(rows),
        column_count=len(rows[0]) if rows else 0,
        columns=list(rows[0].keys()) if rows else [],
        source_counts=dict(source_counts),
        subject_type_counts=dict(subject_type_counts),
        subject_type_id_like_count=subject_type_id_like_count,
        subject_type_broad_count=subject_type_broad_count,
        subject_type_unique_count=len(subject_type_counts),
        human_question_counts=dict(human_question_counts),
        human_answer_counts=dict(human_answer_counts),
        field_counts=dict(field_counts),
        relation_name_counts=dict(relation_name_counts),
        references_stats=describe(references_count),
        question_length_stats=describe(question_lengths),
        answer_length_stats=describe(answer_lengths),
        document_length_stats=describe(document_lengths),
        thought_length_stats=describe(thought_lengths),
        top_subject_ids=top_counts([row["subject_id"] for row in rows], 15),
        top_subject_names=top_counts([row["subject_name"] for row in rows], 15),
    )


def save_bar_chart(labels: list[str], counts: list[int], title: str, xlabel: str, ylabel: str, filename: str) -> None:
    plt.figure(figsize=(13, 6))
    bars = plt.bar(labels, counts, color="#1f77b4")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=35, ha="right")
    for bar, value in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=180)
    plt.close()


def save_histogram(values: list[int], title: str, xlabel: str, filename: str, bins: int = 30) -> None:
    plt.figure(figsize=(12, 6))
    plt.hist(values, bins=bins, color="#ff7f0e", edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=180)
    plt.close()


def save_length_grid(rows: list[dict]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    specs = [
        ("question", [len(row["question"]) for row in rows], QUESTION_LENGTH_BINS),
        ("answer", [len(row["answer"]) for row in rows], ANSWER_LENGTH_BINS),
        ("document", [len(row["document"]) for row in rows], DOCUMENT_LENGTH_BINS),
        ("thought", [len(row["thought"]) for row in rows], THOUGHT_LENGTH_BINS),
    ]
    for axis, (name, values, bins) in zip(axes.flat, specs):
        axis.hist(values, bins=bins, color="#2ca02c", edgecolor="white")
        axis.set_title(f"{name.title()} Length")
        axis.set_xlabel("Characters")
        axis.set_ylabel("Count")
    figure.suptitle("AttackQA Text Length Distributions")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(FIGURES_DIR / "length_distributions.png", dpi=180)
    plt.close(figure)


def write_report(stats: AttackQAStats, rows: list[dict]) -> None:
    row_count = stats.row_count
    human_question_true = int(stats.human_question_counts.get("True", 0))
    human_answer_true = int(stats.human_answer_counts.get("True", 0))
    description_count = int(stats.field_counts.get("description", 0))
    relation_name_none = int(stats.relation_name_counts.get("None", 0))

    lines = [
        "# AttackQA Dataset Analysis and Evaluation",
        "",
        f"Source: [sambanovasystems/attackqa](https://huggingface.co/datasets/sambanovasystems/attackqa)",
        "",
        "## 1. Dataset Overview",
        "",
        f"- Rows: {stats.row_count}",
        f"- Columns: {stats.column_count}",
        f"- Split analyzed: train",
        f"- Raw snapshot: `data/attackqa_train.jsonl`",
        "",
        "### Columns",
        "",
        "- " + "\n- ".join(stats.columns),
        "",
        "## 2. Numeric Profile",
        "",
        "### Main counts",
        f"- Human question rows: {human_question_true} ({pct(human_question_true, row_count):.1f}%)",
        f"- Human answer rows: {human_answer_true} ({pct(human_answer_true, row_count):.1f}%)",
        f"- Rows with `field=description`: {description_count} ({pct(description_count, row_count):.1f}%)",
        f"- Rows with no `relation_name`: {relation_name_none} ({pct(relation_name_none, row_count):.1f}%)",
        "",
        "### References per row",
        f"- Min: {stats.references_stats['min']:.0f}",
        f"- Median: {stats.references_stats['median']:.1f}",
        f"- Mean: {stats.references_stats['mean']:.2f}",
        f"- Max: {stats.references_stats['max']:.0f}",
        "",
        "### Text length in characters",
        f"- Question median: {stats.question_length_stats['median']:.0f}",
        f"- Answer median: {stats.answer_length_stats['median']:.0f}",
        f"- Document median: {stats.document_length_stats['median']:.0f}",
        f"- Thought median: {stats.thought_length_stats['median']:.0f}",
        f"- Question mean: {stats.question_length_stats['mean']:.1f}",
        f"- Answer mean: {stats.answer_length_stats['mean']:.1f}",
        f"- Document mean: {stats.document_length_stats['mean']:.1f}",
        f"- Thought mean: {stats.thought_length_stats['mean']:.1f}",
        "",
        "## 3. Categorical Characteristics",
        "",
        "### Source distribution",
    ]

    for value, count in top_counts([row["source"] for row in rows], 10):
        lines.append(f"- {value}: {count} ({pct(count, row_count):.1f}%)")

    lines.extend([
        "",
        "### Subject field quality",
        f"- Unique `subject_type` values: {stats.subject_type_unique_count}",
        f"- `subject_type` values that look like ATT&CK IDs: {stats.subject_type_id_like_count}",
        f"- Rows with broad subject categories (`techniques`, `software`, `groups`, `campaigns`, `tactics`): {stats.subject_type_broad_count}",
        "",
        "This field is not a clean taxonomy. It mixes broad categories with ATT&CK IDs, so it should be treated carefully in downstream analysis.",
        "",
        "### Top ATT&CK subjects",
        "",
    ])

    lines.append("#### By subject ID")
    for value, count in stats.top_subject_ids:
        lines.append(f"- {value}: {count}")

    lines.append("")
    lines.append("#### By subject name")
    for value, count in stats.top_subject_names:
        lines.append(f"- {value}: {count}")

    lines.extend([
        "",
        "## 4. Evaluation",
        "",
        "### What looks strong",
        "",
        "- The dataset is large enough for meaningful QA and retrieval experiments.",
        "- It is strongly grounded in ATT&CK relationships, which makes it useful for cyber-domain reasoning.",
        "- Questions are relatively short, while answers and documents provide more explanatory context.",
        "- Reference lists are usually small, which keeps the examples focused.",
        "",
        "### What needs caution",
        "",
        "- The dataset has only one split (`train`), so there is no built-in validation/test separation.",
        "- `subject_type` is noisy and overloaded, so it should not be used as a single authoritative label.",
        "- Nearly half the rows have no `relation_name`, which limits how much structure is available for some examples.",
        "- The corpus is skewed toward a small number of high-frequency ATT&CK techniques, so the dataset is long-tailed.",
        "",
        "### Practical interpretation",
        "",
        "- The dataset is well suited for training or evaluating ATT&CK-aware QA and retrieval systems.",
        "- It is less suitable as a clean classification dataset because several metadata fields are mixed or incomplete.",
        "- For RAG work, the answer and document fields are especially valuable because they contain the context needed to support grounding.",
        "",
        "## 5. Conclusion",
        "",
        "AttackQA is a large, ATT&CK-grounded question answering corpus with strong cybersecurity semantics and rich explanatory text. Its main strength is domain realism: the questions, answers, and supporting documents are tied to ATT&CK subjects and relations rather than generic QA pairs. Its main weakness is metadata inconsistency, especially in `subject_type`, which means any downstream analysis should rely on carefully chosen fields rather than assuming all columns are cleanly normalized.",
        "",
        "For next-step work, the best follow-ups are split-based evaluation, retrieval quality checks by subject type, and a comparison of examples that are human-authored versus machine-generated.",
    ])

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    rows = load_attackqa_rows()
    save_raw_snapshot(rows)
    stats = compute_stats(rows)

    save_bar_chart(
        [value for value, _ in top_counts([row["source"] for row in rows], 10)],
        [count for _, count in top_counts([row["source"] for row in rows], 10)],
        "AttackQA Examples by Source",
        "Source",
        "Rows",
        "source_distribution.png",
    )
    save_bar_chart(
        [value for value, _ in top_counts([row["subject_id"] for row in rows], 15)],
        [count for _, count in top_counts([row["subject_id"] for row in rows], 15)],
        "Top ATT&CK Subject IDs",
        "Subject ID",
        "Rows",
        "subject_id_top15.png",
    )
    save_bar_chart(
        [value for value, _ in top_counts([row["relation_name"] or "None" for row in rows], 15)],
        [count for _, count in top_counts([row["relation_name"] or "None" for row in rows], 15)],
        "Top Relation Names",
        "Relation name",
        "Rows",
        "relation_name_top15.png",
    )
    save_bar_chart(
        ["human_question=True", "human_question=False", "human_answer=True", "human_answer=False"],
        [
            int(stats.human_question_counts.get("True", 0)),
            int(stats.human_question_counts.get("False", 0)),
            int(stats.human_answer_counts.get("True", 0)),
            int(stats.human_answer_counts.get("False", 0)),
        ],
        "Human Authorship Flags",
        "Flag",
        "Rows",
        "human_flags.png",
    )
    save_bar_chart(
        list(stats.field_counts.keys()),
        list(stats.field_counts.values()),
        "Field Distribution",
        "Field",
        "Rows",
        "field_distribution.png",
    )
    save_bar_chart(
        ["ID-like", "Broad labels", "Other"],
        [
            stats.subject_type_id_like_count,
            stats.subject_type_broad_count,
            stats.row_count - stats.subject_type_id_like_count - stats.subject_type_broad_count,
        ],
        "Subject Type Quality",
        "Subject type bucket",
        "Rows",
        "subject_type_quality.png",
    )
    save_histogram([len(row["references"]) if row.get("references") else 0 for row in rows], "References per Row", "Reference count", "references_per_row.png", bins=REFERENCE_COUNT_BINS)
    save_length_grid(rows)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as handle:
        json.dump(asdict(stats), handle, indent=2)

    write_report(stats, rows)
    print(f"Wrote AttackQA artifacts to {ROOT}")


if __name__ == "__main__":
    main()
