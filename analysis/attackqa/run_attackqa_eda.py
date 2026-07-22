"""Detailed EDA for AttackQA.

Outputs are written under analysis/attackqa/:
- figures/*.png
- summary.json
- analysis.md
- data/*.jsonl
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from analysis.attackqa.common import (
    attach_documents,
    build_overlap_profile,
    describe,
    doc_length_bucket,
    length_quantiles,
    load_jsonl,
    question_category,
    top_counts,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"
REPORT_PATH = ROOT / "analysis.md"
SUMMARY_PATH = ROOT / "summary.json"

SPLITS = ("train", "dev", "test")


@dataclass
class AttackQAEDAStats:
    corpus_size: int
    qa_size: int
    split_counts: dict[str, int]
    corpus_source_counts: dict[str, int]
    corpus_subject_type_counts: dict[str, int]
    corpus_length_stats: dict[str, float]
    corpus_length_quantiles: dict[str, float]
    question_category_counts: dict[str, int]
    doc_bucket_counts: dict[str, int]
    doc_bucket_thresholds: dict[str, float]
    answer_style_counts: dict[str, int]
    answer_overlap_stats: dict[str, dict[str, float]]
    exact_match_rates: dict[str, float]
    dominant_overlap_field_counts: dict[str, int]
    top_question_prefixes: list[tuple[str, int]]
    short_doc_top_question_categories: dict[str, int]
    long_doc_top_question_categories: dict[str, int]
    short_doc_top_sources: list[tuple[str, int]]
    long_doc_top_sources: list[tuple[str, int]]


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_processed_data() -> tuple[list[dict], dict[str, list[dict]]]:
    corpus = load_jsonl(PROCESSED_DIR / "corpus.jsonl")
    splits = {split: load_jsonl(PROCESSED_DIR / f"qa_{split}.jsonl") for split in SPLITS}
    return corpus, splits


def build_rows(corpus: list[dict], splits: dict[str, list[dict]]) -> list[dict]:
    corpus_lookup = {row["doc_id"]: row for row in corpus}
    all_rows: list[dict] = []
    for split_name, rows in splits.items():
        enriched = attach_documents(rows, corpus_lookup)
        for row in enriched:
            row["split"] = split_name
            row["question_category"] = question_category(row["question"])
            row.update(build_overlap_profile(row))
            all_rows.append(row)
    return all_rows


def build_stats(corpus: list[dict], rows: list[dict]) -> AttackQAEDAStats:
    corpus_lengths = [len(row["text"]) for row in corpus]
    q_low, q_high = length_quantiles(corpus_lengths, 0.33, 0.67)

    for row in rows:
        row["doc_length_bucket"] = doc_length_bucket(row["document_length"], q_low, q_high)

    split_counts = Counter(row["split"] for row in rows)
    corpus_source_counts = Counter(row["source"] for row in corpus)
    corpus_subject_type_counts = Counter(row.get("subject_type") or "None" for row in corpus)
    question_category_counts = Counter(row["question_category"] for row in rows)
    doc_bucket_counts = Counter(row["doc_length_bucket"] for row in rows)
    answer_style_counts = Counter(row["answer_style"] for row in rows)
    dominant_overlap_field_counts = Counter(row["dominant_overlap_field"] for row in rows)

    overlap_stats = {
        field: describe([row[f"answer_{field}_recall"] for row in rows])
        for field in ("doc", "question", "thought")
    }
    exact_match_rates = {
        field: sum(1 for row in rows if row[f"answer_in_{field}"]) / len(rows)
        for field in ("document", "question", "thought")
    }

    question_prefixes = Counter(
        (row["question"].split()[0].lower().strip('"\'“”‘’।,?!:;') if row["question"].split() else "")
        for row in rows
    )

    short_rows = [row for row in rows if row["doc_length_bucket"] == "short"]
    long_rows = [row for row in rows if row["doc_length_bucket"] == "long"]

    return AttackQAEDAStats(
        corpus_size=len(corpus),
        qa_size=len(rows),
        split_counts=dict(split_counts),
        corpus_source_counts=dict(corpus_source_counts),
        corpus_subject_type_counts=dict(corpus_subject_type_counts),
        corpus_length_stats=describe(corpus_lengths),
        corpus_length_quantiles={"q33": q_low, "q67": q_high},
        question_category_counts=dict(question_category_counts),
        doc_bucket_counts=dict(doc_bucket_counts),
        doc_bucket_thresholds={"short_cutoff": q_low, "long_cutoff": q_high},
        answer_style_counts=dict(answer_style_counts),
        answer_overlap_stats=overlap_stats,
        exact_match_rates=exact_match_rates,
        dominant_overlap_field_counts=dict(dominant_overlap_field_counts),
        top_question_prefixes=top_counts([p for p in question_prefixes.elements() if p], 15),
        short_doc_top_question_categories=dict(Counter(row["question_category"] for row in short_rows).most_common(10)),
        long_doc_top_question_categories=dict(Counter(row["question_category"] for row in long_rows).most_common(10)),
        short_doc_top_sources=Counter(row["source"] for row in short_rows).most_common(10),
        long_doc_top_sources=Counter(row["source"] for row in long_rows).most_common(10),
    )


def save_bar(labels: list[str], values: list[float], title: str, xlabel: str, ylabel: str, filename: str) -> None:
    plt.figure(figsize=(13, 6))
    bars = plt.bar(labels, values, color="#1f77b4")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=35, ha="right")
    for bar, value in zip(bars, values):
        label = f"{value:.0f}" if float(value).is_integer() else f"{value:.3f}"
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=180)
    plt.close()


def save_hist(values: list[int], title: str, xlabel: str, filename: str, bins: int = 40) -> None:
    plt.figure(figsize=(12, 6))
    plt.hist(values, bins=bins, color="#ff7f0e", edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=180)
    plt.close()


def save_stacked_question_buckets(rows: list[dict]) -> None:
    categories = ["direct_lookup", "relation_or_detection", "reasoning_translation", "rewrite_or_summarize", "other"]
    buckets = ["short", "medium", "long"]
    matrix = {bucket: Counter() for bucket in buckets}
    for row in rows:
        matrix[row["doc_length_bucket"]][row["question_category"]] += 1

    bottom = [0] * len(buckets)
    plt.figure(figsize=(12, 6))
    for category in categories:
        values = [matrix[bucket][category] for bucket in buckets]
        plt.bar(buckets, values, bottom=bottom, label=category)
        bottom = [b + v for b, v in zip(bottom, values)]
    plt.title("Question Categories by Document Length Bucket")
    plt.xlabel("Document length bucket")
    plt.ylabel("QA rows")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "question_category_by_doc_bucket.png", dpi=180)
    plt.close()


def write_report(stats: AttackQAEDAStats) -> None:
    q_counts = Counter(stats.question_category_counts)
    total_q = sum(q_counts.values())
    total_dominant = sum(stats.dominant_overlap_field_counts.values())

    lines = [
        "# AttackQA Dataset EDA",
        "",
        "This report uses the current AttackQA split files in `data/processed/` and the deduplicated corpus produced by ingestion.",
        "",
        "## 1. Dataset Overview",
        "",
        f"- Corpus documents: {stats.corpus_size}",
        f"- QA rows: {stats.qa_size}",
        f"- Splits: train={stats.split_counts.get('train', 0)}, dev={stats.split_counts.get('dev', 0)}, test={stats.split_counts.get('test', 0)}",
        f"- Corpus doc length median: {stats.corpus_length_stats['median']:.0f} chars",
        f"- Corpus doc length mean: {stats.corpus_length_stats['mean']:.1f} chars",
        "",
        "### Corpus source mix",
    ]

    for source, count in Counter(stats.corpus_source_counts).most_common(12):
        lines.append(f"- {source}: {count}")

    lines.extend([
        "",
        "### Corpus subject-type mix",
    ])
    for subject_type, count in Counter(stats.corpus_subject_type_counts).most_common(12):
        lines.append(f"- {subject_type}: {count}")

    lines.extend([
        "",
        "## 2. Question Analysis",
        "",
        "The questions were grouped with a lightweight heuristic into five intent buckets. The goal is not perfect NLP labeling, but a practical decomposition of retrieval difficulty.",
        "",
    ])

    for category, count in q_counts.most_common():
        lines.append(f"- {category}: {count} ({count / total_q * 100:.1f}%)")

    lines.extend([
        "",
        "### Top question openings",
    ])
    for prefix, count in stats.top_question_prefixes:
        lines.append(f"- {prefix}: {count}")

    lines.extend([
        "",
        "## 3. Document Length Analysis",
        "",
        f"Length buckets are based on the 33rd and 67th percentiles of corpus document length: short <= {stats.doc_bucket_thresholds['short_cutoff']:.1f}, long >= {stats.doc_bucket_thresholds['long_cutoff']:.1f}.",
        "",
        "### Document length buckets",
    ])
    for bucket, count in stats.doc_bucket_counts.items():
        lines.append(f"- {bucket}: {count} ({count / stats.qa_size * 100:.1f}% of QA rows)")

    lines.extend([
        "",
        "### What short documents tend to ask about",
    ])
    for category, count in stats.short_doc_top_question_categories.items():
        lines.append(f"- {category}: {count}")
    lines.extend([
        "",
        "Top sources for short documents:",
    ])
    for src, count in stats.short_doc_top_sources:
        lines.append(f"- {src}: {count}")

    lines.extend([
        "",
        "### What long documents tend to ask about",
    ])
    for category, count in stats.long_doc_top_question_categories.items():
        lines.append(f"- {category}: {count}")
    lines.extend([
        "",
        "Top sources for long documents:",
    ])
    for src, count in stats.long_doc_top_sources:
        lines.append(f"- {src}: {count}")

    lines.extend([
        "",
        "## 4. Answer vs Document / Thought / Question",
        "",
        f"- Answer tokens contained in document on average: {stats.answer_overlap_stats['doc']['mean']:.3f}",
        f"- Answer tokens contained in thought on average: {stats.answer_overlap_stats['thought']['mean']:.3f}",
        f"- Answer tokens contained in question on average: {stats.answer_overlap_stats['question']['mean']:.3f}",
        f"- Exact answer substring in document: {stats.exact_match_rates['document'] * 100:.1f}%",
        f"- Exact answer substring in thought: {stats.exact_match_rates['thought'] * 100:.1f}%",
        f"- Exact answer substring in question: {stats.exact_match_rates['question'] * 100:.1f}%",
        "",
        "### Dominant answer keyword source",
    ])
    for field, count in stats.dominant_overlap_field_counts.items():
        lines.append(f"- {field}: {count} ({count / total_dominant * 100:.1f}%)")

    lines.extend([
        "",
        f"- Extractive answers: {stats.answer_style_counts.get('extractive', 0)}",
        f"- Semi-extractive answers: {stats.answer_style_counts.get('semi_extractive', 0)}",
        f"- Abstractive answers: {stats.answer_style_counts.get('abstractive', 0)}",
        "",
        "Interpretation:",
        "- The document field is the main anchor for answer tokens, which means retrieval quality matters directly for answer generation.",
        "- The thought field contributes additional paraphrastic reasoning in some examples, but it is not usually the main lexical source of the answer.",
        "- Questions themselves rarely contain the full answer text; they usually encode the intent, not the answer span.",
        "",
        "## 5. Evaluation-oriented interpretation",
        "",
        "- Direct lookup questions should be easiest for exact retrieval.",
        "- Relation and detection questions need retrieval plus a small amount of reasoning over the returned document.",
        "- Rewrite/summarize questions are the hardest because the answer must be translated into a user-facing format.",
        "- Long documents are not simply harder because they are long; they also tend to carry richer relation context and broader explanatory text.",
        "",
        "## 6. Figure-by-Figure Notes",
        "",
        "### `source_distribution.png`",
        "Shows how the deduplicated corpus is constructed by ATT&CK source template. A few source templates dominate, which means the corpus is structured and long-tailed rather than uniform.",
        "",
        "### `subject_type_distribution.png`",
        "Shows that corpus metadata mixes broad categories and ATT&CK IDs. This is useful as a data-quality warning, not as a clean class taxonomy.",
        "",
        "### `question_category_distribution.png`",
        "Shows the relative balance of lookup, relation/detection, reasoning, and rewrite-style questions. This is the main chart for question difficulty.",
        "",
        "### `question_category_by_doc_bucket.png`",
        "Shows how question intent shifts as documents get shorter or longer. Short documents lean more toward direct lookup and relation questions; long documents lean more toward reasoning and rewrite questions.",
        "",
        "### `document_length_distribution.png`",
        "Shows the corpus is length-skewed with a long tail of very verbose documents, but many examples are still compact.",
        "",
        "### `question_length_distribution.png`",
        "Questions stay comparatively short even when the supporting document is long, which is exactly the pattern you want for QA retrieval.",
        "",
        "### `answer_length_distribution.png`",
        "Answers are usually longer than questions and often shorter than the full document. This suggests a mix of extractive and paraphrased answers.",
        "",
        "### `answer_style_distribution.png`",
        "Separates extractive, semi-extractive, and abstractive answers. A large extractive/semi-extractive portion means many answers are grounded directly in the document text.",
        "",
        "### `answer_keyword_source.png`",
        "Shows which field usually carries the answer vocabulary. The document field should dominate if the retrieval pipeline is healthy; thought is secondary reasoning support; question is usually the intent signal.",
        "",
        "### `answer_field_recall.png`",
        "Shows average token recall from answer to document/question/thought. This is the cleanest field-level summary of where answer keywords come from.",
        "",
        "### `answer_exact_match_rate.png`",
        "Shows how often the exact answer string appears in each field. A high document rate means answers are often extractive or lightly paraphrased.",
        "",
        "### `answer_doc_recall_distribution.png`",
        "Shows the spread of answer-to-document token overlap. A right-skew toward high recall means many answers are grounded in the document; a broad spread means some are more abstract.",
        "",
        "## 7. Conclusion",
        "",
        "AttackQA is not a uniform QA dataset. It mixes direct lookup questions, relation-based questions, explanatory questions, and reformulation questions. Most answer content is grounded in the document text, with the thought field serving as secondary reasoning support. The corpus is also length-skewed: short documents are often crisp relation or detection snippets, while long documents are more likely to support reasoning and synthesis-style questions.",
    ])

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    corpus, splits = load_processed_data()
    write_jsonl(corpus, DATA_DIR / "attackqa_corpus_snapshot.jsonl")
    rows = build_rows(corpus, splits)

    stats = build_stats(corpus, rows)

    save_bar(
        list(stats.split_counts.keys()),
        list(stats.split_counts.values()),
        "AttackQA QA Split Distribution",
        "Split",
        "Rows",
        "split_distribution.png",
    )
    save_bar(
        [source for source, _ in Counter(stats.corpus_source_counts).most_common(12)],
        [count for _, count in Counter(stats.corpus_source_counts).most_common(12)],
        "Corpus Source Distribution",
        "Source",
        "Documents",
        "source_distribution.png",
    )
    save_bar(
        [subject_type for subject_type, _ in Counter(stats.corpus_subject_type_counts).most_common(12)],
        [count for _, count in Counter(stats.corpus_subject_type_counts).most_common(12)],
        "Corpus Subject-Type Distribution",
        "Subject type",
        "Documents",
        "subject_type_distribution.png",
    )
    save_bar(
        list(stats.question_category_counts.keys()),
        list(stats.question_category_counts.values()),
        "Question Category Distribution",
        "Question category",
        "Rows",
        "question_category_distribution.png",
    )
    save_hist(
        [row["document_length"] for row in rows],
        "Document Length Distribution",
        "Document length (characters)",
        "document_length_distribution.png",
    )
    save_hist(
        [len(row["question"]) for row in rows],
        "Question Length Distribution",
        "Question length (characters)",
        "question_length_distribution.png",
    )
    save_hist(
        [len(row["answer"]) for row in rows],
        "Answer Length Distribution",
        "Answer length (characters)",
        "answer_length_distribution.png",
    )
    save_hist(
        [row["answer_doc_recall"] for row in rows],
        "Answer-Document Token Recall Distribution",
        "Token recall against document",
        "answer_doc_recall_distribution.png",
    )
    save_bar(
        list(stats.answer_style_counts.keys()),
        list(stats.answer_style_counts.values()),
        "Answer Style Distribution",
        "Answer style",
        "Rows",
        "answer_style_distribution.png",
    )
    save_bar(
        list(stats.dominant_overlap_field_counts.keys()),
        list(stats.dominant_overlap_field_counts.values()),
        "Dominant Answer Keyword Source",
        "Field",
        "Rows",
        "answer_keyword_source.png",
    )
    save_bar(
        ["document", "thought", "question"],
        [stats.answer_overlap_stats[f]["mean"] for f in ("doc", "thought", "question")],
        "Mean Answer Token Recall by Field",
        "Field",
        "Mean token recall",
        "answer_field_recall.png",
    )
    save_bar(
        ["document", "thought", "question"],
        [stats.exact_match_rates[f] for f in ("document", "thought", "question")],
        "Exact Answer Substring Rate by Field",
        "Field",
        "Exact match rate",
        "answer_exact_match_rate.png",
    )
    save_stacked_question_buckets(rows)

    write_jsonl(rows, DATA_DIR / "attackqa_enriched_rows.jsonl")

    with open(SUMMARY_PATH, "w", encoding="utf-8") as handle:
        json.dump(asdict(stats), handle, indent=2)

    write_report(stats)
    print(f"Wrote AttackQA EDA artifacts to {ROOT}")


if __name__ == "__main__":
    main()
