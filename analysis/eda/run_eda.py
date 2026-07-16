"""Exploratory data analysis for the processed ATT&CK corpus.

This script reads the processed JSONL artifacts under data/processed and
writes summary tables, charts, and a short markdown report into the same
analysis workspace.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DOCS_PATH = ROOT / "data" / "processed" / "attack_docs.jsonl"
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
OUT_DIR = Path(__file__).resolve().parent


@dataclass
class EDAStats:
    doc_count: int
    chunk_count: int
    domains: dict[str, int]
    attack_types: dict[str, int]
    chunk_counts_per_doc: dict[str, int]
    description_length: dict[str, float]
    chunk_text_length: dict[str, float]
    related_context_lines: dict[str, float]


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def describe(values: list[int | float]) -> dict[str, float]:
    if not values:
        return {"min": math.nan, "median": math.nan, "mean": math.nan, "max": math.nan}
    return {
        "min": float(min(values)),
        "median": float(median(values)),
        "mean": float(mean(values)),
        "max": float(max(values)),
    }


def save_bar_chart(counter: Counter, title: str, xlabel: str, ylabel: str, filename: str, top_n: int | None = None) -> None:
    items = counter.most_common(top_n)
    labels = [item[0] for item in items]
    counts = [item[1] for item in items]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, counts, color="#1f77b4")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=35, ha="right")
    for bar, value in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=180)
    plt.close()


def save_histogram(values: list[int], title: str, xlabel: str, filename: str, bins: int = 30) -> None:
    plt.figure(figsize=(12, 6))
    plt.hist(values, bins=bins, color="#ff7f0e", edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=180)
    plt.close()


def compute_stats(docs: list[dict], chunks: list[dict]) -> EDAStats:
    domains = Counter(doc.get("domain", "unknown") for doc in docs)
    attack_types = Counter(doc.get("attack_type", "unknown") for doc in docs)
    chunk_counts = Counter(chunk.get("doc_id", "unknown") for chunk in chunks)

    description_lengths = [len(doc.get("description", "")) for doc in docs]
    chunk_text_lengths = [len(chunk.get("text", "")) for chunk in chunks]
    related_context_counts = [len(chunk.get("text", "").split("\n\n")[-1].splitlines()) if "\n\n" in chunk.get("text", "") else 0 for chunk in chunks]

    return EDAStats(
        doc_count=len(docs),
        chunk_count=len(chunks),
        domains=dict(domains),
        attack_types=dict(attack_types),
        chunk_counts_per_doc=describe(list(chunk_counts.values())),
        description_length=describe(description_lengths),
        chunk_text_length=describe(chunk_text_lengths),
        related_context_lines=describe(related_context_counts),
    )


def write_report(stats: EDAStats) -> None:
    lines = [
        "# ATT&CK Corpus EDA",
        "",
        f"- Documents: {stats.doc_count}",
        f"- Chunks: {stats.chunk_count}",
        "",
        "## Domains",
    ]
    for domain, count in sorted(stats.domains.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- {domain}: {count}")

    lines.extend([
        "",
        "## Attack Types",
    ])
    for attack_type, count in sorted(stats.attack_types.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- {attack_type}: {count}")

    lines.extend([
        "",
        "## Description Length (characters)",
        f"- min: {stats.description_length['min']:.0f}",
        f"- median: {stats.description_length['median']:.1f}",
        f"- mean: {stats.description_length['mean']:.1f}",
        f"- max: {stats.description_length['max']:.0f}",
        "",
        "## Chunks per Document",
        f"- min: {stats.chunk_counts_per_doc['min']:.0f}",
        f"- median: {stats.chunk_counts_per_doc['median']:.1f}",
        f"- mean: {stats.chunk_counts_per_doc['mean']:.1f}",
        f"- max: {stats.chunk_counts_per_doc['max']:.0f}",
    ])

    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_out_dir()

    docs = load_jsonl(DOCS_PATH)
    chunks = load_jsonl(CHUNKS_PATH)
    stats = compute_stats(docs, chunks)

    save_bar_chart(Counter(stats.domains), "Documents by Domain", "Domain", "Documents", "domains.png")
    save_bar_chart(Counter(stats.attack_types), "Documents by ATT&CK Type", "Attack Type", "Documents", "attack_types.png")
    save_histogram([len(doc.get("description", "")) for doc in docs], "Description Length Distribution", "Description length (characters)", "description_length.png")
    save_histogram([len(chunk.get("text", "")) for chunk in chunks], "Chunk Text Length Distribution", "Chunk length (characters)", "chunk_text_length.png")
    save_histogram(list(Counter(chunk.get("doc_id", "unknown") for chunk in chunks).values()), "Chunks per Document", "Chunks per document", "chunks_per_doc.png")

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(stats), handle, indent=2)

    write_report(stats)
    print(f"Wrote EDA artifacts to {OUT_DIR}")


if __name__ == "__main__":
    main()
