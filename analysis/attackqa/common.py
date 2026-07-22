"""Shared helpers for AttackQA EDA and retrieval evaluation."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median

import numpy as np
import pandas as pd

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_./:-]+")


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall((text or "").lower())


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def token_recall(reference: str, candidate: str) -> float:
    ref_tokens = token_set(reference)
    if not ref_tokens:
        return 0.0
    cand_tokens = token_set(candidate)
    return len(ref_tokens & cand_tokens) / len(ref_tokens)


def token_jaccard(a: str, b: str) -> float:
    a_tokens = token_set(a)
    b_tokens = token_set(b)
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def exact_substring(text: str, container: str) -> bool:
    needle = (text or "").strip().lower()
    if not needle:
        return False
    return needle in (container or "").lower()


def describe(values: list[int | float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": float(min(values)),
        "median": float(median(values)),
        "mean": float(mean(values)),
        "max": float(max(values)),
    }


def top_counts(values: list[str], n: int = 10) -> list[tuple[str, int]]:
    return Counter(values).most_common(n)


def question_category(question: str) -> str:
    q = (question or "").strip().lower()

    relation_keywords = (
        "detect",
        "detection",
        "mitigat",
        "indicator",
        "sign of",
        "used by",
        "use attack technique",
        "use technique",
        "benefit",
        "relationship",
    )
    reasoning_prefixes = ("how ", "why ", "in what way")
    rewrite_prefixes = ("describe", "explain", "summarize", "list", "outline", "translate", "convert")
    lookup_prefixes = ("what ", "which ", "who ", "where ", "when ", "is ", "are ", "can ", "could ", "should ")

    if any(keyword in q for keyword in relation_keywords):
        return "relation_or_detection"
    if q.startswith(reasoning_prefixes):
        return "reasoning_translation"
    if q.startswith(rewrite_prefixes):
        return "rewrite_or_summarize"
    if q.startswith(lookup_prefixes):
        return "direct_lookup"
    return "other"


def doc_length_bucket(length: int, short_cutoff: float, long_cutoff: float) -> str:
    if length <= short_cutoff:
        return "short"
    if length >= long_cutoff:
        return "long"
    return "medium"


def length_quantiles(lengths: list[int], low_q: float = 0.33, high_q: float = 0.67) -> tuple[float, float]:
    series = pd.Series(lengths, dtype="float64")
    return float(series.quantile(low_q)), float(series.quantile(high_q))


def attach_documents(qa_rows: list[dict], corpus_lookup: dict[str, dict]) -> list[dict]:
    enriched = []
    for row in qa_rows:
        relevant_doc_ids = row.get("relevant_doc_ids") or []
        doc_id = relevant_doc_ids[0] if relevant_doc_ids else None
        doc = corpus_lookup.get(doc_id, {})
        enriched.append({
            **row,
            "doc_id": doc_id,
            "document_text": doc.get("text", ""),
            "document_subject_id": doc.get("subject_id"),
            "document_subject_name": doc.get("subject_name"),
            "document_subject_type": doc.get("subject_type"),
            "document_source": doc.get("source"),
            "document_field": doc.get("field"),
            "document_relation_name": doc.get("relation_name"),
            "document_length": len(doc.get("text", "")),
        })
    return enriched


def build_overlap_profile(row: dict) -> dict[str, float | str]:
    answer = row.get("answer", "")
    question = row.get("question", "")
    thought = row.get("thought", "") or ""
    document_text = row.get("document_text", "")

    doc_recall = token_recall(answer, document_text)
    question_recall = token_recall(answer, question)
    thought_recall = token_recall(answer, thought)

    overlaps = {
        "document": doc_recall,
        "question": question_recall,
        "thought": thought_recall,
    }
    dominant_field = max(overlaps, key=overlaps.get)

    if exact_substring(answer, document_text) or doc_recall >= 0.80:
        answer_style = "extractive"
    elif doc_recall >= 0.40:
        answer_style = "semi_extractive"
    else:
        answer_style = "abstractive"

    return {
        "answer_doc_recall": doc_recall,
        "answer_question_recall": question_recall,
        "answer_thought_recall": thought_recall,
        "answer_doc_jaccard": token_jaccard(answer, document_text),
        "answer_question_jaccard": token_jaccard(answer, question),
        "answer_thought_jaccard": token_jaccard(answer, thought),
        "answer_in_document": exact_substring(answer, document_text),
        "answer_in_question": exact_substring(answer, question),
        "answer_in_thought": exact_substring(answer, thought),
        "dominant_overlap_field": dominant_field,
        "answer_style": answer_style,
    }


def median_string(values: list[int | float]) -> str:
    if not values:
        return "0"
    return f"{median(values):.1f}"
