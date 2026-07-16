"""
Entry point: attackqa.parquet -> data/processed/{corpus, qa_train, qa_dev, qa_test, dup_report}.jsonl

Usage:
    python -m src.ingestion.run_ingestion \
        --input data/benchmark/attackqa.parquet \
        --output-dir data/processed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data_models.data_models import Document, QAExample
from src.ingestion.core import build_corpus, build_qa_examples, stratified_split


def _row_to_document(row: dict) -> Document:
    return Document(
        doc_id=row["doc_id"],
        text=row["text"],
        subject_id=row["subject_id"],
        subject_name=row.get("subject_name"),
        subject_type=row.get("subject_type"),
        source=row["source"],
        field=row.get("field"),
        relation_id=row.get("relation_id"),
        relation_name=row.get("relation_name"),
        url=row["url"],
        references=row.get("references"),
    )


def _row_to_qa_example(row: dict) -> QAExample:
    return QAExample(
        qa_id=row["qa_id"],
        question=row["question"],
        answer=row["answer"],
        thought=row.get("thought"),
        relevant_doc_ids=[row["doc_id"]],
        source=row["source"],
        human_question=bool(row.get("human_question", False)),
        human_answer=bool(row.get("human_answer", False)),
    )


def _write_jsonl(records: list, path: Path) -> None:
    with open(path, "w") as f:
        for r in records:
            obj = r.model_dump() if hasattr(r, "model_dump") else r
            f.write(json.dumps(obj) + "\n")


def run(input_path: str, output_dir: str) -> None:
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    # pandas represents missing optional fields as float NaN, which serializes
    # to a bare `NaN` token -- not valid JSON. Convert to None before any
    # downstream processing so every writer emits valid JSON `null`.
    # Columns that are entirely null (e.g. relation_id on rows where it's
    # never used) get inferred as float64 by pandas; .where() alone does not
    # upcast such columns back to object, so NaN survives unless we force
    # object dtype first.
    df = df.astype(object).where(pd.notnull(df), None)

    corpus_df, dup_report_df = build_corpus(df)
    qa_df = build_qa_examples(df)
    split = stratified_split(qa_df)

    documents = [_row_to_document(row) for row in corpus_df.to_dict("records")]
    _write_jsonl(documents, output_dir_p / "corpus.jsonl")

    for split_name, split_df in [("train", split.train), ("dev", split.dev), ("test", split.test)]:
        examples = [_row_to_qa_example(row) for row in split_df.to_dict("records")]
        _write_jsonl(examples, output_dir_p / f"qa_{split_name}.jsonl")

    dup_report_df.to_json(output_dir_p / "dup_report.jsonl", orient="records", lines=True)

    print(f"Queries:        {len(df)}")
    print(f"Corpus:         {len(corpus_df)}")
    print(f"Rel D/Q:        {1}     (sparse retrieval)")
    print(f"QA pairs:               {len(qa_df)}")
    print(f"    train/dev/test:     {len(split.train)} / {len(split.dev)} / {len(split.test)}")
    if split.fallback_sources:
        print(f"    thin sources (<10 rows): {split.fallback_sources}")
    if len(dup_report_df):
        print(f"    ambiguous doc_ids (same text, >1 subject_id) -- inspect dup_report.jsonl: "
              f"{dup_report_df['doc_id'].nunique()}")
    print(f"Wrote outputs to {output_dir_p}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/benchmark/attackqa.parquet")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()
    run(args.input, args.output_dir)
