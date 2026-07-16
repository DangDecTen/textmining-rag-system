"""
Core ingestion logic for AttackQA -> (corpus, qa_examples).

Design notes
------------
- The raw parquet already contains atomic, pre-segmented text snippets in the
  `document` column. There is no free-text splitting to do here; ingestion is
  about (1) deduplicating documents that are shared across multiple QA pairs,
  (2) assigning each unique document a stable id, and (3) linking every QA
  pair back to that id so retrieval evaluation has clean ground truth.
- Dedup key: sha1 hash of the *normalized* document text (only strip leading/
  trailing whitespace, keep internal structure/newlines since AttackQA uses
  them meaningfully, e.g. "How data component X detects technique Y:\\n...").
- Metadata for a deduplicated document is taken from the first row it appears
  in. We separately report cases where the same doc_id shows up under more
  than one distinct subject_id (a signal worth checking manually).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

DOCUMENT_COLUMN = "document"
SOURCE_COLUMN = "source"

METADATA_COLUMNS = [
    "subject_id",
    "subject_name",
    "subject_type",
    "source",
    "field",
    "relation_id",
    "relation_name",
    "url",
    "references",
]


def normalize_text(text: str) -> str:
    return text.strip()


def compute_doc_id(text: str) -> str:
    normalized = normalize_text(text)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def build_corpus(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deduplicate documents and return (corpus_df, dup_report_df).

    corpus_df: one row per unique doc_id with canonical metadata.
    dup_report_df: doc_ids that appear under more than one distinct subject_id,
        for manual inspection (should usually be empty/rare).
    """
    df = df.copy()
    df["doc_id"] = df[DOCUMENT_COLUMN].map(compute_doc_id)

    # canonical metadata = first occurrence per doc_id
    available_meta_cols = [c for c in METADATA_COLUMNS if c in df.columns]
    corpus_df = (
        df.drop_duplicates(subset="doc_id", keep="first")
        .loc[:, ["doc_id", DOCUMENT_COLUMN] + available_meta_cols]
        .rename(columns={DOCUMENT_COLUMN: "text"})
        .reset_index(drop=True)
    )

    # diagnostic: doc_ids whose rows disagree on subject_id
    subj_nunique = df.groupby("doc_id")["subject_id"].nunique()
    ambiguous_doc_ids = subj_nunique[subj_nunique > 1].index
    dup_report_df = df[df["doc_id"].isin(ambiguous_doc_ids)][
        ["doc_id", "subject_id", SOURCE_COLUMN]
    ].drop_duplicates()

    return corpus_df, dup_report_df


def build_qa_examples(df: pd.DataFrame) -> pd.DataFrame:
    """Attach doc_id to every QA row so it references the deduped corpus."""
    df = df.copy()
    df["doc_id"] = df[DOCUMENT_COLUMN].map(compute_doc_id)
    df["qa_id"] = [hashlib.sha1(f"{i}-{q}".encode("utf-8")).hexdigest()[:16]
                    for i, q in enumerate(df["question"])]
    qa_cols = [
        "qa_id", "question", "answer", "thought", "doc_id",
        SOURCE_COLUMN, "human_question", "human_answer",
    ]
    qa_cols = [c for c in qa_cols if c in df.columns]
    return df.loc[:, qa_cols].reset_index(drop=True)


@dataclass
class SplitResult:
    train: pd.DataFrame
    dev: pd.DataFrame
    test: pd.DataFrame
    fallback_sources: list[str]


def stratified_split(
    qa_df: pd.DataFrame,
    stratify_col: str = SOURCE_COLUMN,
    train_size: float = 0.8,
    dev_size: float = 0.1,
    test_size: float = 0.1,
    thin_class_threshold: int = 10,
    random_state: int = 42,
) -> SplitResult:
    """80/10/10 split stratified by `stratify_col`.

    Implemented as a manual per-group allocator (round each group's rows into
    train/dev/test by the target proportions) rather than sklearn's
    `train_test_split(..., stratify=...)`. sklearn's stratify requires every
    class to have enough members to survive *two* nested splits
    (train -> temp -> dev/test), which raises on small classes -- and AttackQA
    has some rare `source` categories, so this would break in practice.
    The manual allocator works for any group size, including a single row
    (which goes entirely to train). Groups smaller than `thin_class_threshold`
    are reported in `fallback_sources` so you know their dev/test coverage is
    thin and metrics on those slices will be noisy.
    """
    assert abs(train_size + dev_size + test_size - 1.0) < 1e-6

    rng = np.random.RandomState(random_state)
    train_idx, dev_idx, test_idx = [], [], []
    thin_classes = []

    for cls, group in qa_df.groupby(stratify_col):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        if n < thin_class_threshold:
            thin_classes.append(cls)
        n_train = round(n * train_size)
        n_dev = round(n * dev_size)
        n_train = min(n_train, n)
        n_dev = min(n_dev, n - n_train)
        train_idx.extend(idx[:n_train])
        dev_idx.extend(idx[n_train:n_train + n_dev])
        test_idx.extend(idx[n_train + n_dev:])

    def _materialize(idx_list):
        return (
            qa_df.loc[idx_list]
            .sample(frac=1, random_state=random_state)
            .reset_index(drop=True)
        )

    train_df, dev_df, test_df = _materialize(train_idx), _materialize(dev_idx), _materialize(test_idx)
    return SplitResult(train_df, dev_df, test_df, fallback_sources=thin_classes)
