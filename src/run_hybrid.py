"""
Evaluation runner for Hybrid Retriever (Dense + BM25).

Usage:
    python -m src.run_hybrid --split dev
    python -m src.run_hybrid --split test
"""

from __future__ import annotations

import argparse
from typing import Literal

from src.data_models.io import load_corpus_lookup, load_qa_examples
from src.indexing.bm25_index import BM25Index
from src.indexing.dense_index import DenseIndex
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.eval.retrieval_eval import evaluate_retriever, print_report

DENSE_INDEX_DIR = "data/index/dense"
BM25_INDEX_DIR = "data/index/bm25"


def main(split: Literal["train", "dev", "test"], alpha: float = 0.5, use_rrf: bool = True) -> None:
    corpus_lookup = load_corpus_lookup()
    qa_examples = load_qa_examples(split=split)
    print(f"Loaded {len(corpus_lookup)} documents, {len(qa_examples)} QA examples ({split})")
    print()

    print(f"Loading Dense Index from {DENSE_INDEX_DIR}/...")
    dense_index = DenseIndex.load(DENSE_INDEX_DIR)
    dense_retriever = DenseRetriever(index=dense_index, corpus_lookup=corpus_lookup)

    print(f"Loading BM25 Index from {BM25_INDEX_DIR}/...")
    bm25_index = BM25Index.load(BM25_INDEX_DIR)
    bm25_retriever = BM25Retriever(index=bm25_index, corpus_lookup=corpus_lookup)

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        alpha=alpha,
        use_rrf=use_rrf,
    )

    mode_str = f"RRF (alpha={alpha})" if use_rrf else f"MinMax Weighted (alpha={alpha})"
    print(f"Evaluating Hybrid Retriever [{mode_str}] on {split} split...")
    report = evaluate_retriever(hybrid_retriever, qa_examples, k_values=(1, 5, 10))
    print_report(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"], help="QA pairs to evaluate")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dense weight alpha (default 0.5)")
    parser.add_argument("--score-fusion", action="store_true", help="Use MinMax weighted score fusion instead of RRF")
    args = parser.parse_args()
    main(split=args.split, alpha=args.alpha, use_rrf=not args.score_fusion)
