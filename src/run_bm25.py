"""
Usage:
    python -m src.run_bm25 --split dev
"""


from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal
from src.config import settings
from src.data_models.io import load_corpus_lookup, load_qa_examples
from src.indexing.bm25_index import BM25Index
from src.retrieval.bm25_retriever import BM25Retriever
from src.data_models.data_models import Document
from src.eval.retrieval_eval import evaluate_retriever, print_report


# Index dir + BM25 hyperparameters live in src/config.py (single source of
# truth, shared with the API) -- not redefined here.
INDEX_DIR = settings.bm25_index_dir


def build_index(corpus: list[Document], rebuild: bool = False) -> BM25Index:
    index_dir_p = Path(INDEX_DIR)
    
    if rebuild or not index_dir_p.exists():
        print("Building BM25 index...")
        index = BM25Index(method=settings.bm25_method, k1=settings.bm25_k1, b=settings.bm25_b)
        index.build(corpus)
        index.save(str(index_dir_p))
        print(f"Saved index to {index_dir_p}/")
    else:
        print(f"Loading existing index from {index_dir_p}/")
        index = BM25Index.load(str(index_dir_p))
    
    return index


def main(rebuild_index: bool, split: Literal['train', 'dev', 'test']) -> None:
    corpus_lookup = load_corpus_lookup()
    qa_examples = load_qa_examples(split=split)
    print(f"Loaded {len(corpus_lookup)} documents, {len(qa_examples)} QA examples ({split})")
    print()

    index = build_index(corpus=list(corpus_lookup.values()), rebuild=rebuild_index)
    retriever = BM25Retriever(index=index, corpus_lookup=corpus_lookup)
    print()

    report = evaluate_retriever(retriever, qa_examples, k_values=(1, 5, 10))
    print_report(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Rebuild Index, or load from existing Index")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"], help="QA pairs to evaluate")
    args = parser.parse_args()
    main(args.rebuild, args.split)
