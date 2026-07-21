"""
Usage:
    python -m src.run_dense --split dev
"""


from __future__ import annotations
import argparse
from pathlib import Path
from typing import Literal
from src.data_models.io import load_corpus_lookup, load_qa_examples
from src.data_models.io import Document
from src.indexing.dense_index import DenseIndex
from src.retrieval.dense_retriever import DenseRetriever
from src.eval.retrieval_eval import evaluate_retriever, print_report


INDEX_DIR = "data/index/dense"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64


def build_index(corpus: list[Document], rebuild: bool = False) -> DenseIndex:
    index_dir_p = Path(INDEX_DIR)
    
    if rebuild or not index_dir_p.exists():
        print(f"Building dense index with {MODEL_NAME}...")
        index = DenseIndex(model_name=MODEL_NAME, batch_size=BATCH_SIZE)
        index.build(corpus)
        index.save(str(index_dir_p))
        print(f"Saved index to {index_dir_p}/")
    else:
        print(f"Loading existing index from {index_dir_p}/")
        index = DenseIndex.load(str(index_dir_p))

    return index


def main(rebuild_index: bool, split: Literal['train', 'dev', 'test']) -> None:
    corpus_lookup = load_corpus_lookup()
    qa_examples = load_qa_examples(split=split)
    print(f"Loaded {len(corpus_lookup)} documents, {len(qa_examples)} QA examples ({split})")
    print()

    index = build_index(corpus=list(corpus_lookup.values()), rebuild=rebuild_index)
    retriever = DenseRetriever(index=index, corpus_lookup=corpus_lookup)
    print()

    report = evaluate_retriever(retriever, qa_examples, k_values=(1, 5, 10))
    print_report(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Rebuild Index, or load from existing Index")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"], help="QA pairs to evaluate")
    args = parser.parse_args()
    main(args.rebuild, args.split)