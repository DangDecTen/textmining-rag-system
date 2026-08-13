"""
Usage:

python -m src.run_generator_eval --prompt baseline
python -m src.run_generator_eval --prompt structured
python -m src.run_generator_eval --prompt evidence

python -m src.run_generator_eval --prompt evidence --limit 200
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from src.data_models.io import load_corpus_lookup, load_qa_examples
from src.data_models.io import Document
from src.indexing.dense_index import DenseIndex
from src.retrieval.dense_retriever import DenseRetriever
from src.generation.generator import Generator
from src.eval.generation_eval import evaluate_generator, print_report

INDEX_DIR = "data/index/dense"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64


def build_index(corpus: list[Document], rebuild: bool = False) -> DenseIndex:

    index_dir = Path(INDEX_DIR)

    if rebuild or not index_dir.exists():

        print(f"Building dense index with {MODEL_NAME}...")

        index = DenseIndex(
            model_name=MODEL_NAME,
            batch_size=BATCH_SIZE,
        )

        index.build(corpus)
        index.save(str(index_dir))

        print(f"Saved index to {index_dir}/")

    else:

        print(f"Loading existing index from {index_dir}/")

        index = DenseIndex.load(str(index_dir))

    return index


def main(
    rebuild_index: bool,
    split: Literal["train", "dev", "test"],
    prompt_mode: str,
    limit: int | None,
):

    corpus_lookup = load_corpus_lookup()

    qa_examples = load_qa_examples(split=split)

    if limit is not None:
        qa_examples = qa_examples[:limit]

    print(
        f"Loaded {len(corpus_lookup)} documents, "
        f"{len(qa_examples)} QA examples ({split})"
    )

    print()

    index = build_index(
        corpus=list(corpus_lookup.values()),
        rebuild=rebuild_index,
    )

    retriever = DenseRetriever(
        index=index,
        corpus_lookup=corpus_lookup,
    )

    generator = Generator()

    cache_path = (
        f"src/eval/"
        f"{split}_{prompt_mode}.json"
    )

    report = evaluate_generator(
        retriever=retriever,
        generator=generator,
        qa_examples=qa_examples,
        prompt_mode=prompt_mode,
        cache_path=cache_path,
        top_k=5,
    )

    print_report(report)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild dense index.",
    )

    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "dev", "test"],
    )

    parser.add_argument(
        "--prompt",
        default="baseline",
        choices=[
            "baseline",
            "structured",
            "evidence",
        ],
        help="Prompt template to evaluate.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N QA samples.",
    )

    args = parser.parse_args()

    main(
        rebuild_index=args.rebuild,
        split=args.split,
        prompt_mode=args.prompt,
        limit=args.limit,
    )