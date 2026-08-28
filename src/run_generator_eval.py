"""
Usage:
    python -m src.run_generator_eval --prompt baseline --retriever rerank --limit 50
    python -m src.run_generator_eval --prompt cot_verification --retriever rerank --limit 50
    python -m src.run_generator_eval --prompt few_shot_analyst --retriever rerank --limit 50
    python -m src.run_generator_eval --prompt concise_extract --retriever rerank --limit 50
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from src.data_models.io import load_corpus_lookup, load_qa_examples
from src.retrieval.retriever_factory import RetrieverFactory
from src.generation.generator import Generator
from src.eval.generation_eval import evaluate_generator, print_report


def main(
    split: Literal["train", "dev", "test"],
    retriever_type: str,
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
    print(f"Retriever: {retriever_type} | Prompt Mode: {prompt_mode}")
    print()

    retriever = RetrieverFactory.create(
        retriever_type=retriever_type,
        corpus_lookup=corpus_lookup,
    )

    generator = Generator()

    cache_dir = Path("analysis/results/gen_eval")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{split}_{retriever_type}_{prompt_mode}.json"

    report = evaluate_generator(
        retriever=retriever,
        generator=generator,
        qa_examples=qa_examples,
        prompt_mode=prompt_mode,
        cache_path=str(cache_path),
        top_k=5,
    )

    print_report(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "dev", "test"],
    )

    parser.add_argument(
        "--retriever",
        default="rerank",
        choices=["dense", "bm25", "hybrid", "rerank"],
        help="Retriever type for candidate context generation.",
    )

    parser.add_argument(
        "--prompt",
        default="baseline",
        choices=[
            "baseline",
            "structured",
            "evidence",
            "cot_verification",
            "few_shot_analyst",
            "concise_extract",
            "rerank_aware",
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
        split=args.split,
        retriever_type=args.retriever,
        prompt_mode=args.prompt,
        limit=args.limit,
    )