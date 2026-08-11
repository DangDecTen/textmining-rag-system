"""
Interactive CLI for the full RAG pipeline (Retriever + Generator).

Usage:
    python run_rag.py                                  # defaults from src/config.py
    python run_rag.py --retriever dense --generator qwen

Retriever/generator names come from the registries in
src/retrieval/registry.py and src/generation/registry.py -- run with
--list to see what's currently registered.
"""
from __future__ import annotations

import argparse

from src.config import settings
from src.factory import available_generators, available_retrievers, get_pipeline
from src.pipeline import Pipeline


def run_query(question: str, top_k: int, pipeline: Pipeline) -> None:
    answer, generation_result = pipeline.answer_with_debug(question, top_k=top_k)

    print("\n----------")
    print(f"Q: {question}")
    print(f"A: {answer.text}")

    if answer.citations:
        print("\n----------")
        print("Retrieved contexts...\n")
        for c in answer.citations:
            print(f"[{c.subject_id} - {c.subject_name}] Facts taken from {c.field + ', ' if c.field else ''}{c.source}.")
            print(f"- Related: {c.relation_name if c.relation_name else 'None'}")
            print(f"- References: {c.references}")
            print(f"- URL: {c.url}\n")

    print(
        f"[abstained={answer.abstained} | latency={generation_result.latency_ms:.0f}ms | "
        f"prompt_tokens={generation_result.prompt_tokens} completion_tokens={generation_result.completion_tokens}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retriever", default=settings.default_retriever, help="Retriever name (see --list)")
    parser.add_argument("--generator", default=settings.default_generator, help="Generator name (see --list)")
    parser.add_argument("--top-k", type=int, default=settings.default_top_k)
    parser.add_argument("--question", default=None, help="Skip the interactive prompt and ask this directly")
    parser.add_argument("--list", action="store_true", help="List available retrievers/generators and exit")
    args = parser.parse_args()

    if args.list:
        print(f"Retrievers: {available_retrievers()}")
        print(f"Generators: {available_generators()}")
        return

    question = args.question or input("Enter question: ").strip()
    if not question:
        print("question cannot be empty")
        return

    print(f"Loading retriever ({args.retriever})...")
    print(f"Loading generator ({args.generator})...")
    pipeline = get_pipeline(retriever_name=args.retriever, generator_name=args.generator)

    run_query(question, args.top_k, pipeline)


if __name__ == "__main__":
    main()
