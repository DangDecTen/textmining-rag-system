import argparse
from src.data_models.io import load_corpus_lookup
from src.pipeline import Pipeline
from src.retrieval.base import Retriever
from src.retrieval.retriever_factory import RetrieverFactory

from src.generation.base import Generator
from src.generation.llama_generator import LlamaGenerator


def get_retriever(retriever_type: str = "hybrid") -> Retriever:
    corpus_lookup = load_corpus_lookup()
    print(f"Loading retriever ({retriever_type})...")
    return RetrieverFactory.create(retriever_type, corpus_lookup=corpus_lookup)


def get_generator() -> Generator:
    print("Loading generator (llama-3.3-70b-versatile)...")
    return LlamaGenerator()


def my_app(question: str, top_k: int, pipeline: Pipeline):
    answer, generation_result = pipeline.answer_with_debug(question, top_k=top_k)

    print("\n----------")
    print(f"Q: {question}")
    print(f"A: {answer.text}")

    if answer.citations:
        print("\n----------")
        print("Retrieved contexts...\n")
        for c in answer.citations:
            field_str = f"{c.field}, " if c.field else ""
            print(f"[{c.subject_id} - {c.subject_name}] Facts taken from {field_str}{c.source}.")
            print(f"- Related: {c.relation_name if c.relation_name else 'None'}")
            print(f"- References: {c.references}")
            print(f"- URL: {c.url}\n")

    print(
        f"[abstained={answer.abstained} | latency={generation_result.latency_ms:.0f}ms | "
        f"prompt_tokens={generation_result.prompt_tokens} completion_tokens={generation_result.completion_tokens}]"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", default="hybrid", choices=["dense", "bm25", "hybrid"])
    args = parser.parse_args()

    question = input("Enter question: ").strip()
    top_k_str = input("Enter k (default 5): ").strip()
    top_k = int(top_k_str) if top_k_str else 5

    if not question:
        print("question cannot be empty")
        return

    retriever = get_retriever(args.retriever)
    generator = get_generator()
    rag_pipeline = Pipeline(retriever, generator)
    my_app(question, top_k, rag_pipeline)


if __name__ == "__main__":
    main()