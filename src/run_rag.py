from pathlib import Path
from src.data_models.io import load_corpus_lookup
from src.indexing.bm25_index import BM25Index
from src.pipeline import Pipeline

from src.retrieval.base import Retriever
from src.retrieval.bm25_retriever import BM25Retriever

from src.generation.base import Generator
from src.generation.qwen_generator import QwenGenerator
from src.generation.llama_generator import LlamaGenerator


INDEX_DIR = "data/index/bm25"


def get_index() -> BM25Index: 
    index_dir_p = Path(INDEX_DIR)    
    index = BM25Index.load(str(index_dir_p))
    return index


def get_retriever() -> Retriever:
    corpus_lookup = load_corpus_lookup()

    print(f"Loading retriever (bm25) at {INDEX_DIR}...")
    index = get_index()
    retriever = BM25Retriever(index=index, corpus_lookup=corpus_lookup)
    return retriever


def get_generator() -> Generator:
    print("Loading generator (llama-3.3-70b-versatile)...")
    generator = LlamaGenerator()
    return generator


def my_app(question: str, top_k: int, pipeline: Pipeline):
    answer, generation_result = pipeline.answer_with_debug(question, top_k=top_k)
 
    print("\n----------")
    print(f"Q: {question}")
    print(f"A: {answer.text}")

    if answer.citations:
        print("\n----------")
        print("Retrieved contexts...\n")
        for c in answer.citations:
            print(f"[{c.subject_id} - {c.subject_name}] Facts taken from {c.field + ", " if c.field else ""}{c.source}.")
            print(f"- Related: {c.relation_name if c.relation_name else "None"}")
            print(f"- References: {c.references}")
            print(f"- URL: {c.url}\n")

    print(
        f"[abstained={answer.abstained} | latency={generation_result.latency_ms:.0f}ms | "
        f"prompt_tokens={generation_result.prompt_tokens} completion_tokens={generation_result.completion_tokens}]"
    )


def main():
    question = input("Enter question: ").strip()
    top_k = int(input("Enter k: ").strip())
    if not question:
        print("question cannot be empty")
        return
    if not top_k:
        print("invalid k")
        return

    retriever = get_retriever()
    generator = get_generator()
    rag_pipeline = Pipeline(retriever, generator)
    my_app(question, top_k, rag_pipeline)


if __name__ == "__main__":
    main()