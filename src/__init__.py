"""
Textmining RAG System -- library package.

See the root README.md for an overview, and src/README.md for retrieval
design notes. Key entry points for reuse in other code:

    from src.factory import get_pipeline
    pipeline = get_pipeline(retriever_name="bm25", generator_name="llama")
    answer, debug = pipeline.answer_with_debug("some question", top_k=5)
"""
