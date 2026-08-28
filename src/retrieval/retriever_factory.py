from pathlib import Path

from src.data_models.io import load_corpus_lookup
from src.indexing.bm25_index import BM25Index
from src.indexing.dense_index import DenseIndex
from src.retrieval.base import Retriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever

DEFAULT_DENSE_INDEX_DIR = Path("data/index/dense_bge_small")
DEFAULT_BM25_INDEX_DIR = Path("data/index/bm25_k1_b25")


def _resolve_index_dir(path: Path | str, fallbacks: list[str]) -> Path:
    p = Path(path)
    if p.exists():
        return p
    for fb in fallbacks:
        fb_p = Path(fb)
        if fb_p.exists():
            return fb_p
    return p


class RetrieverFactory:
    @staticmethod
    def create(
        retriever_type: str,
        corpus_lookup: dict | None = None,
        dense_dir: Path | str = DEFAULT_DENSE_INDEX_DIR,
        bm25_dir: Path | str = DEFAULT_BM25_INDEX_DIR,
        alpha: float = 0.5,
    ) -> Retriever:
        retriever_type = retriever_type.lower()
        dense_dir = _resolve_index_dir(dense_dir, ["data/index/dense_bge_small", "data/index/dense_bge_base", "data/index/dense"])
        bm25_dir = _resolve_index_dir(bm25_dir, ["data/index/bm25_k1_b25", "data/index/bm25"])
        if corpus_lookup is None:
            corpus_lookup = load_corpus_lookup()

        if retriever_type == "dense":
            dense_index = DenseIndex.load(str(dense_dir))
            return DenseRetriever(index=dense_index, corpus_lookup=corpus_lookup)

        elif retriever_type in ("bm25", "lexical"):
            bm25_index = BM25Index.load(str(bm25_dir))
            return BM25Retriever(index=bm25_index, corpus_lookup=corpus_lookup)

        elif retriever_type in ("hybrid", "hybrid_rrf"):
            dense_index = DenseIndex.load(str(dense_dir))
            dense_retriever = DenseRetriever(index=dense_index, corpus_lookup=corpus_lookup)
            bm25_index = BM25Index.load(str(bm25_dir))
            bm25_retriever = BM25Retriever(index=bm25_index, corpus_lookup=corpus_lookup)
            return HybridRetriever(
                dense_retriever=dense_retriever,
                bm25_retriever=bm25_retriever,
                alpha=alpha,
            )

        elif retriever_type in ("cross_encoder", "rerank", "reranker", "hybrid_rerank"):
            from src.retrieval.cross_encoder_retriever import CrossEncoderRetriever
            base_retriever = RetrieverFactory.create("hybrid", corpus_lookup=corpus_lookup, dense_dir=dense_dir, bm25_dir=bm25_dir, alpha=alpha)
            return CrossEncoderRetriever(
                base_retriever=base_retriever,
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                candidate_k=50,
                corpus_lookup=corpus_lookup,
            )

        raise ValueError(f"Unknown retriever type: {retriever_type}. Expected 'dense', 'bm25', 'hybrid', or 'rerank'.")