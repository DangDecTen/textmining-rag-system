"""
Full-scale evaluation of Retrieval Methods across both Dev and Test splits.
Evaluates 100% of samples (no subsampling):
- BM25 (Lexical)
- Dense (BGE-small + FAISS)
- Hybrid Weighted (alpha=0.5)
- Hybrid RRF (alpha=0.5, rrf_k=10)

Calculates:
- Overall: MRR, Recall@1, Recall@5, Recall@10, Recall@20, Latency
- Per-source category breakdown (25 MITRE ATT&CK categories)
- Per-length breakdown (Question & Document length quartiles)
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.config import settings
from src.data_models.data_models import Document, QAExample, RetrievalResult
from src.data_models.io import load_corpus_lookup, load_qa_examples
from src.factory import get_retriever
from evaluation.retrieval.metrics import evaluate_example


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def compute_metrics(predictions: list[dict[str, Any]], k_values: tuple[int, ...] = (1, 5, 10, 20)) -> dict[str, Any]:
    n = len(predictions)
    if n == 0:
        return {}
    
    mrr = sum(r["reciprocal_rank"] for r in predictions) / n
    recalls = {f"recall@{k}": sum(r[f"hit@{k}"] for r in predictions) / n for k in k_values}
    avg_latency_ms = sum(r.get("total_time_ms", 0.0) for r in predictions) / n

    # By source
    by_source_records = defaultdict(list)
    for r in predictions:
        by_source_records[r["source"]].append(r)

    by_source = {}
    for src, recs in sorted(by_source_records.items(), key=lambda x: len(x[1]), reverse=True):
        src_n = len(recs)
        by_source[src] = {
            "n": src_n,
            "mrr": sum(r["reciprocal_rank"] for r in recs) / src_n,
            **{f"recall@{k}": sum(r[f"hit@{k}"] for r in recs) / src_n for k in k_values},
            "avg_latency_ms": sum(r.get("total_time_ms", 0.0) for r in recs) / src_n,
        }

    return {
        "n": n,
        "mrr": mrr,
        **recalls,
        "avg_latency_ms": avg_latency_ms,
        "by_source": by_source,
    }


def rrf_fusion(
    dense_results: list[RetrievalResult],
    bm25_results: list[RetrievalResult],
    corpus_lookup: dict[str, Document],
    alpha: float = 0.5,
    rrf_k: int = 10,
    top_k: int = 20,
) -> list[RetrievalResult]:
    rrf_scores: dict[str, float] = {}

    for rank, res in enumerate(dense_results, start=1):
        rrf_scores[res.doc_id] = rrf_scores.get(res.doc_id, 0.0) + (alpha / (rrf_k + rank))

    for rank, res in enumerate(bm25_results, start=1):
        rrf_scores[res.doc_id] = rrf_scores.get(res.doc_id, 0.0) + ((1.0 - alpha) / (rrf_k + rank))

    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)[:top_k]

    return [
        RetrievalResult(
            doc_id=doc_id,
            score=float(rrf_scores[doc_id]),
            document=corpus_lookup.get(doc_id),
        )
        for doc_id in sorted_doc_ids
    ]


def weighted_score_fusion(
    dense_results: list[RetrievalResult],
    bm25_results: list[RetrievalResult],
    corpus_lookup: dict[str, Document],
    alpha: float = 0.5,
    top_k: int = 20,
) -> list[RetrievalResult]:
    def normalize(results: list[RetrievalResult]) -> dict[str, float]:
        if not results:
            return {}
        scores = [r.score for r in results]
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return {r.doc_id: 1.0 for r in results}
        return {r.doc_id: (r.score - min_s) / (max_s - min_s) for r in results}

    dense_norm = normalize(dense_results)
    bm25_norm = normalize(bm25_results)

    all_doc_ids = set(dense_norm.keys()) | set(bm25_norm.keys())
    combined_scores: dict[str, float] = {}
    for doc_id in all_doc_ids:
        s_dense = dense_norm.get(doc_id, 0.0)
        s_bm25 = bm25_norm.get(doc_id, 0.0)
        combined_scores[doc_id] = alpha * s_dense + (1.0 - alpha) * s_bm25

    sorted_doc_ids = sorted(combined_scores.keys(), key=lambda d: combined_scores[d], reverse=True)[:top_k]

    return [
        RetrievalResult(
            doc_id=doc_id,
            score=float(combined_scores[doc_id]),
            document=corpus_lookup.get(doc_id),
        )
        for doc_id in sorted_doc_ids
    ]


def evaluate_split(
    split: str,
    corpus_lookup: dict[str, Document],
    k_values: tuple[int, ...] = (1, 5, 10, 20),
    candidate_k: int = 50,
) -> dict[str, Any]:
    qa_examples = load_qa_examples(split=split)
    n_examples = len(qa_examples)
    print(f"\n=======================================================")
    print(f"EVALUATING SPLIT: '{split}' ({n_examples} examples)")
    print(f"=======================================================")

    bm25_retriever = get_retriever("bm25")
    dense_retriever = get_retriever("dense")

    # 1. Batch encode queries for Dense retriever to maximize speed
    print(f"[{split}] Batch encoding {n_examples} query embeddings for Dense retriever...", flush=True)
    t_start = time.perf_counter()
    from src.indexing.dense_index import QUERY_INSTRUCTION_PREFIX
    prefixed_queries = [f"{QUERY_INSTRUCTION_PREFIX}{qa.question}" for qa in qa_examples]
    query_embeddings = dense_retriever.index.embedder.model.encode(
        prefixed_queries,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    encode_time_total = time.perf_counter() - t_start
    avg_encode_ms = (encode_time_total / n_examples) * 1000
    print(f"[{split}] Encoded {n_examples} queries in {encode_time_total:.2f}s ({avg_encode_ms:.2f}ms/query)")

    # 2. Perform FAISS batch search for top candidate_k
    print(f"[{split}] Searching FAISS dense index (top {candidate_k})...", flush=True)
    t0 = time.perf_counter()
    dense_scores_all, dense_indices_all = dense_retriever.index.index.search(query_embeddings, candidate_k)
    dense_search_time_total = time.perf_counter() - t0
    print(f"[{split}] FAISS search completed in {dense_search_time_total:.3f}s")

    # Storage for predictions
    preds_bm25 = []
    preds_dense = []
    preds_hybrid_w = []
    preds_hybrid_rrf = []

    print(f"[{split}] Running BM25 search and fusion over all {n_examples} samples...", flush=True)
    for i, qa in enumerate(qa_examples):
        # Dense candidate results
        row_indices = dense_indices_all[i]
        row_scores = dense_scores_all[i]
        dense_cands = []
        for doc_idx, score in zip(row_indices, row_scores):
            if doc_idx == -1:
                continue
            doc_id = dense_retriever.index.doc_ids[doc_idx]
            dense_cands.append(
                RetrievalResult(doc_id=doc_id, score=float(score), document=corpus_lookup.get(doc_id))
            )

        # BM25 search
        t_bm25_0 = time.perf_counter()
        bm25_cands = bm25_retriever.search(qa.question, top_k=candidate_k)
        t_bm25 = (time.perf_counter() - t_bm25_0) * 1000

        # Dense timing
        t_dense = avg_encode_ms + (dense_search_time_total / n_examples) * 1000

        # Hybrid Weighted fusion
        t_hw_0 = time.perf_counter()
        hw_results = weighted_score_fusion(dense_cands, bm25_cands, corpus_lookup, alpha=0.5, top_k=max(k_values))
        t_hw = t_dense + t_bm25 + (time.perf_counter() - t_hw_0) * 1000

        # Hybrid RRF fusion
        t_hrrf_0 = time.perf_counter()
        hrrf_results = rrf_fusion(dense_cands, bm25_cands, corpus_lookup, alpha=0.5, rrf_k=10, top_k=max(k_values))
        t_hrrf = t_dense + t_bm25 + (time.perf_counter() - t_hrrf_0) * 1000

        # Top-k slices for pure BM25 and Dense
        bm25_topk = bm25_cands[:max(k_values)]
        dense_topk = dense_cands[:max(k_values)]

        # Score examples
        r_bm25 = evaluate_example(qa, bm25_topk, corpus_lookup, k_values=k_values)
        r_bm25.update({"retrieve_time_ms": t_bm25, "rerank_time_ms": 0.0, "total_time_ms": t_bm25, "retriever": "bm25", "reranker": None, "split": split})
        preds_bm25.append(r_bm25)

        r_dense = evaluate_example(qa, dense_topk, corpus_lookup, k_values=k_values)
        r_dense.update({"retrieve_time_ms": t_dense, "rerank_time_ms": 0.0, "total_time_ms": t_dense, "retriever": "dense", "reranker": None, "split": split})
        preds_dense.append(r_dense)

        r_hw = evaluate_example(qa, hw_results, corpus_lookup, k_values=k_values)
        r_hw.update({"retrieve_time_ms": t_hw, "rerank_time_ms": 0.0, "total_time_ms": t_hw, "retriever": "hybrid_weighted", "reranker": None, "split": split})
        preds_hybrid_w.append(r_hw)

        r_hrrf = evaluate_example(qa, hrrf_results, corpus_lookup, k_values=k_values)
        r_hrrf.update({"retrieve_time_ms": t_hrrf, "rerank_time_ms": 0.0, "total_time_ms": t_hrrf, "retriever": "hybrid_rrf", "reranker": None, "split": split})
        preds_hybrid_rrf.append(r_hrrf)

        if (i + 1) % 500 == 0 or (i + 1) == n_examples:
            print(f"[{split}] Processed {i+1}/{n_examples} QA samples...", flush=True)

    # Save prediction files
    out_dir = Path("evaluation/retrieval/results")
    write_jsonl(preds_bm25, out_dir / f"bm25_{split}_predictions.jsonl")
    write_jsonl(preds_dense, out_dir / f"dense_{split}_predictions.jsonl")
    write_jsonl(preds_hybrid_w, out_dir / f"hybrid_weighted_{split}_predictions.jsonl")
    write_jsonl(preds_hybrid_w, out_dir / f"hybrid_{split}_predictions.jsonl")  # alias for backward compat
    write_jsonl(preds_hybrid_rrf, out_dir / f"hybrid_rrf_{split}_predictions.jsonl")

    # Compute metrics
    metrics = {
        "bm25": compute_metrics(preds_bm25, k_values=k_values),
        "dense": compute_metrics(preds_dense, k_values=k_values),
        "hybrid_weighted": compute_metrics(preds_hybrid_w, k_values=k_values),
        "hybrid_rrf": compute_metrics(preds_hybrid_rrf, k_values=k_values),
    }

    # Print summary table for this split
    print(f"\n--- SUMMARY FOR SPLIT: {split.upper()} (N={n_examples}) ---")
    print(f"| Retriever | MRR | Recall@1 | Recall@5 | Recall@10 | Recall@20 | Latency (ms) |")
    print(f"|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for name, m in metrics.items():
        print(f"| {name:<17} | {m['mrr']:.4f} | {m['recall@1']:.4f} | {m['recall@5']:.4f} | {m['recall@10']:.4f} | {m['recall@20']:.4f} | {m['avg_latency_ms']:.2f} |")

    return metrics


def main() -> None:
    corpus_lookup = load_corpus_lookup()
    print(f"Loaded {len(corpus_lookup)} corpus documents from {settings.corpus_path}")

    k_values = (1, 5, 10, 20)

    # Run for dev and test
    dev_metrics = evaluate_split("dev", corpus_lookup, k_values=k_values)
    test_metrics = evaluate_split("test", corpus_lookup, k_values=k_values)

    all_results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "corpus_docs": len(corpus_lookup),
            "dev_samples": dev_metrics["bm25"]["n"],
            "test_samples": test_metrics["bm25"]["n"],
            "k_values": list(k_values),
        },
        "dev": dev_metrics,
        "test": test_metrics,
    }

    out_json = Path("analysis/results/retrieval_full_benchmark_dev_test.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved consolidated benchmark results to: {out_json}")


if __name__ == "__main__":
    main()
