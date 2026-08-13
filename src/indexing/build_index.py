"""
Build (or rebuild) the on-disk BM25 and/or dense index from the processed
corpus.

This is the one place that turns `corpus.jsonl` into the artifacts
`src.factory.get_retriever()` loads at query time -- it just drives the
`build()` / `save()` methods `BM25Index` / `DenseIndex` already implement
(see `src/indexing/base.py`, `bm25_index.py`, `dense_index.py`). No new
indexing logic lives here; this is orchestration + a CLI over hyperparameters
that already exist as constructor args on those two classes.

Usage:
    # Build both, using settings.py defaults (bm25_method/k1/b, dense_model_name/batch_size)
    python -m src.indexing.build_index

    # Just one
    python -m src.indexing.build_index --retriever bm25
    python -m src.indexing.build_index --retriever dense

    # Override hyperparameters for this run only (doesn't touch settings.py/.env)
    python -m src.indexing.build_index --retriever bm25 --bm25-k1 1.2 --bm25-b 0.6
    python -m src.indexing.build_index --retriever dense --dense-model-name BAAI/bge-base-en-v1.5 --dense-batch-size 32

    # Build into a separate directory to compare against the current index
    # without overwriting it (e.g. before deciding to switch defaults)
    python -m src.indexing.build_index --retriever bm25 --bm25-output-dir data/index/bm25_v2

    # An index directory that already has files in it is skipped by default
    # (building, especially dense, is expensive -- minutes, not seconds).
    # Force a rebuild in place:
    python -m src.indexing.build_index --retriever all --rebuild

Once built, evaluate with `evaluation.retrieval.run_eval` (or point
`--bm25-output-dir`/`--dense-output-dir` here and `BM25_INDEX_DIR`/
`DENSE_INDEX_DIR` in `.env` at the same path to make it the default the
rest of the app uses).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from src.config import settings
from src.data_models.data_models import Document
from src.data_models.io import load_corpus_lookup
from src.indexing.bm25_index import BM25Index
from src.indexing.dense_index import DenseIndex


def _index_dir_has_content(path: str) -> bool:
    """True if `path` exists and already holds saved index files.

    Used to decide whether to skip a build by default -- an empty or
    missing directory is not "already built", even though it may exist
    (e.g. created by an earlier failed run).
    """
    p = Path(path)
    return p.exists() and any(p.iterdir())


def build_bm25(
    corpus: dict[str, Document],
    output_dir: str,
    method: str,
    k1: float,
    b: float,
    rebuild: bool,
) -> None:
    if _index_dir_has_content(output_dir) and not rebuild:
        print(f"[bm25]  {output_dir} already has an index -- skipping (pass --rebuild to force).")
        return

    print(f"[bm25]  Building (method={method}, k1={k1}, b={b}) over {len(corpus)} documents...")
    start = time.perf_counter()
    index = BM25Index(method=method, k1=k1, b=b)
    index.build(list(corpus.values()))
    index.save(output_dir)
    print(f"[bm25]  Saved to {output_dir} in {time.perf_counter() - start:.1f}s")


def build_dense(
    corpus: dict[str, Document],
    output_dir: str,
    model_name: str,
    batch_size: int,
    rebuild: bool,
) -> None:
    if _index_dir_has_content(output_dir) and not rebuild:
        print(f"[dense] {output_dir} already has an index -- skipping (pass --rebuild to force).")
        return

    print(f"[dense] Building (model={model_name}, batch_size={batch_size}) over {len(corpus)} documents...")
    print("[dense] This embeds every document -- expect this to take a while on CPU.")
    start = time.perf_counter()
    index = DenseIndex(model_name=model_name, batch_size=batch_size)
    index.build(list(corpus.values()))
    index.save(output_dir)
    print(f"[dense] Saved to {output_dir} in {time.perf_counter() - start:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retriever", choices=["bm25", "dense", "all"], default="all")
    parser.add_argument("--corpus-path", default=None, help="Defaults to settings.corpus_path")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild even if an index already exists on disk")

    # BM25 hyperparameters. Each defaults to None here so "not passed" is
    # distinguishable from "explicitly passed the same value settings.py
    # already has" -- the actual default is resolved from `settings` in
    # main(), so a bare `--bm25-k1 0` (a legitimate value) doesn't get
    # silently discarded by an `arg or default` check.
    parser.add_argument("--bm25-output-dir", default=None, help="Defaults to settings.bm25_index_dir")
    parser.add_argument("--bm25-method", default=None, help="Defaults to settings.bm25_method")
    parser.add_argument("--bm25-k1", type=float, default=None, help="Defaults to settings.bm25_k1")
    parser.add_argument("--bm25-b", type=float, default=None, help="Defaults to settings.bm25_b")

    # Dense hyperparameters
    parser.add_argument("--dense-output-dir", default=None, help="Defaults to settings.dense_index_dir")
    parser.add_argument("--dense-model-name", default=None, help="Defaults to settings.dense_model_name")
    parser.add_argument("--dense-batch-size", type=int, default=None, help="Defaults to settings.dense_batch_size")

    args = parser.parse_args()

    corpus = load_corpus_lookup(args.corpus_path)
    print(f"Loaded {len(corpus)} documents from {args.corpus_path or settings.corpus_path}\n")

    def resolved(value: Any, default: Any) -> Any:
        return value if value is not None else default

    if args.retriever in ("bm25", "all"):
        build_bm25(
            corpus,
            output_dir=resolved(args.bm25_output_dir, settings.bm25_index_dir),
            method=resolved(args.bm25_method, settings.bm25_method),
            k1=resolved(args.bm25_k1, settings.bm25_k1),
            b=resolved(args.bm25_b, settings.bm25_b),
            rebuild=args.rebuild,
        )

    if args.retriever in ("dense", "all"):
        build_dense(
            corpus,
            output_dir=resolved(args.dense_output_dir, settings.dense_index_dir),
            model_name=resolved(args.dense_model_name, settings.dense_model_name),
            batch_size=resolved(args.dense_batch_size, settings.dense_batch_size),
            rebuild=args.rebuild,
        )


if __name__ == "__main__":
    main()
