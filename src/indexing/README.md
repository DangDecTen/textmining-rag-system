# Indexing

Turns `corpus.jsonl` (from `src/ingestion/`) into the on-disk artifacts
`src.retrieval` retrievers load at query time.

## The interface

Every index implements `Index` (`base.py`):

```python
class Index(ABC):
    def build(self, chunks: list[Document]) -> None: ...
    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> "Index": ...
```

Two implementations exist today:

| Class | File | Backing | Notes |
|---|---|---|---|
| `BM25Index` | `bm25_index.py` | `bm25s` | lowercase + `[a-z0-9]+` tokenization, no stopwords/stemming |
| `DenseIndex` | `dense_index.py` | FAISS `IndexFlatIP` + `sentence-transformers` | `BAAI/bge-small-en-v1.5` by default, exact search |

There's no `HybridIndex` — `HybridRetriever` (`src/retrieval/hybrid_retriever.py`)
composes an already-built `BM25Index`-backed and `DenseIndex`-backed
retriever rather than indexing anything itself, so there's nothing for
`build_index.py` to build for it.

Design rationale for the tokenization/embedding/FAISS choices above is in
`src/README.md`, not here — this file is about the *build process*, not
the retrieval-time design decisions.

## Building an index (`build_index.py`)

```bash
python -m src.indexing.build_index                    # both, using settings.py defaults
python -m src.indexing.build_index --retriever bm25
python -m src.indexing.build_index --retriever dense
python -m src.indexing.build_index --help              # full flag reference
```

Hyperparameters are tunable per-run via CLI flags, without touching
`src/config.py`/`.env`:

```bash
python -m src.indexing.build_index --retriever bm25 --bm25-k1 1.2 --bm25-b 0.6
python -m src.indexing.build_index --retriever dense --dense-model-name BAAI/bge-base-en-v1.5 --dense-batch-size 32
```

| Flag | Retriever | Maps to | Default |
|---|---|---|---|
| `--bm25-output-dir` | bm25 | `BM25Index` save path | `settings.bm25_index_dir` |
| `--bm25-method` | bm25 | `BM25Index(method=...)` | `settings.bm25_method` |
| `--bm25-k1` | bm25 | `BM25Index(k1=...)` | `settings.bm25_k1` |
| `--bm25-b` | bm25 | `BM25Index(b=...)` | `settings.bm25_b` |
| `--dense-output-dir` | dense | `DenseIndex` save path | `settings.dense_index_dir` |
| `--dense-model-name` | dense | `DenseIndex(model_name=...)` | `settings.dense_model_name` |
| `--dense-batch-size` | dense | `DenseIndex(batch_size=...)` | `settings.dense_batch_size` |
| `--corpus-path` | both | corpus to build from | `settings.corpus_path` |
| `--rebuild` | both | force rebuild even if the output dir already has files | off |

**Skips by default if the output directory already has files in it** —
building, especially dense (embeds every document), takes real time, so an
accidental re-run shouldn't silently redo minutes of work. Pass `--rebuild`
to force it, e.g. after changing a hyperparameter.

**Comparing hyperparameters without touching the live index**: point
`--bm25-output-dir`/`--dense-output-dir` at a scratch path, build there, and
evaluate both with `evaluation.retrieval.run_eval --output-dir ...` before
deciding whether to promote the new settings into `.env`:

```bash
python -m src.indexing.build_index --retriever bm25 --bm25-k1 1.2 --bm25-output-dir data/index/bm25_k1_1.2
```

`src.factory.get_retriever("bm25")` won't see this until `BM25_INDEX_DIR`
in `.env` is repointed at it (or you load it directly via
`BM25Index.load("data/index/bm25_k1_1.2")` for a one-off comparison).

## Once built

- `src.factory.get_retriever(name)` (used by `run_rag.py`, the API,
  `evaluation.retrieval.run_eval`) loads whatever's at
  `settings.bm25_index_dir` / `settings.dense_index_dir` on first use, then
  caches it (`lru_cache`) — restart the process to pick up a rebuilt index.
- See the root README's [Evaluation](../../README.md#evaluation) section
  and `evaluation/retrieval/README.md` for scoring what you built.

## Extending

**New index type** (e.g. a different vector store): implement `Index` in a
new file here, then add a `build_<name>()` function to `build_index.py`
following the `build_bm25`/`build_dense` pattern, plus a `--retriever
<name>` choice and any `--<name>-*` hyperparameter flags. If it backs a new
retriever too, see `src/retrieval/README.md` for wiring that half in.
