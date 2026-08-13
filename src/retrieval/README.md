# Retrieval

## The interface

Every retriever implements `Retriever` (`base.py`):

```python
class Retriever(ABC):
    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Returns list[RetrievalResult], ranked best-first."""
```

That's the entire contract. Anything above it — the CLI, the API, the eval
scripts — only ever calls `.search()`, so a new implementation is a drop-in
replacement as long as it honors this signature.

## What exists today

| Name (registry key) | Class | Backing index | Notes |
|---|---|---|---|
| `bm25` | `BM25Retriever` (`bm25_retriever.py`) | `BM25Index` (`src/indexing/bm25_index.py`, `bm25s`) | lowercase + `[a-z0-9]+` tokenization, no stopwords/stemming |
| `dense` | `DenseRetriever` (`dense_retriever.py`) | `DenseIndex` (`src/indexing/dense_index.py`, FAISS `IndexFlatIP`) | `BAAI/bge-small-en-v1.5` via `sentence-transformers` |
| `hybrid` | `HybridRetriever` (`hybrid_retriever.py`) | composes `dense` + `bm25` retrievers (no index of its own) | fuses both rankings via RRF by default, or min-max normalized weighted sum (`use_rrf=False`); see below |

Design notes for bm25/dense (tokenization choices, embedding/pooling
details, why `IndexFlatIP` is the right call at this corpus size) are in
the root `src/README.md`.

## Lexical Retrieval (BM25)

Uses [`bm25s`](https://github.com/xhluca/bm25s) for indexing and retrieval.

```bash
# Guide
python -m src.indexing.build_index --help

# Build the index (skips if data/index/bm25 already has one)
python -m src.indexing.build_index --retriever bm25

# Force a rebuild, e.g. after changing a hyperparameter
python -m src.indexing.build_index --retriever bm25 --bm25-k1 1.2 --bm25-b 0.6 --rebuild
```

Then evaluate on a split with `evaluation.retrieval.run_eval` (a separate
package — see its README):

```bash
python -m evaluation.retrieval.run_eval --retriever bm25 --split dev
```

|Index & Retriever|Indexing|QA Examples|Metrics|Hyperparameters|
|-|-|-|-|-|
|`bm25s`|~15 min|2,533 (dev)|mrr: 0.724<br>recall@1: 0.639<br>recall@5: 0.831<br>recall@10: 0.886|method: lucene<br>k1: 1.5<br>b: 0.75|

## Dense Retrieval (FAISS)

Uses [Sentence Transformers](https://sbert.net/index.html) to embed
documents and questions, and
[`facebookresearch/faiss`](https://github.com/facebookresearch/faiss/wiki/)
(`IndexFlatIP`, exact search — appropriate at this corpus size) for the
index itself.

```bash
# Optional, set a HF_TOKEN in .env to enable faster model downloads.

# Build the index (skips if data/index/dense already has one; this embeds
# every document, so expect it to take a while on CPU)
python -m src.indexing.build_index --retriever dense

# Force a rebuild with a different model
python -m src.indexing.build_index --retriever dense --dense-model-name BAAI/bge-base-en-v1.5 --rebuild
```

Then evaluate the same way as bm25 above (`--retriever dense`).

|Index & Retriever|Indexing|QA Examples|Metrics|Hyperparameters|
|-|-|-|-|-|
|`IndexFlatIP`|~32 min|2,533 (dev)|mrr: 0.847<br>recall@1: 0.797<br>recall@5: 0.914<br>recall@10: 0.940|model: `BAAI/bge-small-en-v1.5`<br>batch: 64|

Index directories, model names, and hyperparameters all live in
`src/config.py` — see [Configuration](#configuration) to change the
persistent defaults, or pass `--bm25-*`/`--dense-*` flags to
`build_index.py` to override just one build without touching `.env`. See
`src/indexing/README.md` for the full flag reference and design notes.

### Hybrid (`hybrid_retriever.py`)

Composes an already-built `DenseRetriever` and `BM25Retriever` rather than
loading its own index. For each query it pulls `max(top_k * 3, 50)`
candidates from each side, then fuses:

- **RRF (default, `use_rrf=True`)**: `score = alpha / (k + rank_dense) +
  (1 - alpha) / (k + rank_bm25)`, summed over docs appearing in either
  list. `rrf_k` is the RRF constant (default 60, the usual literature
  value); `alpha` weights dense vs. bm25 (default 0.5, i.e. equal weight).
- **Weighted sum (`use_rrf=False`)**: min-max normalizes each retriever's
  raw scores to `[0, 1]` independently, then combines as
  `alpha * dense_norm + (1 - alpha) * bm25_norm`. Sensitive to score
  distribution outliers in a way RRF isn't -- RRF only uses rank, not raw
  score, so it's the safer default across different corpora/queries.

Config knobs (`src/config.py`, overridable via `.env`): `hybrid_alpha`,
`hybrid_rrf_k`, `hybrid_use_rrf`.

Note the constructor's fusion formula weights *both* terms by `alpha`
(dense) / `1 - alpha` (bm25) even in RRF mode -- this is a deliberate
variant on "textbook" RRF (which is unweighted, `1/(k+rank)` for each
list) added so relative trust in dense vs. lexical can be tuned per-corpus
without switching fusion strategies entirely.

## The registry (`registry.py`)

```python
@register_retriever("bm25")
class BM25Retriever(Retriever):
    ...
```

`register_retriever` is a class decorator that adds the class to a
module-level dict under that name, checked at import time. `build_retriever(name, **kwargs)`
looks it up and instantiates it. `available_retrievers()` lists every
registered name — that's what powers `GET /` in the API and the Streamlit
dropdown.

**Why a registry instead of an `if/elif` factory function:** the previous
version of this codebase had exactly that — a `RetrieverFactory` with a
hardcoded `if retriever_type == "dense": ...` branch. It imported a module
that had since been renamed, and called a constructor with a signature that
no longer matched the real class. Nothing caught this until it was actually
run, because the factory lived in a different file from the classes it
was supposed to know about, and nobody kept them in sync by hand.

With a registry, the retriever class registers *itself* at import time —
there's no separate list of "known retrievers" to fall out of date, because
the list is the registry, built directly from the classes that actually
exist.

The registry only answers "what retrievers are available and how do I
construct one given its dependencies" — it does **not** know how to *load*
a BM25 or FAISS index from disk (that needs different arguments per index
type). That part lives in `get_retriever()` in `src/factory.py`, one level
up, which is where an index directory becomes an actual loaded `Index`
object before being handed to the registry.

## Adding a new retriever

Say you want a retriever that merges BM25 and dense results (reciprocal
rank fusion), or one backed by a hosted vector DB (Pinecone, Weaviate,
Qdrant, ...).

1. **Implement the class.** New file, e.g. `src/retrieval/hybrid_retriever.py`:

   ```python
   from src.data_models.data_models import RetrievalResult
   from src.retrieval.base import Retriever
   from src.retrieval.registry import register_retriever

   @register_retriever("hybrid")
   class HybridRetriever(Retriever):
       def __init__(self, bm25_retriever: Retriever, dense_retriever: Retriever, k: int = 60):
           self.bm25_retriever = bm25_retriever
           self.dense_retriever = dense_retriever
           self.k = k  # RRF constant

       def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
           # e.g. reciprocal rank fusion over both retrievers' results
           ...
   ```

2. **Register it for import.** Add one line to the "side-effect imports"
   block near the top of `src/factory.py`:

   ```python
   import src.retrieval.hybrid_retriever  # noqa: F401
   ```

   (This is what actually runs the `@register_retriever("hybrid")`
   decorator — a class decorator only fires when Python imports the module
   it's defined in.)

3. **Teach the factory how to build one.** In `get_retriever()` in
   `src/factory.py`, add a branch:

   ```python
   elif name == "hybrid":
       return build_retriever(
           name,
           bm25_retriever=get_retriever("bm25"),
           dense_retriever=get_retriever("dense"),
       )
   ```

4. **(Optional) add config.** If your retriever needs its own settings
   (an index path, an API key, a hyperparameter), add fields to
   `src/config.py::Settings` rather than hardcoding them, so they're
   overridable via `.env` like everything else.

That's the whole checklist. Once done: `python run_rag.py --retriever hybrid`,
`POST /query {"retriever": "hybrid", ...}`, and the Streamlit dropdown all
work without any further changes, because they all go through
`available_retrievers()` / `get_retriever()` rather than hardcoding a list
of names anywhere.

## Evaluating a retriever

Any registered retriever can be scored with `src/eval/retrieval_eval.py`
(recall@k, MRR, overall and per-`source`). `src/run_bm25.py` and
`src/run_dense.py` are the existing examples — a new retriever's own
`run_*.py` script would look the same, just swapping which retriever/index
gets built.
