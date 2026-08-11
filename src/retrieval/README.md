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

Design notes for both (tokenization choices, embedding/pooling details,
why `IndexFlatIP` is the right call at this corpus size) are in the root
`src/README.md`.

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
