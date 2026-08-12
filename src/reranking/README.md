# Reranking

## The interface

Every reranker implements `Reranker` (`base.py`):

```python
class Reranker(ABC):
    def rerank(self, query: str, results: list[RetrievalResult], top_k: int | None = None) -> list[RetrievalResult]:
        """Re-scores and re-sorts `results` for `query`, best-first."""
```

Unlike a `Retriever`, a `Reranker` doesn't search the corpus -- it takes
candidates a retriever already found and reorders them. Nothing above it
(`Pipeline`, the API, eval scripts) needs to know *how*; they only ever
call `.rerank()`.

## Why this exists / where it sits

bm25 and dense retrieval both score query and document **independently**
(a document's BM25 stats or embedding can be precomputed once, offline,
for the whole corpus). That's what makes them fast enough to search 17.7k
documents per query. A cross-encoder scores a query and a document
**jointly** -- far more accurate at judging relevance, but too slow to run
over an entire corpus, since every document needs its own forward pass at
query time.

The standard fix, and what this module does: use a cheap retriever
(bm25 / dense / hybrid) to pull a wide-ish candidate set, then use the
expensive-but-accurate reranker only on those candidates to pick the final
top-k. `Pipeline.answer_with_debug` does exactly this:

```
Retriever.search(top_k=rerank_candidate_k) -> Reranker.rerank(top_k=top_k) -> Generator.generate()
```

`rerank_candidate_k` (`src/config.py`, default 50) controls how wide that
candidate pool is. It should be meaningfully larger than the `top_k` you
actually want -- reranking a list that's already been cut down to `top_k`
by the retriever just reorders whatever the retriever happened to put in
that narrow set; it can't recover a good document the retriever ranked,
say, 30th.

## What exists today

| Name (registry key) | Class | Model | Notes |
|---|---|---|---|
| `cross_encoder` | `CrossEncoderReranker` (`cross_encoder_reranker.py`) | `BAAI/bge-reranker-base` via `sentence_transformers.CrossEncoder` | scores raw logits, order-only (don't compare against retriever scores) |

## The registry (`registry.py`)

Same pattern as `src/retrieval/registry.py` and `src/generation/registry.py`:
`register_reranker("name")` is a class decorator that self-installs the
class into a module-level dict at import time; `build_reranker(name,
**kwargs)` looks it up and instantiates it; `available_rerankers()` lists
every registered name.

## Config (`src/config.py`)

- `rerank_enabled: bool = True` -- whether `get_pipeline()` attaches a
  reranker at all. Set `RERANK_ENABLED=false` in `.env` to disable
  (e.g. for a fast eval run, or an environment where the cross-encoder
  model can't be downloaded).
- `default_reranker: str = "cross_encoder"`
- `cross_encoder_model_name`, `cross_encoder_batch_size`
- `rerank_candidate_k: int = 50` -- see above.

Reranking is currently **config-controlled only**, not a per-request
parameter (unlike `retriever` / `generator`, which the API and CLI both
accept by name). If per-request toggling is needed later, `get_pipeline()`
and `Pipeline.__init__` already take an optional `reranker`, so the change
is additive: add a `rerank: bool | None` field to the API request model
and thread it through, no structural change needed.

## Adding a new reranker

Same checklist as adding a retriever (`src/retrieval/README.md`), applied
to this module:

1. **Implement the class** in a new file, e.g. `src/reranking/llm_reranker.py`,
   subclassing `Reranker` and decorating with `@register_reranker("llm")`.
2. **Register it for import** -- add `import src.reranking.llm_reranker  #
   noqa: F401` to the side-effect imports block in `src/factory.py`.
3. **Teach `get_reranker()` in `src/factory.py` how to build one** -- add
   an `elif name == "llm": return build_reranker(name, ...)` branch,
   pulling any hyperparameters from `Settings`.
4. **(Optional) add config** -- new fields on `Settings` for any
   model name / API key / hyperparameter the new reranker needs.

Once done, `settings.default_reranker = "llm"` (or `DEFAULT_RERANKER=llm`
in `.env`) switches every entry point over, same as retrievers/generators.
