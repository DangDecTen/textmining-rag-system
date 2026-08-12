> This file covers retrieval **design decisions** (tokenization, embedding
> choices, index type). For the `Retriever` interface, the registry pattern,
> and how to add a new retriever, see `src/retrieval/README.md`. For the
> generation side, see `src/generation/README.md`. For the end-to-end
> architecture and how everything is wired together, see the root
> `README.md`. Paths, model names, and hyperparameters referenced below are
> all defined once in `src/config.py`, not hardcoded per-script.

# Lexical Retrieval (BM25)

Design decisions:
- **Tokenization**: lowercase + `[a-z0-9]+` regex, no stopword removal, no
  stemming. Technique IDs like `T1055.001` split into `t1055`/`001` — fine
  for matching since the same function tokenizes queries and documents, but
  a candidate cybersecurity-specific improvement later (preserve IDs as
  single tokens) if eval shows ID-heavy questions underperforming.



# Dense retrieval (FAISS + bge-small-en-v1.5)

Design decisions:
- **Embedding model**: `BAAI/bge-small-en-v1.5` via `sentence-transformers`
  (chosen over raw `transformers` + manual pooling for lower bug risk).
- **CLS pooling + L2 normalize**: handled internally by sentence-transformers
  for this model — confirmed against the model card before writing any code.
- **Query instruction prefix**: `"Represent this sentence for searching
  relevant passages: "` is applied **explicitly** in `src/embedding/embedder.py`
  to queries only (never to documents), rather than relying on
  sentence-transformers' automatic `prompt_name="query"` mechanism — that
  only fires if the checkpoint's `config_sentence_transformers.json` defines
  the prompt, which isn't guaranteed. Explicit is safer. (BAAI's own docs
  note this prefix gives only a *slight* boost for v1.5 models specifically —
  so this is a minor lever, not correctness-critical, but free to get right.)
- **FAISS `IndexFlatIP`**: exact search, appropriate at ~17.7k documents —
  no approximate index (IVF/HNSW) needed at this scale.

# Hybrid retrieval (RRF fusion of BM25 + dense)

Design decisions:
- **Composes retrievers, not indexes.** `HybridRetriever` has no index of
  its own — it wraps an already-built `BM25Retriever` and `DenseRetriever`
  and fuses their rankings. This is why it's the one retriever `get_retriever()`
  in `src/factory.py` doesn't load an index for; see `src/retrieval/README.md`.
- **RRF by default, not raw score averaging.** BM25 scores and cosine
  similarity are on unrelated scales, so combining them directly would let
  whichever happens to have larger numbers dominate. Reciprocal Rank
  Fusion sidesteps this by fusing on *rank* rather than raw score. A
  min-max-normalized weighted-sum mode (`use_rrf=False`) exists as an
  alternative but is more sensitive to score-distribution outliers.
- **`alpha` weights dense vs. bm25** (default 0.5, i.e. equal trust) inside
  the RRF formula itself, rather than leaving RRF unweighted — lets
  relative trust in dense vs. lexical be tuned per-corpus without
  switching fusion strategies. Full formula in `src/retrieval/README.md`.

# Reranking (Cross-Encoder + bge-reranker-base)

Design decisions:
- **Model**: `BAAI/bge-reranker-base` via `sentence_transformers.CrossEncoder`
  — same library as dense retrieval's embedder, so no new dependency.
  Chose `-base` over `-v2-m3` for now: `-base` is meaningfully cheaper to
  run per query (it scores `rerank_candidate_k` pairs *every* query, unlike
  the dense embedder which only embeds one query string), and `-v2-m3`'s
  main advantage is multilingual support this corpus doesn't need. Revisit
  if eval shows accuracy is the bottleneck rather than latency.
- **Candidate pool, not full corpus**: a cross-encoder scores query+doc
  jointly, so unlike bm25/dense it can't precompute anything per-document
  offline — it's accurate but too slow to run over the whole corpus. It
  only ever reranks the retriever's top `rerank_candidate_k` (default 50),
  never searches the corpus itself. See `src/reranking/README.md`.
- **Score scale**: `CrossEncoder.predict()` returns raw relevance logits
  for this model (no sigmoid). Only used for sorting within one `rerank()`
  call — never compared against BM25 or cosine-similarity scores, which
  are on unrelated scales.
