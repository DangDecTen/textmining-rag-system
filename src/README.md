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
