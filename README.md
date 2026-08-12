# Textmining RAG System

A RAG (Retrieval-Augmented Generation) system over
[AttackQA](https://arxiv.org/abs/2411.01073), a QA dataset built on the
MITRE ATT&CK cybersecurity knowledge base. Ask a question, get a short,
grounded answer with citations back to the ATT&CK documents it came from —
or an explicit "I don't know" if the answer isn't in the knowledge base.

## Table of Contents

- [How it fits together](#how-it-fits-together)
- [Quickstart](#quickstart)
- [Data Ingestion](#data-ingestion)
- [Indexing & Retrieval](#indexing--retrieval)
  - [Lexical Retrieval (BM25)](#lexical-retrieval-bm25)
  - [Dense Retrieval (FAISS)](#dense-retrieval-faiss)
  - [Hybrid Retrieval (BM25 + Dense)](#hybrid-retrieval-bm25--dense)
- [Reranking](#reranking)
- [Generation](#generation)
  - [Llama Generator](#llama-generator)
  - [Qwen Generator](#qwen-generator)
- [Evaluation](#evaluation)
- [Running the app (API + Streamlit)](#running-the-app-api--streamlit)
- [Configuration](#configuration)
- [Repository Structure](#repository-structure)
- [Extending the System](#extending-the-system)
- [Where to look for what](#where-to-look-for-what)
- [References](#references)

## How it fits together

**Offline: preparing the index.**

```
attackqa.parquet
        │  ingestion (src/ingestion/)
        ▼
corpus.jsonl + qa_{train,dev,test}.jsonl
        │  indexing (src/indexing/)
        ▼
Index on disk (data/index/bm25 or data/index/dense)
```

**Online: answering a question.**

```
question + top_k
        │
        ▼
Pipeline(Retriever, Generator, Reranker?)   <- src/pipeline.py
        │
        ├─ Retriever.search()          <- src/retrieval/  (bm25, dense, or hybrid)
        ├─ Reranker.rerank()           <- src/reranking/  (cross-encoder; on by
        │                                  default, config-controlled — see
        │                                  [Reranking](#reranking))
        ├─ Generator.generate()        <- src/generation/ (llama or qwen)
        └─ ResponseBuilder.build()     <- turns raw model output into a
                                           clean, citation-bearing Answer
        ▼
Answer (text, abstained?, citations)
```

When reranking is on (the default), the retriever is asked for a wider
candidate set than `top_k` (`rerank_candidate_k`, default 50) so the
reranker has a meaningful pool to narrow down from — see
[Reranking](#reranking).

Everything that turns a *name* ("bm25", "dense", "hybrid", "llama", "qwen")
into a working object goes through **one** module, `src/factory.py`, which
reads its defaults from **one** config module, `src/config.py`. The CLI
(`run_rag.py`), the FastAPI backend (`app/backend/api.py`), and the eval
scripts (`src/run_bm25.py`, `src/run_dense.py`) all call into the same
factory, so they can't drift out of sync with each other — see
[Extending the System](#extending-the-system) for why this matters.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # or your preferred env manager
pip install -e .            # installs the package + all dependencies (pyproject.toml)
cp .env.example .env        # fill in GROQ_API_KEY (and optionally HF_TOKEN)
```

Then follow [Data Ingestion](#data-ingestion) → [Indexing](#indexing--retrieval)
→ [Generation](#generation) once to build everything from scratch, or jump
straight to [Running the app](#running-the-app-api--streamlit) if the
indexes under `data/index/` already exist.

## Data Ingestion

Parses AttackQA into a corpus (`corpus.jsonl`) for indexing, and QA pairs
with train/dev/test splits (default 80/10/10) for evaluation.

```bash
python -m src.ingestion.run_ingestion \
    --input data/benchmark/attackqa.parquet \
    --output-dir data/processed
```

No chunking in the current implementation — AttackQA has already done that.
See `src/ingestion/README.md` for dedup/split design decisions, or explore
the [AttackQA dataset](https://huggingface.co/datasets/sambanovasystems/attackqa/blob/main/Getting%20Started%20with%20MITRE%20QA%20Dataset.ipynb)
directly.

## Indexing & Retrieval

Every retriever implements the same `Retriever` interface
(`src/retrieval/base.py`: `.search(query, top_k) -> list[RetrievalResult]`),
so they're interchangeable everywhere in the system — the CLI, the API, and
the eval scripts all just take a retriever *name*. See
`src/retrieval/README.md` for design notes and how to add a new one.

### Lexical Retrieval (BM25)

Uses [`bm25s`](https://github.com/xhluca/bm25s) for indexing and retrieval.

```bash
# Guide
python -m src.run_bm25 --help

# Build the index (first run) and evaluate on the dev split
python -m src.run_bm25 --split dev

# Force a rebuild
python -m src.run_bm25 --split dev --rebuild
```

|Index & Retriever|Indexing|QA Examples|Metrics|Hyperparameters|
|-|-|-|-|-|
|`bm25s`|~15 min|2,533 (dev)|mrr: 0.724<br>recall@1: 0.639<br>recall@5: 0.831<br>recall@10: 0.886|method: lucene<br>k1: 1.5<br>b: 0.75|

### Dense Retrieval (FAISS)

Uses [Sentence Transformers](https://sbert.net/index.html) to embed
documents and questions, and
[`facebookresearch/faiss`](https://github.com/facebookresearch/faiss/wiki/)
(`IndexFlatIP`, exact search — appropriate at this corpus size) for the
index itself.

```bash
# Optional, set a HF_TOKEN in .env to enable faster model downloads.

# Guide
python -m src.run_dense --help

# Build the index (first run) and evaluate on the dev split
python -m src.run_dense --split dev
```

|Index & Retriever|Indexing|QA Examples|Metrics|Hyperparameters|
|-|-|-|-|-|
|`IndexFlatIP`|~32 min|2,533 (dev)|mrr: 0.847<br>recall@1: 0.797<br>recall@5: 0.914<br>recall@10: 0.940|model: `BAAI/bge-small-en-v1.5`<br>batch: 64|

Index directories, model names, and hyperparameters all live in
`src/config.py` — see [Configuration](#configuration) to change them without
editing code.

### Hybrid Retrieval (BM25 + Dense)

Fuses BM25 and dense rankings via Reciprocal Rank Fusion (RRF) by default
(a min-max normalized weighted-sum mode is also available). Doesn't build
its own index — it composes an already-built `BM25Retriever` and
`DenseRetriever`, so both indexes above need to exist first.

```bash
python run_rag.py --retriever hybrid
```

|Index & Retriever|Indexing|QA Examples|Metrics|Hyperparameters|
|-|-|-|-|-|
|`Hybrid`|Reused|2,533 (dev)|mrr: 0.810<br>recall@1: 0.750<br>recall@5: 0.887<br>recall@10: 0.931|method: RRF<br>k: 60<br>alpha: 0.5|
|`Hybrid`|Reused|2,538 (test)|mrr: 0.804<br>recall@1: 0.746<br>recall@5: 0.881<br>recall@10: 0.920|method: RRF<br>k: 60<br>alpha: 0.5|

See `src/retrieval/README.md` for the fusion formulas and why RRF is the
safer default (BM25 and cosine-similarity scores are on unrelated scales,
so combining them directly would let whichever has larger numbers dominate;
RRF fuses on rank instead).

## Reranking

A cross-encoder pass between retrieval and generation. Unlike bm25/dense,
which score query and document independently (so document representations
can be precomputed offline), a cross-encoder scores query+document
*jointly* — more accurate, but too slow to run over the whole corpus. So it
only reranks the candidates a retriever already narrowed down, not the
corpus itself: `Retriever.search(top_k=rerank_candidate_k)` →
`Reranker.rerank(top_k=requested_top_k)`.

**On by default**, config-controlled (`RERANK_ENABLED` in `.env`) rather
than a per-request field like `retriever`/`generator` — every `run_rag.py`
run and every `POST /query` gets reranked results unless you turn it off.
`POST /retrieve` (retrieval-only debugging endpoint) always bypasses it —
see `app/README.md`.

```bash
# Disable for a run, e.g. to compare against reranked output or for a faster eval
RERANK_ENABLED=false python run_rag.py
```

|Reranker|Model|Notes|
|-|-|-|
|`cross_encoder`|`BAAI/bge-reranker-base` via `sentence_transformers.CrossEncoder`|scores raw logits — order-only, don't compare against retriever scores|

Same registry pattern as retrieval/generation (`Reranker` interface,
`@register_reranker`, self-installing). See `src/reranking/README.md` for
design rationale and how to add a new reranker.

## Generation

Takes `list[RetrievalResult]` from the retrieval stage, builds a bounded
context window, and forces the model to answer in strict JSON
(`{"answer": ..., "found": bool}`), which is then parsed into a
`GenerationResult` and turned into a citation-bearing `Answer`. If the model
can't find the answer in context, it abstains explicitly rather than
guessing. See `src/generation/README.md` for the full design rationale and
how to add a new generator.

```bash
# Interactive CLI: pick a retriever + generator, ask a question
python run_rag.py
python run_rag.py --retriever dense --generator qwen
python run_rag.py --list          # show every registered retriever/generator
```

### Llama Generator

Groq API, [`llama-3.3-70b-versatile`](https://console.groq.com/docs/model/llama-3.3-70b-versatile).

```dotenv
# Set in .env to use this generator.
GROQ_API_KEY=<YOUR_API_KEY>
```

Limits: 131,072 context tokens, 32,768 max output tokens, ~1s latency.

### Qwen Generator

Local, [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
via raw `transformers`.

```dotenv
# Optional, set in .env to enable faster model downloads.
HF_TOKEN=<YOUR_API_KEY>
```

Limits: 128k context tokens, 8k max output tokens, ~15 min latency on CPU.

## Evaluation

Retrieval evaluation (recall@k, MRR, overall and per-`source`) lives in
`src/eval/retrieval_eval.py` — any registered retriever can be scored this
way, including `hybrid`. `src/run_bm25.py` and `src/run_dense.py` are the
existing worked examples (see the tables above for their current numbers);
see `src/retrieval/README.md#evaluating-a-retriever` for how to point it at
a new retriever.

## Running the app (API + Streamlit)

Two processes, in two terminals.

**1. Start the API:**

```bash
python -m uvicorn app.backend.api:app --reload
```

Query directly on Linux/macOS:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What campaigns used attack technique ’T1562.001: Disable or Modify Tools’?", "k": 5, "retriever": "bm25", "generator": "llama"}'
```

On Windows PowerShell:

```shell
# Define a small helper
function query-rag($q) {
    irm http://127.0.0.1:8000/query -Method Post -ContentType "application/json" -Body (@{
        query=$q
        k=5
        retriever="bm25"
        generator="llama"
    } | ConvertTo-Json)
}

# Then query
query-rag "What campaigns used attack technique ’T1562.001: Disable or Modify Tools’?"
```

`GET /` lists every currently-registered retriever and generator — useful
after adding a new one. `GET /health` reports whether the default
retriever's index exists on disk. `POST /retrieve` runs retrieval only, for
debugging retrieval quality without paying for generation. More details, visit:
- Swagger UI → http://127.0.0.1:8000/docs
- ReDoc → http://127.0.0.1:8000/redoc

**2. Start the frontend, in a separate terminal:**

```bash
python -m streamlit run app/frontend/app.py
```

The Streamlit app reads its retriever/generator dropdown options straight
from the API's `GET /` response, so a newly-registered retriever or
generator shows up there automatically. See `app/README.md` for the request/
response contract between the two.

## Configuration

`src/config.py` is the single source of truth for every path, model name,
and default that used to be duplicated (and had already drifted) across
`run_rag.py`, `run_bm25.py`, `run_dense.py`, and the old broken
`retriever_factory.py` / `api.py` / `full_api.py`. Everything in it can be
overridden via environment variables or a `.env` file — see `.env.example`
for the full list, e.g.:

```dotenv
DEFAULT_RETRIEVER=dense
DEFAULT_TOP_K=8
BM25_INDEX_DIR=data/index/bm25_v2
RERANK_ENABLED=false
```

## Repository Structure

```
textmining-rag-system/
├── data/
│   ├── benchmark/                 # AttackQA dataset (attackqa.parquet)
│   ├── processed/                 # corpus.jsonl + qa_{train,dev,test}.jsonl
│   └── index/                     # built indexes (bm25/, dense/)
├── src/
│   ├── config.py                  # single source of truth for paths/models/defaults
│   ├── factory.py                 # name -> working Retriever/Generator/Pipeline
│   ├── pipeline.py                # Pipeline: Retriever.search() -> Generator.generate()
│   ├── data_models/                # pydantic models (Document, Answer, ...) + jsonl I/O
│   ├── ingestion/                  # attackqa.parquet -> corpus + QA splits
│   ├── indexing/
│   │   ├── base.py                # abstract Index interface
│   │   ├── bm25_index.py
│   │   └── dense_index.py
│   ├── retrieval/
│   │   ├── base.py                # abstract Retriever interface
│   │   ├── registry.py            # @register_retriever + build_retriever()
│   │   ├── bm25_retriever.py
│   │   ├── dense_retriever.py
│   │   ├── hybrid_retriever.py    # RRF/weighted fusion of bm25 + dense
│   │   └── README.md              # how retrieval works + how to add a retriever
│   ├── reranking/
│   │   ├── base.py                # abstract Reranker interface
│   │   ├── registry.py            # @register_reranker + build_reranker()
│   │   ├── cross_encoder_reranker.py
│   │   └── README.md              # how reranking works + how to add a reranker
│   ├── generation/
│   │   ├── base.py                # abstract Generator interface
│   │   ├── registry.py            # @register_generator + build_generator()
│   │   ├── llama_generator.py
│   │   ├── qwen_generator.py
│   │   ├── context_builder.py     # builds the bounded "Context:" block
│   │   ├── output_parser.py       # fail-safe JSON -> (answer, found) parsing
│   │   ├── response_builder.py    # GenerationResult -> citation-bearing Answer
│   │   ├── prompt.py
│   │   └── README.md              # design rationale + how to add a generator
│   ├── eval/
│   │   └── retrieval_eval.py      # recall@k / MRR, overall + per source
│   ├── run_bm25.py                # build/eval the BM25 index
│   └── run_dense.py               # build/eval the dense index
├── app/
│   ├── backend/
│   │   └── api.py                 # FastAPI: /, /health, /retrieve, /query
│   ├── frontend/
│   │   └── app.py                 # Streamlit chat UI, calls the API over HTTP
│   └── README.md                  # how to run both + the API contract
├── run_rag.py                     # interactive CLI for the full pipeline
├── pyproject.toml                 # pip install -e .
├── requirements.txt
├── .env.example
└── README.md                      # you are here
```

## Extending the System

The three abstract interfaces (`Index`, `Retriever`, `Generator`) plus
three registries (`src/retrieval/registry.py`, `src/generation/registry.py`,
`src/reranking/registry.py`) are what make adding new tech mostly additive
instead of requiring edits scattered across the CLI, the API, and a factory
file — which is exactly how this codebase drifted the first time (a factory
that imported a module that no longer existed, a constructor signature that
no longer matched, a frontend expecting fields the API never sent).

**Add a new retriever** (e.g. a hybrid BM25+dense retriever, or one backed
by a different vector store):

1. Implement it in `src/retrieval/your_retriever.py`, subclassing `Retriever`
   and decorating the class with `@register_retriever("your_name")`.
2. Add one import line to `src/factory.py`'s "side-effect imports" section so
   it gets registered.
3. If it needs its own index type, add a branch to `get_retriever()` in
   `src/factory.py` for how to build/load it (mirroring the `bm25`/`dense`
   branches) and any new settings (index dir, hyperparameters) to
   `src/config.py`.

That's it — `run_rag.py --retriever your_name`, `POST /query` with
`"retriever": "your_name"`, and the Streamlit dropdown all work
immediately, with no further changes. Full walkthrough with a worked example
in `src/retrieval/README.md`.

**Add a new generator** (a new LLM provider, a different local model): same
pattern in `src/generation/`, decorate with `@register_generator("your_name")`,
add the import in `src/factory.py`, and any new settings in
`src/config.py`. Full walkthrough in `src/generation/README.md`.

**Add a new reranker** (e.g. an LLM-based reranker, or a different
cross-encoder checkpoint): same pattern in `src/reranking/`, decorate with
`@register_reranker("your_name")`, add the import in `src/factory.py`, and
teach `get_reranker()` there how to build one. Unlike retrievers/generators,
reranker selection is config-only (`DEFAULT_RERANKER` / `RERANK_ENABLED` in
`.env`), not a per-request API field — see `src/reranking/README.md` for the
full walkthrough and why.

**Swap a piece of infrastructure entirely** (e.g. a different vector DB, a
different embedding provider): implement it behind the existing `Index` or
`Retriever` interface so the rest of the system — pipeline, API,
frontend, eval scripts — doesn't need to change at all.

## Where to look for what

| I want to... | Look at |
|---|---|
| Change a path, model name, or default | `src/config.py`, `.env.example` |
| See how a question turns into an object graph | `src/factory.py` |
| Add a new retriever / vector store | `src/retrieval/README.md` |
| Add a new reranker | `src/reranking/README.md` |
| Add a new generator / LLM provider | `src/generation/README.md` |
| Understand ingestion dedup/split logic | `src/ingestion/README.md` |
| Run or extend the API / Streamlit app | `app/README.md` |
| Run or extend retrieval evaluation | `src/eval/retrieval_eval.py`, `src/retrieval/README.md#evaluating-a-retriever` |

## References

- \[[paper](https://onlinelibrary.wiley.com/doi/full/10.1155/jece/3383674)\] Large Language Models for Security Operations Centers: A Comprehensive Survey.
- \[[paper](https://arxiv.org/abs/2411.01073)\] AttackQA: Development and Adoption of a Dataset for Assisting Cybersecurity Operations using Fine-tuned and Open-Source LLMs.
