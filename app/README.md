# App layer: FastAPI backend + Streamlit frontend

Two independent processes that talk over HTTP:

```
app/frontend/app.py  --HTTP-->  app/backend/api.py  -->  src/factory.py  -->  src/pipeline.py
   (Streamlit)                     (FastAPI)
```

The frontend never imports anything from `src/` directly — it only knows
about the API's JSON contract below. That means the backend can be deployed,
scaled, or swapped out independently of the UI.

## Running it

```bash
# terminal 1
python -m uvicorn app.backend.api:app --reload

# terminal 2
python -m streamlit run app/frontend/app.py
```

By default the frontend expects the API at `http://localhost:8000`
(`API_URL` at the top of `app/frontend/app.py` — change it if you deploy the
API elsewhere).

## API contract (`app/backend/api.py`)

### `GET /`

Service info, plus what's currently registered:

```json
{
  "message": "Textmining RAG System API",
  "available_retrievers": ["bm25", "dense"],
  "available_generators": ["llama", "qwen"],
  "defaults": {"retriever": "bm25", "generator": "llama"}
}
```

The frontend calls this on load to populate its retriever/generator
dropdowns — so a new `@register_retriever(...)` / `@register_generator(...)`
(see `src/retrieval/README.md`, `src/generation/README.md`) shows up there
with no frontend changes.

### `GET /health`

```json
{"status": "ok", "default_retriever": "bm25", "index_dir": "data/index/bm25"}
```

`status` is `"missing_index"` if the default retriever's index directory
doesn't exist yet — build it first (see the root README's
[Indexing & Retrieval](../README.md#indexing--retrieval) section).

### `POST /retrieve`

Retrieval only — useful for debugging retrieval quality without paying for
generation.

Request:
```json
{"query": "mitigations for command and control", "k": 5, "retriever": "bm25"}
```

Response (`results` is `list[RetrievalResult]` — see
`src/data_models/data_models.py`):
```json
{
  "query": "...",
  "retriever": "bm25",
  "k": 5,
  "results": [
    {"doc_id": "...", "score": 12.3, "document": {"text": "...", "subject_name": "...", "url": "...", "...": "..."}}
  ]
}
```

### `POST /query`

Full pipeline: retrieve + generate + build a citation-bearing answer.

Request:
```json
{"query": "mitigations for command and control", "k": 5, "retriever": "bm25", "generator": "llama"}
```

Response:
```json
{
  "query": "...",
  "retriever": "bm25",
  "generator": "llama",
  "answer": "...",
  "abstained": false,
  "citations": [
    {"doc_id": "...", "subject_id": "...", "subject_name": "...", "source": "...", "url": "...", "...": "..."}
  ],
  "retrieved_context": [
    {"doc_id": "...", "score": 12.3, "document": {"text": "...", "...": "..."}}
  ],
  "latency_ms": 812.4,
  "prompt_tokens": 640,
  "completion_tokens": 41
}
```

Two different fields carry retrieval information, on purpose:

- **`citations`** — deduped, UI-facing metadata (`Citation`, no raw text).
  This is what `answer.citations` looks like after `ResponseBuilder`
  processes it (see `src/generation/README.md`). Use this to show "where did
  this answer come from."
- **`retrieved_context`** — the raw, un-deduped `RetrievalResult`s
  (including document text and scores) that were handed to the generator.
  Use this for a "show me the actual retrieved chunks" debug view — that's
  what the Streamlit app's "🔍 Retrieved chunks (debug)" expander does.

If `abstained` is `true`, `answer` is a fixed "I don't have enough
information..." message and `citations` is empty — the retrieval attempt
still happened, so `retrieved_context` may still be non-empty (useful for
diagnosing *why* it abstained: was the right document even retrieved?).

## Extending the app

- **New retriever/generator**: nothing to change here — see
  [Extending the System](../README.md#extending-the-system) in the root
  README. It'll appear in `GET /` and the Streamlit dropdown automatically.
- **New endpoint**: add it to `app/backend/api.py`, building objects via
  `src.factory` (don't construct `Retriever`/`Generator`/`Pipeline` objects
  directly — that's exactly the duplication that caused the API and CLI to
  drift out of sync before).
- **New frontend**: any client can talk to the API using the contract above
  — it doesn't have to be Streamlit.
