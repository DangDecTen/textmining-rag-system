# Retrieval Evaluation Harness

Runs any registered retriever (see `src/retrieval/README.md`) — optionally
followed by any registered reranker (see `src/reranking/README.md`) — over a
QA split and saves results as JSONL, structured so you can slice and
re-aggregate metrics later — by `source`, by document length, or
anything else — without rerunning retrieval.

Use this harness instead when you want the **per-example** results saved
for later analysis.

## Run

Build the index first to use the retriever (see root README, Indexing & Retrieval). Output goes to `evaluation/retrieval/results/` by default (`--output-dir` to change it).

Calculate the retrieval metrics for each QA pairs.

```bash
# Evaluate each QA pairs and save as JSONL with metadata, details in run_eval.py
python -m evaluation.retrieval.run_eval --retriever bm25 --split dev

# Output:
#     bm25_dev_predictions.jsonl
```

To evaluate a retriever *and* reranker together — i.e. measure what
`src.pipeline.Pipeline` actually does at query time, not just raw retrieval
— add `--rerank`:

```bash
# Retrieves settings.rerank_candidate_k candidates, reranks down to max(k), then scores
python -m evaluation.retrieval.run_eval --retriever bm25 --split dev --rerank
python -m evaluation.retrieval.run_eval --retriever bm25 --split dev --reranker cross_encoder  # same, explicit

# Output (note the '+', keeps this separate from the retrieval-only file above):
#     bm25+cross_encoder_dev_predictions.jsonl
```

Do your own analysis to see how the retrievers perform on different types of QA pairs.

```bash
# Get overal result, or group by question types (source, question length, etc.), details in group_eval.py
python -m evaluation.retrieval.group_eval --retriever bm25 --split dev --group_by source

# Output:
#     bm25_dev_metrics_by_source.jsonl
```

`group_eval.py` needs the **same** `--rerank`/`--reranker` flags as the
`run_eval.py` call that produced the file it's reading — it doesn't infer
this from what's on disk, since a bare `--retriever bm25` is ambiguous
between the two files above:

```bash
python -m evaluation.retrieval.group_eval --retriever bm25 --split dev --group_by source --rerank
# Output:
#     bm25+cross_encoder_dev_metrics_by_source.jsonl
```

## `predictions.jsonl` — one row per QA example

```json
{
  "qa_id": "q_00123",
  "question": "What mitigations exist for T1059?",
  "source": "technique",
  "human_question": false,
  "human_answer": false,
  "num_relevant_docs": 1,
  "relevant_doc_ids": ["a1b2c3d4e5f6..."],
  "relevant_docs_meta": [
    {"doc_id": "a1b2c3d4e5f6...", "subject_type": "technique", "field": "mitigations", "chars": 412, "words": 68}
  ],
  "retrieved": [
    {"rank": 1, "doc_id": "a1b2c3d4e5f6...", "score": 14.2, "is_relevant": true},
    {"rank": 2, "doc_id": "9f8e7d6c5b4a...", "score": 11.8, "is_relevant": false}
  ],
  "rank_of_first_relevant": 1,
  "reciprocal_rank": 1.0,
  "k_values": [1, 5, 10],
  "hit@1": 1, "hit@5": 1, "hit@10": 1,
  "retriever": "bm25",
  "reranker": null,
  "split": "dev"
}
```

`"reranker"` is `null` for a retrieval-only run, or the reranker name (e.g.
`"cross_encoder"`) when the file was produced with `--rerank` — so a
retrieval-only and reranked `predictions.jsonl` for the same retriever stay
distinguishable even after concatenating both into one dataframe (see
[Comparing retrievers](#comparing-retrievers) below).

This is deliberately the most granular file — it does **not** include the
full document text (that would bloat the file with duplicated content
across every question), only ids, scores, and lightweight metadata
(`chars`/`words` length, `subject_type`, `field`) for the *relevant*
document(s). Everything needed to recompute `hit@k` / MRR for any subset of
rows is already in each row — you don't need to re-run retrieval to re-slice
the results.

## `metrics_by_{group_by}.jsonl` — one row per group

An `"overall"` row plus rows when run `group_eval.py` for each `group_by` value.

```json
{"group_by": null, "group": "overall", "n": 2533, "mrr": 0.724, "recall@1": 0.639, "recall@5": 0.831, "recall@10": 0.886, "retriever": "bm25", "reranker": null, "split": "dev"}

{"group_by": "source", "group": "technique", "n": 1204, "mrr": 0.741, "recall@1": 0.652, "recall@5": 0.849, "recall@10": 0.901, "retriever": "bm25", "reranker": null, "split": "dev"}

{"group_by": "human_question", "group": false, "n": 778, "mrr": 0.640, "recall@1": 0.521, "recall@5": 0.793, "recall@10": 0.881, "retriever": "bm25", "reranker": null, "split": "dev"}

{"group_by": "question_len", "group": "q4", "n": 638, "mrr": 0.651, "recall@1": 0.495, "recall@5": 0.873, "recall@10": 0.913, "retriever": "bm25", "reranker": null, "split": "dev"}

{"group_by": "document_len", "group": "q1", "n": 622, "mrr": 0.934, "recall@1": 0.900, "recall@5": 0.974, "recall@10": 0.990, "retriever": "bm25", "reranker": null, "split": "dev"}
```

## View the Analysis and Write Reports

Use `view_metrics.py` to view the metrics from `metrics_by_{group_by}.jsonl` and do analysis. Write your own report, e.g. see `bm25_report.md` for references.

## Comparing retrievers (and rerankers)

Since every row carries `"retriever"`, `"reranker"`, and `"split"`, you can
concatenate multiple runs' `predictions.jsonl` files and compare directly —
across retrievers, or a retriever with and without reranking:

```python
bm25 = pd.read_json("evaluation/retrieval/results/bm25_dev_predictions.jsonl", lines=True)
dense = pd.read_json("evaluation/retrieval/results/dense_dev_predictions.jsonl", lines=True)
bm25_reranked = pd.read_json("evaluation/retrieval/results/bm25+cross_encoder_dev_predictions.jsonl", lines=True)

all_runs = pd.concat([bm25, dense, bm25_reranked])
all_runs.groupby(["retriever", "reranker", "source"])[["hit@1", "hit@5", "hit@10"]].mean()

# Isolate what reranking changed for bm25 specifically (reciprocal_rank is
# per-example here; mean of it is MRR -- the aggregated "mrr" field only
# exists in metrics_by_*.jsonl, not in predictions.jsonl):
pd.concat([bm25, bm25_reranked]).groupby("reranker")["reciprocal_rank"].mean()
```

## Extending

- **New metric per example** (e.g. NDCG, a graded-relevance metric): add a
  field in `evaluate_example()` in `metrics.py`. It'll show up in every
  future `predictions.jsonl` row automatically.
- **New retriever or reranker to evaluate**: nothing to change here —
  `run_eval.py --retriever your_name` / `--reranker your_name` works as
  soon as it's registered via `@register_retriever`/`@register_reranker`
  (see `src/retrieval/README.md`, `src/reranking/README.md`).
