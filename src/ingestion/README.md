# Ingestion

Parses AttackQA into a corpus (`corpus.jsonl`) for indexing, and QA pairs
with train/dev/test splits (default 80/10/10) for evaluation.

```bash
python -m src.ingestion.run_ingestion

# See src/config.py for where to add attackqa.parquet and where to get
# processed data.
```

No chunking in the current implementation — AttackQA has already done that.
See `src/ingestion/README.md` for dedup/split design decisions, or explore
the [AttackQA dataset](https://huggingface.co/datasets/sambanovasystems/attackqa/blob/main/Getting%20Started%20with%20MITRE%20QA%20Dataset.ipynb)
directly.

## Data Ingestion

Turns `data/benchmark/attackqa.parquet` into three artifacts under
`data/processed/`:

- `corpus.jsonl` — one row per **unique** document (deduplicated), matching `Document`
- `qa_train.jsonl` / `qa_dev.jsonl` / `qa_test.jsonl` — one row per QA pair, each
  pointing at the doc_id(s) it should retrieve, matching `QAExample`
- `dup_report.jsonl` — doc_ids whose text is identical but whose `subject_id`
  differs across rows (should be rare/empty on the real data — worth a manual look
  if not, since it means the same passage is claimed by more than one subject)

## Design decisions (confirmed)

- **No further chunking.** AttackQA's `document` field is already an atomic
  snippet; a `Document` *is* the retrieval unit for stage01.
- **Raw text, no metadata enrichment.** Indexed text is exactly the `document`
  field, unmodified — matches the AttackQA baseline. (We can revisit
  contextual enrichment later as a "cybersecurity-specific improvement" in a
  later stage, per the project plan.)
- **Dedup key:** `sha1(document.strip())[:16]`. Deterministic and content-derived,
  so re-running ingestion always produces the same doc_id for the same text.
- **Split:** 80/10/10, stratified by `source` (not a random split), using a
  manual per-group allocator rather than `sklearn.train_test_split(...,
  stratify=...)` — the latter fails outright on `source` categories with very
  few rows (confirmed while testing: a rare category with 4 rows crashes
  sklearn's nested train→temp→dev/test stratification). The manual allocator
  degrades gracefully instead: any `source` with fewer than 10 rows is still
  split proportionally but flagged in the console output as "thin", so you
  know dev/test metrics on that slice will be noisy.


Prints a summary: raw row count, unique document count + dedup rate, split
sizes, any thin `source` categories, and whether the dup_report flagged
anything.

## Known limitation / next check

This was built and tested against a **synthetic** stand-in dataset with the
same schema (since I don't have the real parquet file in this environment) —
correctness of the row-level logic (hashing, dedup, split, JSON serialization)
is verified, including a `NaN`-vs-`null` JSON bug caught in testing. What I
could *not* verify against the real data:
- actual dedup rate (expect the real dataset's 25,335 → 17,760 ratio, i.e. ~30%
  dedup — matches what I saw synthetically, which is reassuring but coincidental)
- whether any `source` categories are thin enough to trigger the fallback path
- whether `dup_report.jsonl` comes back non-trivial (would indicate the same
  document text is legitimately shared across subjects in MITRE ATT&CK, e.g.
  a generic detection note reused for several techniques — worth knowing either way)

Once you run this against the real `attackqa.parquet`, share the printed
summary and I'll sanity-check the numbers with you before we move to indexing.
