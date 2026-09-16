"""
Generator evaluation: for each QA example in a split, run the full
Retriever [-> Reranker] -> Generator pipeline, then score the generated
answer against the sample (ground-truth) answer with:

    - Retrieval:  Recall@k, MRR@k  (k = the top_k actually used for generation)
    - Lexical:    BLEU (sacrebleu), ROUGE-1/2/L (rouge-score)
    - Semantic:   BERTScore (bert_score)

Writes one JSONL row per example (metrics + metadata, for debugging
individual failures) plus a summary JSON (overall + grouped by `source`).

Usage:
    python -m evaluation.generation.run_eval
    python -m evaluation.generation.run_eval --split dev_small --retriever bm25 --generator hf
    python -m evaluation.generation.run_eval --split test --top-k 5 --no-rerank

Notes:
- Abstained answers (generator returned found=False -> ResponseBuilder's
  fixed ABSTAIN_MESSAGE) are scored as-is against the sample answer, same as
  any other generated text -- no special-casing. `abstained` is still
  recorded per row so you can slice it out afterward if you want to.
- Reranking is forced on/off for the whole run via --no-rerank rather than
  silently following whatever settings.rerank_enabled happens to be in
  .env, so "Recall@k after reranking" in the report is guaranteed to mean
  what it says.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from src.config import settings
from src.data_models.io import load_qa_examples
from src.factory import get_pipeline

from evaluation.generation.retrieval_metrics import mrr_at_k, recall_at_k
from evaluation.generation.text_metrics import bertscore_batch, corpus_bleu, rouge_scores, sentence_bleu


def _run_pipeline(pipeline, examples, top_k: int) -> list[dict]:
    """Run the pipeline over every example, returning per-row dicts with
    everything except the text metrics (BLEU/ROUGE/BERTScore), which are
    computed afterward in a batch (see module docstring on BERTScore)."""
    rows = []
    for i, ex in enumerate(examples, start=1):
        answer, generation_result = pipeline.answer_with_debug(ex.question, top_k=top_k)
        retrieved_doc_ids = [r.doc_id for r in generation_result.retrieval_results]

        rows.append(
            {
                "qa_id": ex.qa_id,
                "source": ex.source,
                "question": ex.question,
                "sample_answer": ex.answer,
                "generated_answer": answer.text,
                "abstained": answer.abstained,
                "relevant_doc_ids": ex.relevant_doc_ids,
                "retrieved_doc_ids": retrieved_doc_ids,
                "k": top_k,
                "recall_at_k": recall_at_k(retrieved_doc_ids, ex.relevant_doc_ids),
                "mrr_at_k": mrr_at_k(retrieved_doc_ids, ex.relevant_doc_ids),
                "latency_ms": generation_result.latency_ms,
            }
        )
        if i % 20 == 0 or i == len(examples):
            print(f"  ...{i}/{len(examples)}")
    return rows


def _add_text_metrics(rows: list[dict]) -> None:
    """Adds sentence-BLEU, ROUGE-1/2/L, and BERTScore to each row in place.

    BERTScore is computed as one batched call over the whole split (see
    text_metrics.bertscore_batch docstring for why); BLEU/ROUGE are cheap
    pure-Python-ish per-example calls so a simple loop is fine for those.
    """
    hypotheses = [r["generated_answer"] for r in rows]
    references = [r["sample_answer"] for r in rows]

    for row, hyp, ref in zip(rows, hypotheses, references):
        row["bleu"] = sentence_bleu(hyp, ref)
        row.update(rouge_scores(hyp, ref))

    print(f"Computing BERTScore ({settings.bertscore_model_name}) over {len(rows)} examples...")
    bert_scores = bertscore_batch(
        hypotheses,
        references,
        model_type=settings.bertscore_model_name,
        batch_size=settings.bertscore_batch_size,
    )
    for row, bs in zip(rows, bert_scores):
        row.update(bs)


_MEAN_METRICS = [
    "recall_at_k", "mrr_at_k", "bleu", "rouge1", "rouge2", "rougeL",
    "bertscore_precision", "bertscore_recall", "bertscore_f1", "latency_ms",
]


def _summarize(rows: list[dict]) -> dict:
    """Mean of every metric in _MEAN_METRICS, plus true corpus-BLEU
    (not the mean of per-row sentence-BLEU -- see text_metrics docstring)
    and the abstention rate, for `rows` as a whole."""
    if not rows:
        return {"n": 0}
    summary = {"n": len(rows)}
    for metric in _MEAN_METRICS:
        summary[metric] = statistics.mean(r[metric] for r in rows)
    summary["corpus_bleu"] = corpus_bleu(
        [r["generated_answer"] for r in rows], [r["sample_answer"] for r in rows]
    )
    summary["abstention_rate"] = statistics.mean(1.0 if r["abstained"] else 0.0 for r in rows)
    return summary


def build_report(rows: list[dict]) -> dict:
    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)

    return {
        "overall": _summarize(rows),
        "by_source": {source: _summarize(group) for source, group in sorted(by_source.items())},
    }


def _write_jsonl(rows: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _print_report(report: dict) -> None:
    def _fmt(summary: dict) -> str:
        if summary.get("n", 0) == 0:
            return "n=0"
        return (
            f"n={summary['n']:<4} "
            f"Recall@k={summary['recall_at_k']:.3f} MRR@k={summary['mrr_at_k']:.3f} "
            f"BLEU={summary['bleu']:.2f} (corpus={summary['corpus_bleu']:.2f}) "
            f"ROUGE-1/2/L={summary['rouge1']:.3f}/{summary['rouge2']:.3f}/{summary['rougeL']:.3f} "
            f"BERTScore-F1={summary['bertscore_f1']:.3f} "
            f"abstain_rate={summary['abstention_rate']:.3f}"
        )

    print("\n===== Overall =====")
    print(_fmt(report["overall"]))
    print("\n===== By source =====")
    for source, summary in report["by_source"].items():
        print(f"[{source}]")
        print(f"  {_fmt(summary)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default=settings.eval_default_split,
                         help="QA split to evaluate, e.g. dev_small/dev/test_small/test")
    parser.add_argument("--retriever", default=settings.default_retriever)
    parser.add_argument("--generator", default=settings.default_generator)
    parser.add_argument("--top-k", type=int, default=settings.default_top_k)
    parser.add_argument("--no-rerank", action="store_true",
                         help="Force reranking off for this run regardless of settings.rerank_enabled")
    parser.add_argument("--limit", type=int, default=None,
                         help="Evaluate only the first N examples (quick smoke test)")
    parser.add_argument("--output-dir", default=None, help="Defaults to settings.eval_results_dir")
    args = parser.parse_args()

    if args.no_rerank:
        settings.rerank_enabled = False

    print(f"Loading split='{args.split}'...")
    examples = load_qa_examples(args.split)
    if args.limit:
        examples = examples[: args.limit]
    print(f"  {len(examples)} examples")

    print(f"Building pipeline (retriever={args.retriever}, generator={args.generator}, "
          f"rerank={'off' if args.no_rerank else settings.rerank_enabled})...")
    pipeline = get_pipeline(retriever_name=args.retriever, generator_name=args.generator)

    print("Running pipeline over examples...")
    rows = _run_pipeline(pipeline, examples, top_k=args.top_k)

    _add_text_metrics(rows)

    report = build_report(rows)
    _print_report(report)

    output_dir = Path(args.output_dir or settings.eval_results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{args.split}_{args.retriever}_{args.generator}_{timestamp}"

    rows_path = output_dir / f"{run_name}.jsonl"
    summary_path = output_dir / f"{run_name}_summary.json"
    _write_jsonl(rows, rows_path)
    with open(summary_path, "w") as f:
        json.dump(
            {
                "split": args.split, "retriever": args.retriever, "generator": args.generator,
                "top_k": args.top_k, "rerank_enabled": settings.rerank_enabled,
                **report,
            },
            f,
            indent=2,
        )

    print(f"\nWrote {len(rows)} rows to {rows_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
