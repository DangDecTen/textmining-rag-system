"""
Generated-answer vs. sample-answer text metrics: lexical (BLEU, ROUGE) and
semantic (BERTScore).

Design notes
------------
- BLEU: `sacrebleu` rather than `nltk` -- sacrebleu's tokenization and
  smoothing are standardized/reproducible across runs and papers, which
  matters if these numbers get compared against anything published. We
  report BOTH a per-example *sentence*-BLEU (for the per-row JSONL, so you
  can spot individual bad generations) and a true *corpus*-BLEU over the
  whole split (for the aggregate report) -- corpus-BLEU is NOT the mean of
  the per-example sentence-BLEU scores (BLEU's n-gram precision clipping is
  not linear/averageable that way), so both numbers coexist rather than one
  being derived from the other.
- ROUGE: `rouge-score` (Google's reference implementation), F-measure of
  ROUGE-1/2/L. All three computed together since it's a single library call
  either way and each variant surfaces something different (1 = unigram
  overlap/keyword recall, L = longest common subsequence/fluency-sensitive).
- BERTScore: `bert_score`, default `roberta-large` model + its own default
  layer. Computed as ONE batched call over the whole split rather than in
  the per-example loop -- bert_score is designed for batched input, and
  calling it once per example would reload/re-run the model far more than
  necessary and be much slower.
- All three libraries are imported lazily (inside functions) so importing
  this module doesn't force their installation/model-download just to, say,
  run the retrieval-only part of an eval.
"""
from __future__ import annotations


def sentence_bleu(hypothesis: str, reference: str) -> float:
    """Per-example sentence-BLEU (0-100 scale, sacrebleu convention)."""
    from sacrebleu.metrics import BLEU

    if not hypothesis.strip():
        # sacrebleu scores an empty hypothesis as 0 anyway, but guard
        # explicitly so an abstained/empty answer doesn't raise on some
        # sacrebleu versions that warn/error on empty input.
        return 0.0
    bleu = BLEU(effective_order=True)  # effective_order: needed for short/sentence-level BLEU
    return bleu.sentence_score(hypothesis, [reference]).score


def corpus_bleu(hypotheses: list[str], references: list[str]) -> float:
    """Corpus-level BLEU (0-100 scale) over an entire split -- NOT the mean
    of `sentence_bleu` scores; see module docstring."""
    from sacrebleu.metrics import BLEU

    bleu = BLEU()
    return bleu.corpus_score(hypotheses, [references]).score


def rouge_scores(hypothesis: str, reference: str) -> dict[str, float]:
    """ROUGE-1/2/L F-measure for one (hypothesis, reference) pair."""
    from rouge_score import rouge_scorer

    scorer = _get_rouge_scorer()
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


_ROUGE_SCORER = None


def _get_rouge_scorer():
    # Cached at module level: rouge_scorer.RougeScorer construction does
    # non-trivial setup (stemmer, tokenizer) -- avoid redoing it per example.
    global _ROUGE_SCORER
    if _ROUGE_SCORER is None:
        from rouge_score import rouge_scorer

        _ROUGE_SCORER = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
    return _ROUGE_SCORER


def bertscore_batch(
    hypotheses: list[str],
    references: list[str],
    model_type: str = "roberta-large",
    batch_size: int = 32,
    device: str | None = None,
) -> list[dict[str, float]]:
    """BERTScore precision/recall/F1 for a whole split in one batched call.

    Returns a list aligned index-for-index with `hypotheses`/`references`.
    An empty hypothesis (e.g. an abstained answer with empty generated text)
    is still scored as-is against the reference -- bert_score handles empty
    strings by producing a low-but-defined score rather than erroring, and
    we deliberately do not special-case abstentions here (see run_eval.py).
    """
    from bert_score import score as bert_score_fn

    precision, recall, f1 = bert_score_fn(
        hypotheses,
        references,
        model_type=model_type,
        batch_size=batch_size,
        device=device,
        verbose=False,
    )
    return [
        {"bertscore_precision": p.item(), "bertscore_recall": r.item(), "bertscore_f1": f.item()}
        for p, r, f in zip(precision, recall, f1)
    ]
