# AttackQA Dataset EDA

This report uses the current AttackQA split files in `data/processed/` and the deduplicated corpus produced by ingestion.

## 1. Dataset Overview

- Corpus documents: 17760
- QA rows: 25335
- Splits: train=20268, dev=2533, test=2534
- Corpus doc length median: 181 chars
- Corpus doc length mean: 274.6 chars

### Corpus source mix
- relationships_uses_software: 8534
- relationships_detects: 1618
- relationships_mitigations: 1131
- relationships_techniques_for_software: 677
- software: 676
- techniques_tactics: 637
- techniques: 618
- relationships_detections_summaries: 597
- relationships_mitigations_summaries: 544
- relationships_groups_for_software: 498
- techniques_parent: 435
- relationships_software_for_technique: 428

### Corpus subject-type mix
- techniques: 6653
- software: 1936
- T1059: 604
- T1027: 445
- groups: 435
- T1071: 401
- T1070: 378
- T1218: 265
- T1036: 251
- T1547: 242
- T1055: 225
- T1573: 220

## 2. Question Analysis

The questions were grouped with a lightweight heuristic into five intent buckets. The goal is not perfect NLP labeling, but a practical decomposition of retrieval difficulty.

- relation_or_detection: 16511 (65.2%)
- direct_lookup: 6731 (26.6%)
- rewrite_or_summarize: 1417 (5.6%)
- reasoning_translation: 654 (2.6%)
- other: 22 (0.1%)

### Top question openings
- how: 12822
- what: 9508
- describe: 1431
- which: 657
- who: 296
- when: 290
- why: 217
- where: 58
- is: 23
- in: 12
- can: 8
- are: 3
- against: 3
- has: 3
- at: 2

## 3. Document Length Analysis

Length buckets are based on the 33rd and 67th percentiles of corpus document length: short <= 159.0, long >= 222.0.

### Document length buckets
- medium: 7569 (29.9% of QA rows)
- long: 10723 (42.3% of QA rows)
- short: 7043 (27.8% of QA rows)

### What short documents tend to ask about
- relation_or_detection: 4369
- direct_lookup: 2448
- rewrite_or_summarize: 213
- reasoning_translation: 11
- other: 2

Top sources for short documents:
- relationships_uses_software: 3181
- software: 807
- techniques_tactics: 632
- relationships_groups_for_software: 470
- relationships_mitigations_summaries: 466
- relationships_detections_summaries: 324
- relationships_groups_for_technique: 250
- techniques_parent: 222
- relationships_software_for_technique: 189
- relationships_campaigns_for_technique: 162

### What long documents tend to ask about
- relation_or_detection: 5788
- direct_lookup: 3322
- rewrite_or_summarize: 1006
- reasoning_translation: 591
- other: 16

Top sources for long documents:
- relationships_detects: 2206
- techniques: 2139
- relationships_mitigations: 1731
- relationships_uses_software: 1106
- software: 1095
- relationships_techniques_for_software: 537
- groups: 487
- relationships_mitigations_summaries: 338
- relationships_detections_summaries: 301
- relationships_software_for_technique: 184

## 4. Answer vs Document / Thought / Question

- Answer tokens contained in document on average: 0.873
- Answer tokens contained in thought on average: 0.352
- Answer tokens contained in question on average: 0.323
- Exact answer substring in document: 35.1%
- Exact answer substring in thought: 0.5%
- Exact answer substring in question: 0.0%

### Dominant answer keyword source
- document: 24504 (96.7%)
- question: 595 (2.3%)
- thought: 236 (0.9%)

- Extractive answers: 19141
- Semi-extractive answers: 5771
- Abstractive answers: 423

Interpretation:
- The document field is the main anchor for answer tokens, which means retrieval quality matters directly for answer generation.
- The thought field contributes additional paraphrastic reasoning in some examples, but it is not usually the main lexical source of the answer.
- Questions themselves rarely contain the full answer text; they usually encode the intent, not the answer span.

## 5. Evaluation-oriented interpretation

- Direct lookup questions should be easiest for exact retrieval.
- Relation and detection questions need retrieval plus a small amount of reasoning over the returned document.
- Rewrite/summarize questions are the hardest because the answer must be translated into a user-facing format.
- Long documents are not simply harder because they are long; they also tend to carry richer relation context and broader explanatory text.

## 6. Figure-by-Figure Notes

### `source_distribution.png`
Shows how the deduplicated corpus is constructed by ATT&CK source template. A few source templates dominate, which means the corpus is structured and long-tailed rather than uniform.

### `subject_type_distribution.png`
Shows that corpus metadata mixes broad categories and ATT&CK IDs. This is useful as a data-quality warning, not as a clean class taxonomy.

### `question_category_distribution.png`
Shows the relative balance of lookup, relation/detection, reasoning, and rewrite-style questions. This is the main chart for question difficulty.

### `question_category_by_doc_bucket.png`
Shows how question intent shifts as documents get shorter or longer. Short documents lean more toward direct lookup and relation questions; long documents lean more toward reasoning and rewrite questions.

### `document_length_distribution.png`
Shows the corpus is length-skewed with a long tail of very verbose documents, but many examples are still compact.

### `question_length_distribution.png`
Questions stay comparatively short even when the supporting document is long, which is exactly the pattern you want for QA retrieval.

### `answer_length_distribution.png`
Answers are usually longer than questions and often shorter than the full document. This suggests a mix of extractive and paraphrased answers.

### `answer_style_distribution.png`
Separates extractive, semi-extractive, and abstractive answers. A large extractive/semi-extractive portion means many answers are grounded directly in the document text.

### `answer_keyword_source.png`
Shows which field usually carries the answer vocabulary. The document field should dominate if the retrieval pipeline is healthy; thought is secondary reasoning support; question is usually the intent signal.

### `answer_field_recall.png`
Shows average token recall from answer to document/question/thought. This is the cleanest field-level summary of where answer keywords come from.

### `answer_exact_match_rate.png`
Shows how often the exact answer string appears in each field. A high document rate means answers are often extractive or lightly paraphrased.

### `answer_doc_recall_distribution.png`
Shows the spread of answer-to-document token overlap. A right-skew toward high recall means many answers are grounded in the document; a broad spread means some are more abstract.

## 7. Conclusion

AttackQA is not a uniform QA dataset. It mixes direct lookup questions, relation-based questions, explanatory questions, and reformulation questions. Most answer content is grounded in the document text, with the thought field serving as secondary reasoning support. The corpus is also length-skewed: short documents are often crisp relation or detection snippets, while long documents are more likely to support reasoning and synthesis-style questions.