# AttackQA Dataset Analysis and Evaluation

Source: [sambanovasystems/attackqa](https://huggingface.co/datasets/sambanovasystems/attackqa)

## 1. Dataset Overview

- Rows: 25335
- Columns: 16
- Split analyzed: train
- Raw snapshot: `data/attackqa_train.jsonl`

### Columns

- question
- thought
- answer
- document
- subject_id
- subject_name
- subject_type
- url
- source
- references
- human_question
- human_answer
- field
- relation_id
- relation_name
- __index_level_0__

## 2. Numeric Profile

### Main counts
- Human question rows: 17535 (69.2%)
- Human answer rows: 4994 (19.7%)
- Rows with `field=description`: 5454 (21.5%)
- Rows with no `relation_name`: 12045 (47.5%)

### References per row
- Min: 0
- Median: 1.0
- Mean: 1.11
- Max: 7

### Text length in characters
- Question median: 93
- Answer median: 139
- Document median: 199
- Thought median: 114
- Question mean: 88.6
- Answer mean: 175.9
- Document mean: 335.6
- Thought mean: 122.4

## 3. Categorical Characteristics

### Source distribution
- relationships_uses_software: 8534 (33.7%)
- software: 2607 (10.3%)
- relationships_detects: 2534 (10.0%)
- relationships_mitigations: 2222 (8.8%)
- techniques: 2139 (8.4%)
- relationships_detections_summaries: 1431 (5.6%)
- relationships_mitigations_summaries: 1307 (5.2%)
- relationships_techniques_for_software: 677 (2.7%)
- techniques_tactics: 637 (2.5%)
- groups: 553 (2.2%)

### Subject field quality
- Unique `subject_type` values: 99
- `subject_type` values that look like ATT&CK IDs: 11846
- Rows with broad subject categories (`techniques`, `software`, `groups`, `campaigns`, `tactics`): 13489

This field is not a clean taxonomy. It mixes broad categories with ATT&CK IDs, so it should be treated carefully in downstream analysis.

### Top ATT&CK subjects

#### By subject ID
- T1105: 371
- T1082: 329
- T1071.001: 300
- T1083: 271
- T1059.003: 249
- T1057: 242
- T1140: 232
- T1016: 219
- T1070.004: 212
- T1033: 191
- T1106: 181
- T1005: 168
- T1027: 162
- T1573.001: 156
- T1041: 153

#### By subject name
- Ingress Tool Transfer: 371
- System Information Discovery: 329
- Application Layer Protocol: Web Protocols: 300
- File and Directory Discovery: 271
- Command and Scripting Interpreter: Windows Command Shell: 249
- Process Discovery: 242
- Deobfuscate/Decode Files or Information: 232
- System Network Configuration Discovery: 219
- Indicator Removal: File Deletion: 212
- System Owner/User Discovery: 191
- Native API: 181
- Data from Local System: 168
- Obfuscated Files or Information: 162
- Encrypted Channel: Symmetric Cryptography: 156
- Exfiltration Over C2 Channel: 153

## 4. Evaluation

### What looks strong

- The dataset is large enough for meaningful QA and retrieval experiments.
- It is strongly grounded in ATT&CK relationships, which makes it useful for cyber-domain reasoning.
- Questions are relatively short, while answers and documents provide more explanatory context.
- Reference lists are usually small, which keeps the examples focused.

### What needs caution

- The dataset has only one split (`train`), so there is no built-in validation/test separation.
- `subject_type` is noisy and overloaded, so it should not be used as a single authoritative label.
- Nearly half the rows have no `relation_name`, which limits how much structure is available for some examples.
- The corpus is skewed toward a small number of high-frequency ATT&CK techniques, so the dataset is long-tailed.

### Practical interpretation

- The dataset is well suited for training or evaluating ATT&CK-aware QA and retrieval systems.
- It is less suitable as a clean classification dataset because several metadata fields are mixed or incomplete.
- For RAG work, the answer and document fields are especially valuable because they contain the context needed to support grounding.

## 5. Conclusion

AttackQA is a large, ATT&CK-grounded question answering corpus with strong cybersecurity semantics and rich explanatory text. Its main strength is domain realism: the questions, answers, and supporting documents are tied to ATT&CK subjects and relations rather than generic QA pairs. Its main weakness is metadata inconsistency, especially in `subject_type`, which means any downstream analysis should rely on carefully chosen fields rather than assuming all columns are cleanly normalized.

For next-step work, the best follow-ups are split-based evaluation, retrieval quality checks by subject type, and a comparison of examples that are human-authored versus machine-generated.

## 6. Figure-by-Figure Analysis

### `source_distribution.png`

This bar chart shows the example count by `source`. It is the clearest view of how the dataset is constructed.

The dominant source is `relationships_uses_software`, which accounts for about one-third of the corpus. That means a large share of the dataset is centered on software-to-technique relationships. The next largest groups are `software`, `relationships_detects`, `relationships_mitigations`, and `techniques`, all of which indicate that the dataset is not just a flat QA set but a structured ATT&CK-derived corpus built from multiple relation templates.

The distribution has a long tail of smaller sources. This matters because it implies the dataset is multi-purpose: some rows are direct technique descriptions, while others are generated from specific ATT&CK relationships. For modeling, this is useful because it gives both explanation-style and relation-style examples, but it also means the dataset is not homogeneous.

### `subject_id_top15.png`

This chart shows the most frequent ATT&CK subject IDs. It is a long-tailed distribution with a small set of high-frequency techniques dominating the dataset.

`T1105` (Ingress Tool Transfer) and `T1082` (System Information Discovery) are the two most common subjects, followed by `T1071.001`, `T1083`, and `T1059.003`. The top items are all operationally important ATT&CK techniques, which suggests the dataset prioritizes commonly discussed or highly relevant behaviors.

The steep drop after the top few IDs indicates strong skew. That is important for evaluation because models may perform very well on recurring techniques but much less consistently on rare ones. Any retrieval or QA benchmark built on this corpus should therefore report performance by subject frequency bucket, not only overall averages.

### `relation_name_top15.png`

This figure shows the most common relation names. Unlike the subject ID chart, this one exposes the semantic roles that appear in the dataset.

The top relation names include `Process Creation`, `Command Execution`, `OS API Execution`, `Network Traffic Content`, and `Network Traffic Flow`. These are broad behavioral or detection-oriented concepts, which means the dataset covers not only ATT&CK technique labels but also how techniques are detected or manifested.

The figure also has a large `None` category, which indicates that many rows do not have a relation name attached. This is a useful warning sign: the dataset is rich, but relation metadata is incomplete for a substantial portion of the rows. In practice, that means relation-based filtering will work for some slices and be unavailable for others.

### `human_flags.png`

This bar chart compares the counts of human-authored versus non-human-authored questions and answers.

Human questions are the majority at about 69%, while human answers are much rarer at about 20%. That suggests the dataset is mixed: the question side is relatively more natural and likely curated, but a large share of answers appear to be machine-generated or otherwise non-human.

This imbalance has a practical implication. If you use the dataset for training or evaluation, the question distribution is more natural-language-like than the answer distribution. That means answer quality, style, and phrasing may vary more than question quality. For generative evaluation, this figure is a reminder to inspect outputs carefully, because the target responses are not uniformly human-written.

### `field_distribution.png`

This figure shows the distribution of the `field` column.

Most rows have `field=None`, while a smaller portion are labeled `description`. That means the field metadata is sparse and should be treated as a secondary attribute rather than a primary organizing label.

The chart is still useful because it shows that one well-defined subset of the dataset focuses on description-style content. That subset is likely the most useful for retrieval and summarization tasks, while the rest of the dataset captures broader ATT&CK relation examples.

### `subject_type_quality.png`

This chart is a data-quality diagnostic rather than a semantic distribution plot.

The figure separates `subject_type` into three buckets: ATT&CK-ID-like values, broad category labels, and other values. The key takeaway is that the field is mixed. A large number of rows use ATT&CK-like technique IDs, while another large block uses broad labels such as `techniques`, `software`, `groups`, `campaigns`, and `tactics`.

This means `subject_type` is not a clean taxonomy. It appears to mix subject labels at different abstraction levels, which limits its usefulness for downstream grouping. The chart is valuable because it makes this inconsistency visually obvious.

### `references_per_row.png`

This histogram shows how many references each row contains.

The median is 1 and the mean is only slightly above 1, so most rows have either zero or one reference, with a few rows having several. That is a compact distribution and suggests the dataset is not overloaded with citations.

For QA and retrieval, this is a good sign because the evidence structure is lightweight. At the same time, it also means that many examples rely primarily on the generated document text rather than a large set of supporting references. The small right tail up to 7 references indicates a few richer examples, but they are the exception.

### `length_distributions.png`

This is a 2x2 grid showing the length distributions of `question`, `answer`, `document`, and `thought` fields.

The pattern is consistent across the grid: questions are shortest, thoughts are slightly longer, answers are longer still, and documents are the longest. This is exactly what you would expect in a QA dataset where the question is concise, the answer is explanatory, and the document provides broader context.

The document distribution has the widest spread, which indicates that some examples are much more verbose than others. That is useful for RAG, because long documents provide more retrieval context, but it also means the dataset contains a mixture of compact and elaborate examples. The length grid is especially helpful for spotting that the dataset is not uniform: there is no single fixed-length style across all fields.

### Overall figure interpretation

Taken together, the figures show a dataset that is:

- ATT&CK-centered rather than generic QA
- skewed toward a few very frequent techniques and relation templates
- mixed in terms of metadata cleanliness
- rich enough in text to support retrieval and explanation tasks

The charts also show why AttackQA is a strong fit for RAG-style work: the dataset combines short questions, contextual documents, and structured ATT&CK relations. Its main limitation is not lack of scale, but the unevenness of its metadata and the long-tailed nature of its subject coverage.