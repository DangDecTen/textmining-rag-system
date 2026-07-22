# AttackQA Retrieval Analysis

This report evaluates how well retrieval works for AttackQA using the saved BM25 and dense indexes.

## 1. Overall Retrieval Quality

| Split | Retriever | Recall@1 | Recall@5 | Recall@10 | MRR@10 | Mean Rank |
|---|---|---:|---:|---:|---:|---:|
| dev | bm25 | 0.639 | 0.831 | 0.886 | 0.724 | 2.84 |
| dev | dense | 0.797 | 0.914 | 0.940 | 0.847 | 1.99 |
| test | bm25 | 0.636 | 0.824 | 0.882 | 0.720 | 2.91 |
| test | dense | 0.784 | 0.906 | 0.933 | 0.836 | 2.07 |

Dense improves test Recall@1 by 0.148 and MRR@10 by 0.116.

## 2. What the figures show

### `overall_metrics.png`
This figure compares BM25 and dense retrieval on dev and test. Dense is consistently ahead on every metric, with the biggest gain on Recall@1 and MRR@10.

### `slice_comparison_dev.png` and `slice_comparison_test.png`
These figures break performance down by question category and document length bucket. They show where the retrievers struggle and where dense retrieval closes the gap the most.

### `rank_distribution_dev.png` and `rank_distribution_test.png`
These figures show how often the gold document lands at rank 1 through 10, or misses the top 10 entirely. Dense shifts more queries into rank-1 hits and reduces the miss tail.

## 3. Question-category analysis

Dense retrieval is strongest on direct lookup and relation/detection questions, which are the bulk of the dataset. BM25 is acceptable there, but dense is more robust when the wording changes or when the question needs semantic matching.

The hardest bucket is rewrite_or_summarize. BM25 drops sharply there, while dense still retains a meaningful advantage. That is the clearest sign that semantic matching matters more than literal token overlap for the harder questions.

## 4. Document-length analysis

Short and medium documents are easier for both retrievers. Long documents are consistently harder, which is expected because they contain more surface noise and more competing concepts. Dense remains better than BM25, but the gap narrows on the long-document slice because long documents dilute the signal.

## 5. Failure patterns

### Biggest dense gains on DEV
- rewrite_or_summarize: +0.289 Recall@10
- doc::long: +0.110 Recall@10
- relation_or_detection: +0.043 Recall@10

### Biggest dense gains on TEST
- rewrite_or_summarize: +0.260 Recall@10
- doc::long: +0.120 Recall@10
- relation_or_detection: +0.044 Recall@10

## 6. Interpretation

- Retrieval is already strong overall: dense Recall@10 is above 0.93 on both dev and test.
- BM25 is a solid baseline, but dense consistently wins on every major metric.
- The real weakness is not the easy lookup slice; it is the rewrite/summarize and long-document cases.
- That means the retrieval stack is good enough for a first-stage RAG system, but the hardest questions will benefit from reranking, query rewriting, or a stronger instruction-tuned embedding model.

In the test split, dense Recall@10 is 0.933 overall versus 0.882 for BM25, and dense MRR@10 is 0.836 versus 0.720.

## 7. Conclusion

The current retriever is good, not perfect. It is very strong on exact or near-exact ATT&CK lookups, strong on relation-heavy questions, and noticeably weaker on answer-rewriting questions and long documents. Dense retrieval is the better default choice, but the remaining error profile is concentrated in the most semantically demanding questions.