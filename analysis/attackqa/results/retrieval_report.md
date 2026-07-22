# AttackQA Retriever Evaluation

This report evaluates the saved BM25 and dense indexes on the AttackQA dev/test QA splits.

## Overall Metrics

| Split | Retriever | Queries | Recall@1 | Recall@5 | Recall@10 | MRR@10 | Mean Rank | Median Rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dev | bm25 | 2533 | 0.639 | 0.831 | 0.886 | 0.724 | 2.84 | 1.00 |
| dev | dense | 2533 | 0.797 | 0.914 | 0.940 | 0.847 | 1.99 | 1.00 |
| test | bm25 | 2534 | 0.636 | 0.824 | 0.882 | 0.720 | 2.91 | 1.00 |
| test | dense | 2534 | 0.784 | 0.906 | 0.933 | 0.836 | 2.07 | 1.00 |

## Slice Metrics

### DEV - bm25

| Slice | Count | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| relation_or_detection | 1654 | 0.666 | 0.863 | 0.904 | 0.752 |
| direct_lookup | 665 | 0.689 | 0.869 | 0.934 | 0.771 |
| rewrite_or_summarize | 142 | 0.077 | 0.282 | 0.458 | 0.180 |
| reasoning_translation | 71 | 0.648 | 0.817 | 0.887 | 0.722 |
| other | 1 | 1.000 | 1.000 | 1.000 | 1.000 |

### DEV - dense

| Slice | Count | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| relation_or_detection | 1654 | 0.820 | 0.927 | 0.947 | 0.866 |
| direct_lookup | 665 | 0.833 | 0.944 | 0.967 | 0.880 |
| rewrite_or_summarize | 142 | 0.465 | 0.655 | 0.746 | 0.547 |
| reasoning_translation | 71 | 0.563 | 0.831 | 0.901 | 0.669 |
| other | 1 | 1.000 | 1.000 | 1.000 | 1.000 |

### TEST - bm25

| Slice | Count | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| relation_or_detection | 1654 | 0.660 | 0.861 | 0.905 | 0.747 |
| direct_lookup | 654 | 0.700 | 0.856 | 0.927 | 0.774 |
| rewrite_or_summarize | 146 | 0.082 | 0.240 | 0.377 | 0.159 |
| reasoning_translation | 78 | 0.615 | 0.872 | 0.949 | 0.729 |
| other | 2 | 1.000 | 1.000 | 1.000 | 1.000 |

### TEST - dense

| Slice | Count | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| relation_or_detection | 1654 | 0.817 | 0.932 | 0.949 | 0.865 |
| direct_lookup | 654 | 0.818 | 0.942 | 0.968 | 0.870 |
| rewrite_or_summarize | 146 | 0.322 | 0.493 | 0.637 | 0.393 |
| reasoning_translation | 78 | 0.667 | 0.846 | 0.872 | 0.746 |
| other | 2 | 1.000 | 1.000 | 1.000 | 1.000 |

## Interpretation

- `Recall@k` tells us whether the gold document appears in the top-k retrieved results.
- `MRR@10` rewards the system for ranking the correct document earlier.
- Slice tables show whether the retriever is systematically better on direct lookup questions, relation-heavy questions, or long-document cases.

The full per-query outputs are saved beside the report for manual error analysis.