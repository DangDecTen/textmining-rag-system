# Main Report

Test with metrics
- MRR
- Recall @1, @5, @10
- Rerank with @10, @20, @50 candidates

Trade-off
- Does x% improvement worth the compute (model size) or time or memory (retrieved candidates) cost?

## BM25
Conclusion with:

```python
k1 = 1.0  # controls the impact of repeated terms, higher means term frequency matters more
b = 0.25  # controls document-length normalization
```

| k1 | b | mrr | recall@1 | recall@5 | recall@10 |
|-:|-:|-:|-:|-:|-:|
|1.0|0.25| 0.785| 0.694| 0.901| 0.942|
|1.0|0.75| 0.750| 0.669| 0.848| 0.900|
|2.0|0.25| 0.731| 0.628| 0.876| 0.928|
|2.0|0.75| 0.700| 0.609| 0.815| 0.878|

## Dense

Choosing model `BAAI/bge-small-en-v1.5` for inference, because small improvement not worth the embedding time on CPU.

| emb_model | device | mrr | recall@1 | recall@5 | recall@10 | avg_retrieve_ms |
|:--|:--|-:|-:|-:|-:|-:|
|`BAAI/bge-small-en-v1.5` |CPU | 0.847| 0.797| 0.914| 0.940| 86.635|
|`BAAI/bge-base-en-v1.5` |CPU | 0.863| 0.809| 0.932| 0.958| 228.226|
|`BAAI/bge-base-en-v1.5` |CUDA | 0.863| 0.809| 0.932| 0.958| 24.483|

## Hybrid

Use BM25 (`k1=1.0, b=0.25`) and Dense (`bge_small`). The implementation of Hybrid retriever comes with two type of score fusion, RRF and weighted-score. First start the experiment with RRF score (`user_rrf=True`).

### a. RRF Score

We tune `alpha` first to narrow the experiments down. We can see that good results in the range of `0.4-0.6`.

| use_rrf | alpha | rrf_k | mrr | recall@1 | recall@5 | recall@10 |
|:--|-:|-:|-:|-:|-:|-:|
|True | 0.2| 60| 0.834| 0.767| 0.921| 0.953|
|True | 0.4| 60| 0.854| 0.800| 0.928| 0.955|
|True | 0.5| 60| 0.860| 0.808| 0.930| 0.955|
|True | 0.6| 60| 0.861| 0.809| 0.931| 0.951|
|True | 0.8| 60| 0.857| 0.805| 0.924| 0.949|

The `rrf_k` control the impact of ranked documents, large value make the rank contribution flatter. We can see that smaller value yield better result.

| use_rrf | alpha | rrf_k | mrr | recall@1 | recall@5 | recall@10 |
|:--|-:|-:|-:|-:|-:|-:|
|True | 0.6|   5| 0.870| 0.818| 0.939| 0.965|
|True | 0.6|  10| 0.867| 0.813| 0.935| 0.962|
|True | 0.6|  20| 0.863| 0.810| 0.933| 0.955|
|True | 0.6|  40| 0.861| 0.809| 0.931| 0.951|
|True | 0.6|  60| 0.861| 0.809| 0.931| 0.951|
|True | 0.6| 100| 0.860| 0.809| 0.930| 0.950|

This is our final experiments for RRF scores. Also, we can notice that higher `alpha` use more 

| use_rrf | device | alpha | rrf_k | mrr | recall@1 | recall@5 | recall@10 | avg_retrieve_ms |
|:--|:--|-:|-:|-:|-:|-:|-:|-:|
|True |GPU | 0.4|   5| 0.868| 0.810| 0.947| 0.964|  22.462|
|True |GPU | 0.5|   5| 0.868| 0.810| 0.947| 0.964|  22.505|
|True |CPU | 0.6|   5| 0.870| 0.818| 0.939| 0.965| 151.950|

### b. Weighted Score

First tune the `alpha`.

| use_rrf | alpha | rrf_k | mrr | recall@1 | recall@5 | recall@10 |
|:--|-:|-:|-:|-:|-:|-:|
|False | 0.2|  5| 0.839| 0.771| 0.931| 0.960|
|False | 0.4|  5| 0.875| 0.822| 0.946| 0.968|
|False | 0.5|  5| 0.876| 0.823| 0.946| 0.968|
|False | 0.5| 10| 0.877| 0.825| 0.946| 0.968|
|False | 0.5| 20| 0.876| 0.824| 0.946| 0.968|
|False | 0.6|  5| 0.875| 0.827| 0.940| 0.965|
|False | 0.8|  5| 0.866| 0.821| 0.927| 0.951|

While changing the `rrf_k` does not make much differences.

| use_rrf | device | alpha | rrf_k | mrr | recall@1 | recall@5 | recall@10 | avg_retrieve_ms |
|:--|:--|-:|-:|-:|-:|-:|-:|-:|
|False |GPU | 0.5|  5| 0.876| 0.823| 0.946| 0.968|  22.669|
|False |CPU | 0.5| 10| 0.877| 0.825| 0.946| 0.968| 107.688|
|False |GPU | 0.5| 20| 0.876| 0.824| 0.946| 0.968|  22.669|

## Re-ranking

To use reranker, we must achieve the best recall in retriever before reranking. Review the recall both overall and for each question type.

- From the recall of both BM25 and Dense, recall@20 is a reasonable trade-off between recall (lowest is 0.714) and other costs (time and compute on CPU).
- Also, both BM25 and Dense had their strength in different types of questions (group by `source`), which showed that Hybrid retrieval is a good model. We may also build seperate models for different types of questions if needed, or analyze why some question types can be dealed with Dense, while other can be dealed with BM25.
- Experiments on Cross-encoder models were done on GPU, time on CPU might approximately 7x the GPU time.


| retriever | n | mrr@10 | recall@1 | recall@5 | recall@10 | avg_retrieve_ms | avg_rerank_ms | avg_ms | device | rerank_candidate_k |
|:--|-:|-:|-:|-:|-:|-:|-:|-:|:--|-:|
|bm25 | 2533| 0.785| 0.694| 0.901| 0.942| 6.141| 0.000| 6.141|CPU | 0|
|bm25+cross_encoder | 2533| 0.870| 0.818| 0.938| 0.942| 2.552| 178.375| 180.927|CUDA | 20|
|dense | 2533| 0.863| 0.809| 0.932| 0.958| 228.226| 0.000| 228.226|CPU | 0|
|dense+cross_encoder | 2533| 0.882| 0.830| 0.946| 0.956| 19.151| 333.667| 352.819|CUDA | 20|
|hybrid | 2533| 0.877| 0.825| 0.946| 0.968| 107.688| 0.000| 107.688|CPU | 0|
|hybrid+cross_encoder | 2533| 0.887| 0.826| 0.967| 0.978| 24.723| 415.301| 440.024|CUDA | 20|

#### a. BM25 - Metric Recall

|    | group_by   | group                                 |   n |   mrr@100 |   avg_retrieve_ms |   recall@5 |   recall@10 |   recall@20 |   recall@50 |   recall@100 |
|---:|:-----------|:--------------------------------------|----:|------:|------------------:|-----------:|------------:|------------:|------------:|-------------:|
|  0 | source     | relationships_groups_for_campaign     |   1 | 1     |             9.691 |      1     |       1     |       1     |       1     |        1     |
|  1 | source     | relationships_campaigns_for_group     |   1 | 1     |             9.844 |      1     |       1     |       1     |       1     |        1     |
|  2 | source     | relationships_software_for_campaign   |   3 | 0.5   |             7.919 |      1     |       1     |       1     |       1     |        1     |
|  3 | source     | relationships_techniques_for_campaign |   3 | 0.511 |             5.479 |      1     |       1     |       1     |       1     |        1     |
|  4 | source     | tactics                               |   5 | 0.5   |            19.151 |      1     |       1     |       1     |       1     |        1     |
|  5 | source     | relationships_campaigns_for_software  |   8 | 0.938 |             7.845 |      1     |       1     |       1     |       1     |        1     |
|  6 | source     | techniques_sub                        |   9 | 1     |            14.04  |      1     |       1     |       1     |       1     |        1     |
|  7 | source     | campaigns                             |  10 | 1     |             7.682 |      1     |       1     |       1     |       1     |        1     |
|  8 | source     | relationships_techniques_for_group    |  14 | 0.548 |             9.672 |      0.643 |       0.643 |       0.714 |       0.857 |        1     |
|  9 | source     | relationships_software_for_group      |  14 | 0.964 |             6.425 |      1     |       1     |       1     |       1     |        1     |
| 10 | source     | relationships_campaigns_for_technique |  22 | 0.693 |             9.942 |      0.909 |       1     |       1     |       1     |        1     |
| 11 | source     | relationships_groups_for_technique    |  42 | 0.624 |             9.688 |      0.833 |       0.905 |       0.976 |       1     |        1     |
| 12 | source     | relationships_software_for_technique  |  43 | 0.662 |             9.645 |      0.837 |       0.884 |       0.884 |       0.884 |        0.93  |
| 13 | source     | techniques_parent                     |  44 | 0.955 |             9.82  |      1     |       1     |       1     |       1     |        1     |
| 14 | source     | relationships_groups_for_software     |  50 | 0.916 |            10.787 |      0.98  |       1     |       1     |       1     |        1     |
| 15 | source     | groups                                |  55 | 0.895 |             7.38  |      0.982 |       0.982 |       0.982 |       1     |        1     |
| 16 | source     | techniques_tactics                    |  64 | 0.54  |            14.54  |      0.953 |       0.984 |       1     |       1     |        1     |
| 17 | source     | relationships_techniques_for_software |  68 | 0.638 |             9.003 |      0.853 |       0.897 |       0.926 |       0.971 |        0.985 |
| 18 | source     | relationships_mitigations_summaries   | 131 | 0.427 |             9.853 |      0.84  |       0.901 |       0.954 |       0.985 |        0.985 |
| 19 | source     | relationships_detections_summaries    | 143 | 0.684 |             9.739 |      0.867 |       0.93  |       0.972 |       0.972 |        0.993 |
| 20 | source     | techniques                            | 214 | 0.713 |            10.358 |      0.804 |       0.907 |       0.939 |       0.977 |        0.995 |
| 21 | source     | relationships_mitigations             | 222 | 0.671 |            11.207 |      0.829 |       0.887 |       0.941 |       0.977 |        0.986 |
| 22 | source     | relationships_detects                 | 253 | 0.685 |             9.923 |      0.842 |       0.909 |       0.945 |       0.972 |        0.992 |
| 23 | source     | software                              | 261 | 0.697 |             9.242 |      0.789 |       0.877 |       0.966 |       0.992 |        1     |
| 24 | source     | relationships_uses_software           | 853 | 0.99  |            11.456 |      1     |       1     |       1     |       1     |        1     |

#### b. Dense - Metric Recall

|    | group_by   | group                                 |   n |   mrr@100 |   avg_retrieve_ms |   recall@5 |   recall@10 |   recall@20 |   recall@50 |   recall@100 |
|---:|:-----------|:--------------------------------------|----:|------:|------------------:|-----------:|------------:|------------:|------------:|-------------:|
|  0 | source     | relationships_groups_for_campaign     |   1 | 1     |            18.434 |      1     |       1     |       1     |       1     |        1     |
|  1 | source     | relationships_campaigns_for_group     |   1 | 1     |            16.608 |      1     |       1     |       1     |       1     |        1     |
|  2 | source     | relationships_software_for_campaign   |   3 | 1     |            22.121 |      1     |       1     |       1     |       1     |        1     |
|  3 | source     | relationships_techniques_for_campaign |   3 | 0.567 |            16.817 |      1     |       1     |       1     |       1     |        1     |
|  4 | source     | tactics                               |   5 | 0.9   |            17.446 |      1     |       1     |       1     |       1     |        1     |
|  5 | source     | relationships_campaigns_for_software  |   8 | 1     |            21.59  |      1     |       1     |       1     |       1     |        1     |
|  6 | source     | techniques_sub                        |   9 | 0.87  |            18.869 |      1     |       1     |       1     |       1     |        1     |
|  7 | source     | campaigns                             |  10 | 0.9   |            21.389 |      1     |       1     |       1     |       1     |        1     |
|  8 | source     | relationships_techniques_for_group    |  14 | 0.618 |            17.613 |      0.714 |       0.786 |       1     |       1     |        1     |
|  9 | source     | relationships_software_for_group      |  14 | 0.891 |            17.173 |      0.929 |       1     |       1     |       1     |        1     |
| 10 | source     | relationships_campaigns_for_technique |  22 | 0.909 |            21.075 |      0.955 |       1     |       1     |       1     |        1     |
| 11 | source     | relationships_groups_for_technique    |  42 | 0.976 |            19.199 |      1     |       1     |       1     |       1     |        1     |
| 12 | source     | relationships_software_for_technique  |  43 | 0.934 |            19.437 |      0.977 |       1     |       1     |       1     |        1     |
| 13 | source     | techniques_parent                     |  44 | 0.989 |            20.022 |      1     |       1     |       1     |       1     |        1     |
| 14 | source     | relationships_groups_for_software     |  50 | 1     |            18.728 |      1     |       1     |       1     |       1     |        1     |
| 15 | source     | groups                                |  55 | 0.953 |            19.757 |      0.964 |       0.964 |       0.982 |       1     |        1     |
| 16 | source     | techniques_tactics                    |  64 | 0.797 |            22.761 |      0.938 |       0.969 |       1     |       1     |        1     |
| 17 | source     | relationships_techniques_for_software |  68 | 0.745 |            22.488 |      0.882 |       0.926 |       0.941 |       1     |        1     |
| 18 | source     | relationships_mitigations_summaries   | 131 | 0.889 |            20.544 |      0.954 |       0.962 |       0.977 |       0.985 |        0.985 |
| 19 | source     | relationships_detections_summaries    | 143 | 0.976 |            22.066 |      0.986 |       0.993 |       0.993 |       0.993 |        1     |
| 20 | source     | techniques                            | 214 | 0.658 |            22.58  |      0.738 |       0.799 |       0.836 |       0.925 |        0.935 |
| 21 | source     | relationships_mitigations             | 222 | 0.642 |            21.377 |      0.793 |       0.851 |       0.892 |       0.941 |        0.955 |
| 22 | source     | relationships_detects                 | 253 | 0.584 |            20.194 |      0.751 |       0.81  |       0.866 |       0.917 |        0.953 |
| 23 | source     | software                              | 261 | 0.879 |            19.749 |      0.935 |       0.969 |       0.992 |       1     |        1     |
| 24 | source     | relationships_uses_software           | 853 | 0.97  |            21.004 |      0.992 |       0.996 |       1     |       1     |        1     |

#### c. Hybrid - Metric Recall

|    | group_by   | group                                 |   n |   mrr@100 |   avg_retrieve_ms |   recall@5 |   recall@10 |   recall@20 |   recall@50 |   recall@100 |
|---:|:-----------|:--------------------------------------|----:|------:|------------------:|-----------:|------------:|------------:|------------:|-------------:|
|  0 | source     | relationships_groups_for_campaign     |   1 | 1     |            31.019 |      1     |       1     |       1     |       1     |        1     |
|  1 | source     | relationships_campaigns_for_group     |   1 | 1     |            21.586 |      1     |       1     |       1     |       1     |        1     |
|  2 | source     | relationships_software_for_campaign   |   3 | 0.833 |            33.031 |      1     |       1     |       1     |       1     |        1     |
|  3 | source     | relationships_techniques_for_campaign |   3 | 0.75  |            32.348 |      1     |       1     |       1     |       1     |        1     |
|  4 | source     | tactics                               |   5 | 0.867 |            27.016 |      1     |       1     |       1     |       1     |        1     |
|  5 | source     | relationships_campaigns_for_software  |   8 | 1     |            22.821 |      1     |       1     |       1     |       1     |        1     |
|  6 | source     | techniques_sub                        |   9 | 1     |            24.451 |      1     |       1     |       1     |       1     |        1     |
|  7 | source     | campaigns                             |  10 | 0.95  |            31.081 |      1     |       1     |       1     |       1     |        1     |
|  8 | source     | relationships_techniques_for_group    |  14 | 0.682 |            27.679 |      0.714 |       0.857 |       0.929 |       1     |        1     |
|  9 | source     | relationships_software_for_group      |  14 | 0.964 |            47.732 |      1     |       1     |       1     |       1     |        1     |
| 10 | source     | relationships_campaigns_for_technique |  22 | 0.964 |            25.694 |      1     |       1     |       1     |       1     |        1     |
| 11 | source     | relationships_groups_for_technique    |  42 | 0.976 |            29.491 |      1     |       1     |       1     |       1     |        1     |
| 12 | source     | relationships_software_for_technique  |  43 | 0.899 |            27.165 |      0.93  |       0.93  |       0.953 |       0.977 |        1     |
| 13 | source     | techniques_parent                     |  44 | 0.989 |            25.312 |      1     |       1     |       1     |       1     |        1     |
| 14 | source     | relationships_groups_for_software     |  50 | 1     |            28.009 |      1     |       1     |       1     |       1     |        1     |
| 15 | source     | groups                                |  55 | 0.97  |            27.269 |      0.982 |       1     |       1     |       1     |        1     |
| 16 | source     | techniques_tactics                    |  64 | 0.838 |            25.463 |      0.969 |       0.984 |       1     |       1     |        1     |
| 17 | source     | relationships_techniques_for_software |  68 | 0.832 |            26.181 |      0.897 |       0.926 |       0.941 |       0.985 |        1     |
| 18 | source     | relationships_mitigations_summaries   | 131 | 0.898 |            25.034 |      0.969 |       0.977 |       0.985 |       0.985 |        0.985 |
| 19 | source     | relationships_detections_summaries    | 143 | 0.947 |            26.4   |      0.979 |       0.993 |       0.993 |       1     |        1     |
| 20 | source     | techniques                            | 214 | 0.747 |            28.173 |      0.827 |       0.893 |       0.93  |       0.981 |        0.995 |
| 21 | source     | relationships_mitigations             | 222 | 0.701 |            27.698 |      0.856 |       0.932 |       0.973 |       0.991 |        0.995 |
| 22 | source     | relationships_detects                 | 253 | 0.666 |            26.56  |      0.85  |       0.905 |       0.957 |       0.984 |        0.996 |
| 23 | source     | software                              | 261 | 0.887 |            25.148 |      0.962 |       0.977 |       0.992 |       1     |        1     |
| 24 | source     | relationships_uses_software           | 853 | 0.99  |            27.638 |      1     |       1     |       1     |       1     |        1     |
