"""
Usage: python -m evaluation.retrieval.view_metrics
"""
import pandas as pd

from evaluation.retrieval.naming import run_name

# Get the calculated metrics from path
RETRIEVER = 'bm25'
RERANKER = None  # e.g. 'cross_encoder' to view a reranked run; None for retrieval-only
SPLIT = 'dev'
GROUP_BY = 'source'  # source, human_question, question_len, document_len
RESULT_METRICS_PATH = "evaluation/retrieval/results/{}_{}_metrics_by_{}.jsonl".format(
    run_name(RETRIEVER, RERANKER), SPLIT, GROUP_BY
)

# Report the following metrics
RECALL_K = [5, 10, 20, 50, 100]
RECALL_COLS = [f"recall@{k}" for k in RECALL_K]
COLUMNS = ['group_by', 'group', 'n', 'mrr', 'avg_retrieve_ms']
COLUMNS.extend(RECALL_COLS)


df = pd.read_json(RESULT_METRICS_PATH, lines=True)

# View `source`
# source_description_types = ['software', 'techniques', 'groups', 'campaigns', 'tactics']
# print(df[
#             ~df['group'].isin(source_description_types)
#         ].round(3).sort_values(by='mrr', ascending=True)[
#             COLUMNS
#         ].reset_index(drop=True).to_markdown(index=True))


print(df[
            ~df['group'].isin(['overall'])
        ].round(3).sort_values(by='n', ascending=True)[
            COLUMNS
        ].reset_index(drop=True).to_markdown(index=True))