"""
Usage: python -m evaluation.retrieval.view_metrics
"""
import pandas as pd


# Get the calculated metrics from path
RETRIEVER = 'bm25'
SPLIT = 'dev'
GROUP_BY = 'document_len'  # source, human_question, question_len, document_len
RESULT_METRICS_PATH = "evaluation/retrieval/results/{}_{}_metrics_by_{}.jsonl".format(RETRIEVER, SPLIT, GROUP_BY)

# Report the following metrics
COLUMNS = ['group_by', 'group', 'n', 'mrr']


df = pd.read_json(RESULT_METRICS_PATH, lines=True)

# # View `source`
# source_description_types = ['software', 'techniques', 'groups', 'campaigns', 'tactics']
# print(df[
#             ~df['group'].isin(source_description_types)
#         ].round(3).sort_values(by='mrr', ascending=True)[
#             COLUMNS
#         ].reset_index(drop=True).to_markdown(index=True))


print(df[
            ~df['group'].isin(['overall'])
        ].round(3).sort_values(by='mrr', ascending=True)[
            COLUMNS
        ].reset_index(drop=True).to_markdown(index=True))