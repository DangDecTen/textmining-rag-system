import json
from typing import Literal
from src.data_models.data_models import Document, QAExample


CORPUS_DIR = "data/processed/corpus.jsonl"
QA_DIR = "data/processed/qa_{split}.jsonl"


def load_corpus_lookup() -> dict[str, Document]:
    path = CORPUS_DIR
    corpus = {}
    with open(path) as f:
        for line in f:
            doc = Document(**json.loads(line))
            corpus[doc.doc_id] = doc
    return corpus
 
 
def load_qa_examples(split: Literal["train", "dev", "test"]) -> list[QAExample]:
    path = QA_DIR.format(split=split)
    with open(path) as f:
        return [QAExample(**json.loads(line)) for line in f]