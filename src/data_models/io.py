import json
from typing import Literal
from src.config import settings
from src.data_models.data_models import Document, QAExample


def load_corpus_lookup(path: str | None = None) -> dict[str, Document]:
    """path defaults to settings.corpus_path (see src/config.py)."""
    path = path or settings.corpus_path
    corpus = {}
    with open(path) as f:
        for line in f:
            doc = Document(**json.loads(line))
            corpus[doc.doc_id] = doc
    return corpus


def load_qa_examples(split: Literal["train", "dev", "test"], path_template: str | None = None) -> list[QAExample]:
    """path_template defaults to settings.qa_path_template (see src/config.py)."""
    path = (path_template or settings.qa_path_template).format(split=split)
    with open(path) as f:
        return [QAExample(**json.loads(line)) for line in f]