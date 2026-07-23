from pydantic import BaseModel


class Document(BaseModel):
    """A single, atomic retrievable unit in the corpus.

    For stage01 a Document *is* the retrieval unit (no further chunking):
    AttackQA's `document` field is already a short, semantically atomic
    snippet (e.g. one detection note, one technique description paragraph).
    """

    doc_id: str  # sha1(text)[:16] -- stable, content-derived id
    text: str

    subject_id: str
    subject_name: str | None = None
    subject_type: str | None = None
    source: str
    field: str | None = None
    relation_id: str | None = None
    relation_name: str | None = None
    url: str
    references: list[dict] | None = None


class QAExample(BaseModel):
    """One question/answer pair with pointer(s) to the ground-truth document(s).

    `relevant_doc_ids` is a list (rather than a single id) so the schema
    doesn't need to change if a future dataset version has multi-document
    support. For AttackQA today, expect exactly one id per example.
    """

    qa_id: str
    question: str
    answer: str
    thought: str | None = None
    relevant_doc_ids: list[str]
    source: str
    human_question: bool = False
    human_answer: bool = False
 

class RetrievalResult(BaseModel):
    doc_id: str
    score: float
    document: Document | None = None


class GenerationResult(BaseModel):
    """Output produced by a Generator."""

    answer: str

    # Whether the generator located the answer in the provided context, vs.
    # abstained.
    found: bool

    # Prompt actually sent to the LLM.
    # Useful for debugging, evaluation, and reproducibility.
    prompt: str

    # Retrieved documents that were provided as context.
    retrieval_results: list[RetrievalResult]

    # Generation statistics.
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class Citation(BaseModel):
    doc_id: str

    subject_id: str
    subject_name: str | None = None
    source: str
    field: str | None = None
    relation_id: str | None = None
    relation_name: str | None = None
    url: str
    references: list[dict] | None = None


class Answer(BaseModel):
    """Final, UI-facing response -- debug fields (prompt, latency, raw retrieval
    scores) are deliberately left behind in GenerationResult."""

    text: str
    abstained: bool
    citations: list[Citation]
