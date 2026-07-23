from src.data_models.data_models import Answer, Citation, GenerationResult

ABSTAIN_MESSAGE = "I don't have enough information in the knowledge base to answer this question."


class ResponseBuilder:
    def build(self, generation_result: GenerationResult) -> Answer:
        if not generation_result.found:
            return Answer(text=ABSTAIN_MESSAGE, abstained=True, citations=[])

        citations = []
        seen_doc_ids = set()
        for r in generation_result.retrieval_results:
            if r.document is None or r.doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(r.doc_id)
            citations.append(
                Citation(
                    doc_id=r.doc_id,
                    subject_id=r.document.subject_id,
                    subject_name=r.document.subject_name,
                    source=r.document.source,
                    field=r.document.field,
                    relation_id=r.document.relation_id,
                    relation_name=r.document.relation_name,
                    url=r.document.url,
                    references=r.document.references,
                )
            )

        return Answer(text=generation_result.answer, citations=citations, abstained=False)
