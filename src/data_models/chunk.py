from dataclasses import dataclass

@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    attack_domain: str
    attack_type: str
    name: str
    text: str
    url: str | None = None