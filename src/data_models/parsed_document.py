from dataclasses import dataclass

@dataclass(slots=True)
class ParsedDocument:
    stix_id: str
    attack_id: str
    domain: str
    stix_type: str
    attack_type: str
    name: str
    description: str
    text: str
    url: str | None = None
    tactics: list[str] | None = None
    related_context: list[str] = []