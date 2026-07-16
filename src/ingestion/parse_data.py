"""
Parse the MITRE ATT&CK Enterprise STIX bundle into flat, retrievable
documents, using the official `stix2` library (MemoryStore + Filter)
instead of hand-parsing raw JSON.

Pulls directly from the versioned attack-stix-data GitHub repo, so the
corpus is pinned and reproducible across the team / across Colab sessions.
"""

import json
from itertools import chain
from pathlib import Path
from typing import Optional
from src.data_models.parsed_document import ParsedDocument

import requests
from stix2 import Filter, MemoryStore

MITRE_ATTACK_DOMAIN = "enterprise-attack"
MITRE_ATTACK_VERSION = "14.1"
MAX_RELATED_PER_LABEL = 15  # cap so a heavily-used technique (e.g. T1059)
                            # doesn't blow up its chunk with 200 group names

TYPE_MAP = {
    "attack-pattern": "technique",
    "x-mitre-tactic": "tactic",
    "malware": "software",
    "tool": "software",
    "intrusion-set": "group",
    "campaign": "campaign",
    "course-of-action": "mitigation",
}

ENRICHMENT_RULES = {
    ("incoming", "uses"): "Used by",
    ("outgoing", "uses"): "Uses",
    ("incoming", "mitigates"): "Mitigated by",
    ("outgoing", "mitigates"): "Mitigates",
    ("incoming", "subtechnique-of"): "Parent technique of",
    ("outgoing", "subtechnique-of"): "Sub-technique of",
    ("incoming", "attributed-to"): "Activities",
    ("outgoing", "attributed-to"): "Attributed to",
}



def get_data_from_version(domain: str, version: str) -> MemoryStore:
    """Fetch the ATT&CK STIX bundle for a given version from GitHub.

    domain: 'enterprise-attack', 'mobile-attack', or 'ics-attack'
    """
    url = f"https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/{domain}/{domain}-{version}.json"
    stix_json = requests.get(url).json()
    return MemoryStore(stix_data=stix_json["objects"])


def get_software(thesrc: MemoryStore) -> list:
    return list(chain.from_iterable(
        thesrc.query(f) for f in [
            Filter("type", "=", "tool"),
            Filter("type", "=", "malware"),
        ]
    ))


def get_all_objects(thesrc: MemoryStore) -> list:
    """Query every ATT&CK object type we care about into one flat list."""
    techniques = thesrc.query([Filter("type", "=", "attack-pattern")])
    tactics = thesrc.query([Filter("type", "=", "x-mitre-tactic")])
    groups = thesrc.query([Filter("type", "=", "intrusion-set")])
    campaigns = thesrc.query([Filter("type", "=", "campaign")])
    mitigations = thesrc.query([Filter("type", "=", "course-of-action")])
    softwares = get_software(thesrc)

    return techniques + tactics + groups + campaigns + mitigations + softwares


def _get_attack_id(obj) -> Optional[str]:
    for ref in getattr(obj, "external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id")
    return None


def _get_url(obj) -> Optional[str]:
    for ref in getattr(obj, "external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("url")
    return None


def _get_tactics(obj) -> Optional[list]:
    phases = getattr(obj, "kill_chain_phases", None)
    if not phases:
        return None
    return [p["phase_name"] for p in phases if p.get("kill_chain_name") == "mitre-attack"]


def parse_object_into_document(obj) -> ParsedDocument | None:
    """Turn a single stix2 object into a flat document dict.

    Returns None for objects we should skip: revoked/deprecated entries,
    or objects missing a stable ATT&CK ID (a handful of STIX objects,
    e.g. some relationship-adjacent types, don't carry one).
    """
    if getattr(obj, "revoked", False) or getattr(obj, "x_mitre_deprecated", False):
        return None

    attack_id = _get_attack_id(obj)
    if attack_id is None:
        return None

    stix_type = obj.type
    attack_type = TYPE_MAP.get(stix_type)
    if attack_type is None:
        return None

    # Sub-technique detection: ATT&CK IDs with a dot, e.g. T1059.001
    if attack_type == "technique" and "." in attack_id:
        attack_type = "sub-technique"

    name = getattr(obj, "name", "")
    description = getattr(obj, "description", "")
    domain = obj.x_mitre_domains[0] if getattr(obj, "x_mitre_domains", None) else MITRE_ATTACK_DOMAIN

    return ParsedDocument(
        stix_id=obj.id,
        attack_id=attack_id,
        domain=domain,
        stix_type=stix_type,
        attack_type=attack_type,
        name=name,
        description=description,
        text=f"{name}\n\n{description}",
        url=_get_url(obj),
        tactics=_get_tactics(obj),
    )


# --- Relationship enrichment -------------------------------------------
#
# Raw name/description text misses cross-object context that many AttackQA
# questions actually ask about, e.g. "what mitigates T1059?" or "which
# groups use spearphishing?". We resolve and append this context here so a
# single chunk of text is self-contained enough to answer
# relationship-style questions without needing multi-hop retrieval.


def _build_name_lookup(thesrc: MemoryStore) -> dict:
    """stix_id -> display name, for every object in the store (not just
    the types we keep as documents) so relationship targets like
    data-components still resolve to a readable name if ever needed."""
    all_objects = thesrc.query()
    lookup = {}
    for obj in all_objects:
        name = getattr(obj, "name", None)
        if name:
            lookup[obj.id] = name
    return lookup


def _build_relationship_index(thesrc: MemoryStore) -> tuple[dict, dict]:
    """Return (outgoing, incoming) indices keyed by stix_id.

    outgoing[stix_id]  -> list of (relationship_type, target_id)
    incoming[stix_id]  -> list of (relationship_type, source_id)
    Revoked/deprecated relationships are skipped.
    """
    relationships = thesrc.query([Filter("type", "=", "relationship")])
    outgoing, incoming = {}, {}
    for rel in relationships:
        if getattr(rel, "revoked", False) or getattr(rel, "x_mitre_deprecated", False):
            continue
        rel_type = rel.relationship_type
        outgoing.setdefault(rel.source_ref, []).append((rel_type, rel.target_ref))
        incoming.setdefault(rel.target_ref, []).append((rel_type, rel.source_ref))
    return outgoing, incoming


def _enrichment_lines(doc: ParsedDocument, outgoing: dict, incoming: dict, name_lookup: dict) -> list[str]:
    lines = []
    directions = [("outgoing", outgoing.get(doc.stix_id, [])),
                  ("incoming", incoming.get(doc.stix_id, []))]

    # group by label so e.g. multiple "uses" edges become one "Uses: a, b, c" line
    grouped: dict[str, list[str]] = {}
    for direction, edges in directions:
        for rel_type, other_id in edges:
            rule_key = (direction, rel_type)
            label = ENRICHMENT_RULES.get(rule_key)
            if label is None:
                continue
            other_name = name_lookup.get(other_id)
            if other_name is None:
                continue
            grouped.setdefault(label, []).append(other_name)

    for label, names in grouped.items():
        names = sorted(set(names))
        shown = names[:MAX_RELATED_PER_LABEL]
        suffix = f" (+{len(names) - MAX_RELATED_PER_LABEL} more)" if len(names) > MAX_RELATED_PER_LABEL else ""
        lines.append(f"{label}: {', '.join(shown)}{suffix}")

    return lines


def enrich_with_relationships(thesrc: MemoryStore, docs: list[ParsedDocument]) -> list[ParsedDocument]:
    """Append relationship-derived context to each doc's `text` field."""
    name_lookup = _build_name_lookup(thesrc)
    outgoing, incoming = _build_relationship_index(thesrc)

    for doc in docs:
        lines = _enrichment_lines(doc, outgoing, incoming, name_lookup)
        if lines:
            doc.related_context = lines
            doc.text = doc.text + "\n\n" + "\n".join(lines)
        else:
            doc.related_context = []

    return docs


def parse_all(thesrc: MemoryStore) -> list[ParsedDocument]:
    objects = get_all_objects(thesrc)
    docs = [parse_object_into_document(obj) for obj in objects]
    docs = [d for d in docs if d is not None]
    docs = enrich_with_relationships(thesrc, docs)
    return docs


def save_docs(docs: list[ParsedDocument], out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")


if __name__ == "__main__":
    src = get_data_from_version(MITRE_ATTACK_DOMAIN, MITRE_ATTACK_VERSION)
    docs = parse_all(src)
    save_docs(docs, "data/processed/attack_docs.jsonl")
    print(f"Parsed {len(docs)} ATT&CK objects -> data/processed/attack_docs.jsonl")