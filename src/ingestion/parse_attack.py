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

import requests
from stix2 import Filter, MemoryStore

MITRE_ATTACK_DOMAIN = "enterprise-attack"
MITRE_ATTACK_VERSION = "14.1"

TYPE_MAP = {
    "attack-pattern": "technique",
    "x-mitre-tactic": "tactic",
    "malware": "software",
    "tool": "software",
    "intrusion-set": "group",
    "campaign": "campaign",
    "course-of-action": "mitigation",
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


def parse_object_into_document(obj) -> Optional[dict]:
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

    return {
        "stix_id": obj.id,
        "attack_id": attack_id,
        "domain": domain,
        "stix_type": stix_type,
        "attack_type": attack_type,
        "name": name,
        "description": description,
        "text": f"{name}\n\n{description}",   # what gets embedded downstream
        "url": _get_url(obj),
        "tactics": _get_tactics(obj),
    }


def parse_all(thesrc: MemoryStore) -> list[dict]:
    objects = get_all_objects(thesrc)
    docs = [parse_object_into_document(obj) for obj in objects]
    return [d for d in docs if d is not None]


def save_docs(docs: list[dict], out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")


if __name__ == "__main__":
    src = get_data_from_version(MITRE_ATTACK_DOMAIN, MITRE_ATTACK_VERSION)
    docs = parse_all(src)
    save_docs(docs, "data/processed/attack_docs.jsonl")
    print(f"Parsed {len(docs)} ATT&CK objects -> data/processed/attack_docs.jsonl")