"""
Parses the model's raw text output into (answer, found). Kept separate from
the Generator so it's independently testable -- a 1.5B model WILL
occasionally produce malformed JSON, extra prose around the JSON, or nothing
parseable at all, and we need well-defined, fail-safe behavior for that:
when parsing fails, we treat it as an abstention (found=False) rather than
risk surfacing an ungrounded answer. In a cybersecurity QA tool, "I don't
know" is a much cheaper mistake than a confident wrong answer.
"""
from __future__ import annotations

import json
import re

_JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def parse_structured_output(raw_output: str) -> tuple[str, bool]:
    match = _JSON_PATTERN.search(raw_output)
    if not match:
        return "", False

    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return "", False

    found = bool(obj.get("found", False))
    answer = str(obj.get("answer", "") or "").strip()

    if not found or not answer:
        return "", False
    return answer, True
