from __future__ import annotations

from typing import Final

from src.generation.prompt import PROMPTS

APP_PROMPT_LABELS: Final[dict[str, str]] = {
    "baseline": "Baseline",
    "structured": "Structured",
    "evidence": "Evidence-focused",
    "cot_verification": "Verification",
    "few_shot_analyst": "Few-shot Analyst",
    "concise_extract": "Concise Extraction",
    "rerank_aware": "Rerank-aware",
}


def available_prompts() -> list[str]:
    return sorted(PROMPTS.keys())


def prompt_label(name: str) -> str:
    return APP_PROMPT_LABELS.get(
        name,
        name.replace("_", " ").title(),
    )
