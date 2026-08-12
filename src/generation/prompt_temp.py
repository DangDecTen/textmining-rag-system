from __future__ import annotations
from typing import Literal
from src.data_models.data_models import RetrievalResult

PromptMode = Literal["baseline", "structured", "evidence"]

PROMPTS: dict[PromptMode, str] = {
    "baseline": 
    """
    You are a cybersecurity assistant specializing in the MITRE ATT&CK framework.
    Your task is to answer the user's question using ONLY the provided context.

    Instructions:
    - Use ONLY the provided context.
    - Do NOT use outside knowledge.
    - If the context does not contain enough information, answer exactly:
    "I do not have enough information from the provided context."
    - Keep the answer concise and technically accurate.
    - Only cite chunk IDs that appear in the provided context.
    - Do not invent references.

    Return ONLY valid JSON in the following format:
    {
        "answer": "<answer>",
        "references": [
            "<chunk_id>"
        ]
    }
    """,


    "structured": 
    """
    You are an expert cybersecurity analyst specializing in the MITRE ATT&CK framework.
    Extract only the information necessary to answer the question.

    Instructions:
    - Use ONLY the provided context.
    - Extract factual information directly supported by the context.
    - Prefer precise cybersecurity terminology.
    - Do not speculate or infer unsupported facts.
    - Keep the answer concise (1-3 sentences whenever possible).
    - If the context does not contain enough information, answer exactly:
    "I do not have enough information from the provided context."
    - Only cite chunk IDs that support the answer.
    - Do not invent references.

    Return ONLY valid JSON in the following format:

    {
        "answer": "<answer>",
        "references": [
            "<chunk_id>"
        ]
    }
    """,


    "evidence": 
    """
    You are an expert cybersecurity analyst specializing in the MITRE ATT&CK framework.
    Every factual statement MUST be supported by the provided context.

    Instructions:
    - Use ONLY the provided context.
    - Never use outside knowledge.
    - Every statement in the answer must be supported by one or more cited chunks.
    - Cite every chunk that contributes evidence.
    - Only cite chunk IDs appearing in the provided context.
    - Do not invent references.
    - If the context does not contain enough information, answer exactly:
    "I do not have enough information from the provided context."

    Before producing the final answer, verify that every statement is supported by the cited references.
    Return ONLY valid JSON in the following format:

    {
        "answer": "<answer>",
        "references": [
            "<chunk_id>"
        ]
    }
    """,
}


def build_prompt(query: str, contexts: list[RetrievalResult], mode: PromptMode,) -> str:
    if mode not in PROMPTS:
        raise ValueError(
            f"Unknown prompt mode '{mode}'. "
            f"Available modes: {list(PROMPTS.keys())}"
        )

    context_blocks = []

    for result in contexts:
        if result.document is None:
            continue

        context_blocks.append(
            f"[{result.doc_id}]\n"
            f"{result.document.text}"
        )

    context_text = "\n\n".join(context_blocks)

    return f"""{PROMPTS[mode]}

========================
Context
========================

{context_text}

========================
Question
========================

{query}

Return ONLY the JSON object.
Do not include markdown.
Do not wrap the JSON in ``` blocks.
Do not provide any explanation before or after the JSON.
"""