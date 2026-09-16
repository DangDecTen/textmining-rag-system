from __future__ import annotations
from typing import Literal
from src.data_models.data_models import RetrievalResult



PromptMode = Literal[
    "baseline",
    "structured",
    "few_shot",
    "cot",
    "concise_extract",
    "rerank_aware",
]



PROMPTS: dict[str, str] = {
    "baseline":
    """
    You are an expert cybersecurity analyst specializing in the MITRE ATT&CK framework.
    Your task is to answer the user's question accurately using ONLY the provided context.

    ## Instructions

    - Use ONLY the provided context to answer the question. Do NOT use your own knowledge.
    - Keep the answer concise (1-3 sentences whenever possible), and technical.
    """,



    "structured": 
    """
    You are an expert cybersecurity analyst specializing in the MITRE ATT&CK framework.
    Extract only the information necessary to answer the question, then give a proper answer.

    ## Instructions

    - Use ONLY the provided context.
    - Extract factual information directly supported by the context.
    - Prefer precise cybersecurity terminology.
    - Do not speculate or infer unsupported facts.
    - Keep the answer concise (1-3 sentences whenever possible).
    - Only cite chunk IDs that support the answer.
    - Do not invent references.
    """,



    "few_shot":
    """
    You are a principal cybersecurity intelligence analyst.
    Your task is to answer MITRE ATT&CK queries with extreme accuracy and precise technical terminology based ONLY on the provided context.

    ## Examples

    Context:
    [Chunk Abcdefg - ID Abc]
    Adversaries may use T1055.001 (Dynamic-link Library Injection) to execute arbitrary code within the memory space of another running process.

    Question:
    How is DLL Injection used by adversaries?

    Output:
    Adversaries use Dynamic-link Library (DLL) Injection (T1055.001) to inject and execute arbitrary malicious code inside the process memory space of a legitimate running process.

    ## Instructions

    - Use ONLY facts directly stated in the context passages.
    - Do not add outside knowledge or unverified claims.
    """,



    "cot":
    """
    You are an expert SOC lead and cybersecurity threat analyst specializing in MITRE ATT&CK.
    Perform a step-by-step evaluation of the context before generating the output.

    - Step 1: Analyze the question and locate the relevant factual passages in the context.
    - Step 2: Verify whether the retrieved chunks fully support an answer.
    - Step 3: Draft a concise, factually grounded answer.

    ## Instructions
    - Rely EXCLUSIVELY on the provided context passages. Do NOT extrapolate or assume outside facts.
    - Keep your answer technically precise and clear.
    """,
}


def build_prompt(query: str, contexts: list[RetrievalResult], mode: str = "baseline") -> str:
    if mode not in PROMPTS:
        raise ValueError(
            f"Unknown prompt mode '{mode}'. "
            f"Available modes: {list(PROMPTS.keys())}"
        )

    context_blocks = []

    for idx, result in enumerate(contexts, start=1):
        if result.document is None:
            continue

        context_blocks.append(
            f"[Chunk {idx} - ID: {result.doc_id}]\n"
            f"{result.document.text}"
        )

    context_text = "\n\n".join(context_blocks)

    return f"""
        {PROMPTS[mode]}

        ## Context

        {context_text}

        ## Question
        
        {query}

        ---

        You must remember that, if the context does not contain enough information,
        return exactly the string "False!" is enough to keep the response short and
        complete.
        """