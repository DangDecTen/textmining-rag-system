"""
Groq-backed generator for the application.
The application can expose the same user-facing choices:
    llama
    qwen
but both are executed remotely through Groq instead of loading a local
Hugging Face generation model.

Input:
    list[RetrievalResult]

Output:
    GenerationResult
"""

from __future__ import annotations

import os
import time
import json

from groq import Groq

from src.data_models.data_models import GenerationResult, RetrievalResult
from src.generation.base import Generator
from src.generation.prompt import build_prompt, PromptMode, PROMPTS
from src.generation.registry import register_generator
from app.backend.app_config import GROQ_API_KEY


@register_generator("groq")
class GroqGenerator(Generator):
    """Generator that sends selectable prompt strategy to a Groq-hosted model."""

    def __init__(
        self,
        model_name: str,
        prompt_mode: PromptMode = "baseline",
        max_new_tokens: int = 350,
    ):
        if prompt_mode not in PROMPTS:
            raise ValueError(
                f"Unknown prompt mode '{prompt_mode}'. "
                f"Available: {list(PROMPTS.keys())}"
            )
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not configured.")

        self.client = Groq(api_key=GROQ_API_KEY)
        self.model_name = model_name
        self.prompt_mode = prompt_mode
        self.max_new_tokens = max_new_tokens


    def generate(
        self,
        question: str,
        contexts: list[RetrievalResult],
    ) -> GenerationResult:
        start = time.time()

        prompt = build_prompt(
            query=question,
            contexts=contexts,
            mode=self.prompt_mode,
        )

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.2,
            max_completion_tokens=self.max_new_tokens,
        )


        content = response.choices[0].message.content or ""
        content = content.strip()

        # Some reasoning-capable models may include a thinking section.
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()

        try:
            result = json.loads(content)

        except json.JSONDecodeError:
            # The model violated the JSON-only requirement.
            # Return the raw response as the answer rather than crashing the whole request.
            result = {
                "answer": content,
                "found": False,
                "references": [],
            }

        answer = result.get("answer", "")
        found = result.get("found", False)
        references = result.get("references", [])

        # Ensure expected types.
        if not isinstance(answer, str):
            answer = str(answer)
        if not isinstance(found, bool):
            found = bool(found)
        if not isinstance(references, list):
            references = []
        references = [str(ref) for ref in references]

        latency_ms = (time.time() - start) * 1000

        # Prefer provider-reported token counts when available.
        usage = getattr(response, "usage", None)

        prompt_tokens = (
            getattr(usage, "prompt_tokens", None)
            if usage is not None
            else None
        )

        completion_tokens = (
            getattr(usage, "completion_tokens", None)
            if usage is not None
            else None
        )

        return GenerationResult(
            answer=answer,
            found=found,
            prompt=prompt,
            retrieval_results=contexts,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )
