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

from groq import Groq
from transformers import AutoTokenizer

from src.data_models.data_models import GenerationResult, RetrievalResult
from src.generation.base import Generator
from src.generation.context_builder import ContextBuilder
from src.generation.output_parser import parse_structured_output
from src.generation.prompt import SYSTEM_PROMPT, PROMPTS, build_user_message
from app.backend.app_config import GROQ_API_KEY


class GroqGenerator(Generator):
    """Generator that sends selectable prompt strategy to a Groq-hosted model."""

    def __init__(
        self,
        model_name: str,
        prompt_mode: str = "baseline",
        max_context_tokens: int = 1500,
        max_new_tokens: int = 128,
    ):
        if prompt_mode not in PROMPTS:
            raise ValueError(
                f"Unknown prompt mode '{prompt_mode}'. "
                f"Available: {sorted(PROMPTS)}"
            )

        self.client = Groq(api_key=GROQ_API_KEY)
        self.model_name = model_name
        self.prompt_mode = prompt_mode
        self.max_new_tokens = max_new_tokens
        
        # For now, use a lightweight tokenizer appropriate for the selected
        # Groq model. This keeps the app from downloading multi-GB models.
        if "qwen" in model_name.lower():
            tokenizer_name = "Qwen/Qwen2.5-1.5B-Instruct"
        else:
            tokenizer_name = "unsloth/Llama-3.3-70B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.context_builder = ContextBuilder(
            self.tokenizer,
            max_context_tokens=max_context_tokens,
        )

    def generate(
        self,
        question: str,
        contexts: list[RetrievalResult],
    ) -> GenerationResult:
        start = time.time()

        context_block = self.context_builder.build(contexts)

        # Use the selected app prompt instead of the shared SYSTEM_PROMPT.
        system_prompt = PROMPTS[self.prompt_mode]

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": build_user_message(
                    question,
                    context_block,
                ),
            },
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        prompt_tokens = len(
            self.tokenizer.encode(prompt_text)
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.2,
            #max_completion_tokens=self.max_new_tokens,
            max_completion_tokens=512,
            response_format={"type": "json_object"}
        )

        raw_output = response.choices[0].message.content or ""

        # Some reasoning-capable models may include a thinking section.
        if "</think>" in raw_output:
            raw_output = raw_output.split("</think>")[-1].strip()

        answer, found = parse_structured_output(raw_output)

        latency_ms = (time.time() - start) * 1000

        # Prefer provider-reported token counts when available.
        usage = getattr(response, "usage", None)

        response_prompt_tokens = (
            getattr(usage, "prompt_tokens", None)
            if usage is not None
            else None
        )

        response_completion_tokens = (
            getattr(usage, "completion_tokens", None)
            if usage is not None
            else None
        )

        return GenerationResult(
            answer=answer,
            found=found,
            prompt=prompt_text,
            retrieval_results=contexts,
            latency_ms=latency_ms,
            prompt_tokens=(
                response_prompt_tokens
                if response_prompt_tokens is not None
                else prompt_tokens
            ),
            completion_tokens=(
                response_completion_tokens
                if response_completion_tokens is not None
                else len(self.tokenizer.encode(raw_output))
            ),
        )
