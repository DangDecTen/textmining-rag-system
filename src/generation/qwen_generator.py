"""
Generator implementation using Qwen2.5-1.5B-Instruct via raw `transformers`
(no inference framework like vLLM/TGI -- overkill for a single free-tier GPU,
and consistent with using lower-level components elsewhere in this project).

Greedy decoding (do_sample=False): for closed-domain factoid QA with a strict
output format, we want deterministic, reproducible output, not diversity.
"""
from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data_models.data_models import GenerationResult
from src.generation.base import Generator
from src.generation.context_builder import ContextBuilder
from src.generation.output_parser import parse_structured_output
from src.generation.prompt import SYSTEM_PROMPT, build_user_message


class QwenGenerator(Generator):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str | None = None,
        max_context_tokens: int = 1500,
        max_new_tokens: int = 128,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

        self.context_builder = ContextBuilder(self.tokenizer, max_context_tokens=max_context_tokens)

    def generate(self, question: str, contexts: list) -> GenerationResult:
        start = time.time()

        context_block = self.context_builder.build(contexts)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(question, context_block)},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        completion_ids = output_ids[0][prompt_tokens:]
        raw_output = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        answer, found = parse_structured_output(raw_output)

        latency_ms = (time.time() - start) * 1000

        return GenerationResult(
            answer=answer,
            found=found,
            prompt=prompt_text,
            retrieval_results=contexts,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(completion_ids),
        )
