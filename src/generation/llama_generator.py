"""
Generator with "llama-3.3-70b-versatile" via Groq API
- input: `list[RetrievalResult]`
- output: GenerationResult
"""

from __future__ import annotations

import os
import time
from groq import Groq
from dotenv import load_dotenv
from transformers import AutoTokenizer

from src.data_models.data_models import RetrievalResult, GenerationResult
from src.generation.base import Generator
from src.generation.context_builder import ContextBuilder
from src.generation.output_parser import parse_structured_output
from src.generation.prompt import SYSTEM_PROMPT, build_user_message


load_dotenv()


class LlamaGenerator(Generator):
    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        max_context_tokens: int = 1500,
        max_new_tokens: int = 128,
    ):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        model_id = "meta-llama/Llama-3.3-70B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id) # Groq model use tokenizer from original model
        self.context_builder = ContextBuilder(self.tokenizer, max_context_tokens=max_context_tokens)

    def generate(self, question: str, contexts: list[RetrievalResult]) -> GenerationResult:
        start = time.time()
        
        context_block = self.context_builder.build(contexts)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(question, context_block)},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = len(self.tokenizer.encode(prompt_text))


        response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.2,
                max_completion_tokens=self.max_new_tokens,
                response_format={"type": "json_object"}
        )
        raw_output = response.choices[0].message.content
        completion_tokens = len(self.tokenizer.encode(raw_output))
        answer, found = parse_structured_output(raw_output)

        latency_ms = (time.time() - start) * 1000

        return GenerationResult(
            answer=answer,
            found=found,
            prompt=prompt_text,
            retrieval_results=contexts,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
