"""
Generator with "llama-3.3-70b-versatile" via Groq API
- input: `list[RetrievalResult]`
- output: GenerationResult
"""

from __future__ import annotations

import os
import time
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
        api_key = os.getenv("GROQ_API_KEY")
        self.client = None
        if api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=api_key)
            except Exception:
                self.client = None

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        model_id = "meta-llama/Llama-3.3-70B-Instruct"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.context_builder = ContextBuilder(self.tokenizer, max_context_tokens=max_context_tokens)
        except Exception:
            self.tokenizer = None
            self.context_builder = None

    def generate(self, question: str, contexts: list[RetrievalResult]) -> GenerationResult:
        start = time.time()

        if not self.client:
            fallback_text = (
                "[Note: GROQ_API_KEY is not configured in .env or environment]\n\n"
                "Top Retrieved Contexts:\n"
            )
            for i, c in enumerate(contexts, 1):
                if c.document:
                    fallback_text += f"\n[{i}] ({c.doc_id}) {c.document.text}\n"

            return GenerationResult(
                answer=fallback_text.strip(),
                found=len(contexts) > 0,
                prompt=f"Question: {question}",
                retrieval_results=contexts,
                latency_ms=(time.time() - start) * 1000,
                prompt_tokens=0,
                completion_tokens=0,
            )

        try:
            if self.context_builder:
                context_block = self.context_builder.build(contexts)
            else:
                context_block = "\n\n".join([c.document.text for c in contexts if c.document])

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(question, context_block)},
            ]

            prompt_text = str(messages)
            prompt_tokens = 0
            if self.tokenizer:
                try:
                    prompt_text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    prompt_tokens = len(self.tokenizer.encode(prompt_text))
                except Exception:
                    pass

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.2,
                max_completion_tokens=self.max_new_tokens,
                response_format={"type": "json_object"}
            )
            raw_output = response.choices[0].message.content
            completion_tokens = len(self.tokenizer.encode(raw_output)) if self.tokenizer else 0
            answer, found = parse_structured_output(raw_output)

            return GenerationResult(
                answer=answer if found else raw_output,
                found=found,
                prompt=prompt_text,
                retrieval_results=contexts,
                latency_ms=(time.time() - start) * 1000,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except Exception as e:
            fallback_text = f"[Groq API Error: {str(e)}]\n\nTop Retrieved Contexts:\n"
            for i, c in enumerate(contexts, 1):
                if c.document:
                    fallback_text += f"\n[{i}] ({c.doc_id}) {c.document.text}\n"

            return GenerationResult(
                answer=fallback_text.strip(),
                found=len(contexts) > 0,
                prompt=f"Question: {question}",
                retrieval_results=contexts,
                latency_ms=(time.time() - start) * 1000,
                prompt_tokens=0,
                completion_tokens=0,
            )
