"""
Generation models via Hugging Face:
- input: `list[RetrievalResult]`
- output: GenerationResult
"""
from __future__ import annotations

import os
import time
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import settings
from src.data_models.data_models import RetrievalResult, GenerationResult
from src.generation.base import Generator
from src.generation.prompt_hf import build_prompt, PromptMode
from src.generation.registry import register_generator



load_dotenv()
if hf_token := os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = hf_token



@register_generator("hf")
class HFGenerator(Generator):

    def __init__(
        self,
        model_name: str = settings.hf_model_name,
        max_new_tokens: int = settings.max_new_tokens,
    ):
        self.model_name = model_name

        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    def generate(
        self,
        question: str,
        contexts: list[RetrievalResult],
        prompt_mode: PromptMode = settings.prompt_mode,
    ) -> GenerationResult:
        start = time.time()

        # prepare the model input
        prompt = build_prompt(
            query=question,
            contexts=contexts,
            mode=prompt_mode,
        )
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        found = True

        # Remove thinking blocks if present
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()

        # If no answer found
        if "False!" in content:
            content = ""
            found = False

        latency_ms = (time.time() - start) * 1000
        
        return GenerationResult(
            answer=content,
            found=found,
            prompt=prompt,
            retrieval_results=contexts,
            latency_ms=latency_ms,
            prompt_tokens=len(model_inputs.input_ids[0]),
            completion_tokens=len(output_ids),
        )
