from __future__ import annotations

import json
import os
from dotenv import load_dotenv
from src.generation.prompt import PromptMode, build_prompt
from src.data_models.data_models import RetrievalResult

load_dotenv()


class Generator:
    def __init__(
        self,
        model: str = "qwen/qwen3.6-27b",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):
        api_key = os.getenv("GROQ_API_KEY")
        self.client = None
        if api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=api_key)
            except Exception:
                self.client = None

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _parse_response(self, response_text: str) -> dict:
        text = response_text.strip()

        # Remove thinking blocks if present
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()

        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # Attempt JSON substring extraction
        output = None
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = text[start_idx : end_idx + 1]
            try:
                output = json.loads(json_str)
            except Exception:
                output = None

        if not isinstance(output, dict):
            try:
                output = json.loads(text)
            except Exception:
                output = {"answer": response_text.strip(), "references": []}

        answer = output.get("answer", "")
        references = output.get("references", [])

        if not isinstance(answer, str):
            answer = str(answer)

        if not isinstance(references, list):
            references = []

        references = [str(ref) for ref in references]

        return {
            "answer": answer.strip(),
            "references": references,
        }

    def generate(
        self,
        query: str,
        contexts: list[RetrievalResult],
        prompt_mode: PromptMode = "baseline",
    ) -> dict:
        if not self.client:
            fallback_text = (
                "[Note: GROQ_API_KEY is not set in environment or .env file]\n\n"
                "Top Retrieved Contexts:\n"
            )
            for i, c in enumerate(contexts, 1):
                if c.document:
                    fallback_text += f"\n[{i}] ({c.doc_id}) {c.document.text}\n"
            return {
                "answer": fallback_text.strip(),
                "references": [c.doc_id for c in contexts],
                "raw_output": fallback_text,
            }

        prompt = build_prompt(
            query=query,
            contexts=contexts,
            mode=prompt_mode,
        )

        import time
        for attempt in range(5):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                )
                response_text = response.choices[0].message.content
                parsed = self._parse_response(response_text)
                parsed["raw_output"] = response_text
                return parsed
            except Exception as e:
                if ("429" in str(e) or "rate_limit" in str(e).lower()) and attempt < 4:
                    time.sleep(6 * (attempt + 1))
                else:
                    fallback_text = f"[Groq API Error: {str(e)}]\n\nTop Retrieved Contexts:\n"
                    for i, c in enumerate(contexts, 1):
                        if c.document:
                            fallback_text += f"\n[{i}] ({c.doc_id}) {c.document.text}\n"
                    return {
                        "answer": fallback_text.strip(),
                        "references": [c.doc_id for c in contexts],
                        "raw_output": fallback_text,
                    }