from __future__ import annotations

import json
import os
from src.generation.prompt import PromptMode, build_prompt
from src.data_models.data_models import RetrievalResult


class Generator:
    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.2,
        max_tokens: int = 512,
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

        if text.startswith("```"):
            lines = text.splitlines()

            if lines and lines[0].startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]

            text = "\n".join(lines).strip()

        try:
            output = json.loads(text)
        except json.JSONDecodeError:
            return {
                "answer": response_text.strip(),
                "references": [],
            }

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
            fallback_text = f"[Groq API Error: {str(e)}]\n\nTop Retrieved Contexts:\n"
            for i, c in enumerate(contexts, 1):
                if c.document:
                    fallback_text += f"\n[{i}] ({c.doc_id}) {c.document.text}\n"
            return {
                "answer": fallback_text.strip(),
                "references": [c.doc_id for c in contexts],
                "raw_output": fallback_text,
            }