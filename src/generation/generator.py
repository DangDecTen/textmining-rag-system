from __future__ import annotations

import json
import os
from groq import Groq
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
        if api_key is None:
            raise ValueError("GROQ_API_KEY is not set.")

        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _parse_response(self, response_text: str) -> dict:
        """
        Parse model output into:
        {
            "answer": str,
            "references": list[str]
        }
        """

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
        prompt = build_prompt(
            query=query,
            contexts=contexts,
            mode=prompt_mode,
        )

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