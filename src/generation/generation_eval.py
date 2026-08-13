from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

from groq import Groq

from src.data_models.data_models import QAExample
from src.retrieval.base import Retriever
from src.generation.generator import Generator


JUDGE_MODEL = "llama-3.1-8b-instant"


class LLMJudge:
    def __init__(
        self,
        model: str = JUDGE_MODEL,
    ):
        self.client = Groq(api_key=os.getenv("JUDGE_API_KEY"))
        self.model = model

    def evaluate(
        self,
        question: str,
        true_answer: str,
        generated_answer: str,
        retrieved_context: str,
    ) -> dict:

        prompt = f"""
You are an impartial evaluator for a Retrieval-Augmented Generation (RAG) system
specializing in cybersecurity and the MITRE ATT&CK framework.

Question:
{question}

Retrieved Context:
{retrieved_context}

True Answer:
{true_answer}

Generated Answer:
{generated_answer}

Evaluate the generated answer according to the following metrics.

1. Hard Accuracy (0-10)

Compare ONLY with the True Answer.

Score based on:
- Correctness of factual information.
- Completeness of the answer.
- Presence of incorrect or contradictory information.

Do NOT penalize:
- Different wording or sentence structure.
- Different ordering of information.

Scoring guideline:
10 = Factually equivalent to the true answer.
8-9 = Correct with only minor omissions or wording differences.
6-7 = Mostly correct but missing important details.
3-5 = Partially correct with significant omissions or inaccuracies.
1-2 = Mostly incorrect.
0 = Completely incorrect or unrelated.

----------------------------------------------------

2. Faithfulness (0-10)

Compare ONLY with the Retrieved Context.

Evaluate whether every factual statement in the generated answer is supported by the retrieved context.

Do NOT consider the True Answer.

Scoring guideline:
10 = Every factual statement is fully supported by the retrieved context.
8-9 = Almost entirely supported with only minor unsupported details.
6-7 = Mostly supported but contains some unsupported statements.
3-5 = Many unsupported or speculative statements.
1-2 = Mostly unsupported.
0 = Entirely hallucinated or contradicts the retrieved context.

----------------------------------------------------

3. Answer Relevancy (0-10)

Compare ONLY with the Question.

Evaluate how well the generated answer addresses the user's question.

Do NOT consider factual correctness.

Scoring guideline:
10 = Completely answers the question.
8-9 = Answers the question with only minor missing details.
6-7 = Partially answers the question.
3-5 = Only weakly addresses the question.
1-2 = Barely related.
0 = Completely irrelevant.

----------------------------------------------------

Return ONLY valid JSON.

{{
    "hard_accuracy": 0,
    "faithfulness": 0,
    "answer_relevancy": 0
}}
"""

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=128,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        return json.loads(text)

def load_cache(cache_path: Path) -> list[dict]:
    if not cache_path.exists():
        return []
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(cache_path: Path, results: list[dict],):
    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def evaluate_generator(
    retriever: Retriever,
    generator: Generator,
    qa_examples: list[QAExample],
    prompt_mode: str,
    cache_path: str,
    top_k: int = 5,
):

    judge = LLMJudge()

    cache_path = Path(cache_path)
    results = load_cache(cache_path)
    finished_questions = {
        item["question"]
        for item in results
    }
    print(f"Loaded {len(results)} cached evaluation(s).")

    for idx, qa in enumerate(qa_examples, start=1):
        if qa.question in finished_questions:
            continue
        print(f"[{idx}/{len(qa_examples)}] {qa.question}")

        retrieved = retriever.search(
            qa.question,
            top_k=top_k,
        )

        generated = generator.generate(
            query=qa.question,
            contexts=retrieved,
            prompt_mode=prompt_mode,
        )

        answer = generated["answer"]
        references = generated["references"]

        retrieved_context = "\n\n".join(
            [
                f"[{r.doc_id}]\n{r.document.text}"
                for r in retrieved
                if r.document is not None
            ]
        )

        retrieved_ids = {
            r.doc_id
            for r in retrieved
        }

        # if len(references) == 0:
        #     citation_accuracy = 0.0
        # else:
        #     citation_accuracy = (
        #         sum(
        #             ref in retrieved_ids
        #             for ref in references
        #         )
        #         / len(references)
        #     )

        try:
            judge_result = judge.evaluate(
                question=qa.question,
                true_answer=qa.answer,
                generated_answer=answer,
                retrieved_context=retrieved_context,
            )

        except Exception as e:
            print("\nEvaluation stopped.")
            print(e)
            save_cache(cache_path,results)
            print(f"Saved {len(results)} evaluated samples to:")
            print(f"  {cache_path}")
            raise
        

        result = {
            "question": qa.question,
            "source": qa.source,
            "true_answer": qa.answer,
            "generated_answer": answer,
            "references": references,
            #"citation_accuracy": citation_accuracy,
            "hard_accuracy": judge_result["hard_accuracy"],
            "faithfulness": judge_result["faithfulness"],
            "answer_relevancy": judge_result["answer_relevancy"],
        }
        results.append(result)
        save_cache(cache_path, results)

    report = {
        "n": len(results),
        "overall": {},
        "by_source": {},
    }
    overall = defaultdict(float)
    by_source = defaultdict(
        lambda: {
            "n": 0,
            "hard_accuracy": 0.0,
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            #"citation_accuracy": 0.0,
        }
    )

    for result in results:
        overall["hard_accuracy"] += result["hard_accuracy"]
        overall["faithfulness"] += result["faithfulness"]
        overall["answer_relevancy"] += result["answer_relevancy"]
        #overall["citation_accuracy"] += result["citation_accuracy"]

        src = result["source"]

        by_source[src]["n"] += 1
        by_source[src]["hard_accuracy"] += result["hard_accuracy"]
        by_source[src]["faithfulness"] += result["faithfulness"]
        by_source[src]["answer_relevancy"] += result["answer_relevancy"]
        #by_source[src]["citation_accuracy"] += result["citation_accuracy"]

    n = len(results)
    if n == 0:
        raise ValueError("No evaluated samples found.")

    report["overall"] = {
        "hard_accuracy": overall["hard_accuracy"] / n,
        "faithfulness": overall["faithfulness"] / n,
        "answer_relevancy": overall["answer_relevancy"] / n,
        #"citation_accuracy": overall["citation_accuracy"] / n,
    }

    for src, metrics in by_source.items():
        total = metrics["n"]
        report["by_source"][src] = {
            "n": total,
            "hard_accuracy": metrics["hard_accuracy"] / total,
            "faithfulness": metrics["faithfulness"] / total,
            "answer_relevancy": metrics["answer_relevancy"] / total,
            #"citation_accuracy": metrics["citation_accuracy"] / total,
        }

    return report


def print_report(report: dict) -> None:
    print("\n===================================")
    print("Generator Evaluation Report")
    print("===================================")

    print(f"Samples: {report['n']}")

    print("\nOverall:")

    print(f"Hard Accuracy: {report['overall']['hard_accuracy']:.2f}/10")
    print(f"Faithfulness: {report['overall']['faithfulness']:.2f}/10")
    print(f"Answer Relevancy: {report['overall']['answer_relevancy']:.2f}/10")
    #print(f"Citation Accuracy: {report['overall']['citation_accuracy']:.3f}")

    print("\nBy Source:")
    for src, metrics in sorted(
        report["by_source"].items(),
        key=lambda x: -x[1]["n"],
    ):

        print(f"\n{src} (n={metrics['n']})")

        print(f"Hard Accuracy: {metrics['hard_accuracy']:.2f}/10")
        print(f"Faithfulness: {metrics['faithfulness']:.2f}/10")
        print(f"Answer Relevancy: {metrics['answer_relevancy']:.2f}/10")
        #print(f"Citation Accuracy: {metrics['citation_accuracy']:.3f}")