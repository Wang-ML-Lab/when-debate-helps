from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .io import iter_jsonl


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    answer: str
    choices: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def load_examples(path: str | Path, limit: int | None = None) -> list[Example]:
    examples: list[Example] = []
    seen: set[str] = set()
    for index, row in enumerate(iter_jsonl(path)):
        example_id = str(row.get("id", index))
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not example_id or not question or not answer:
            raise ValueError(f"Example {index} must contain non-empty id, question, and answer fields")
        if example_id in seen:
            raise ValueError(f"Duplicate example id: {example_id}")
        seen.add(example_id)
        raw_choices = row.get("choices", [])
        if isinstance(raw_choices, dict):
            raw_choices = [raw_choices[key] for key in sorted(raw_choices)]
        if not isinstance(raw_choices, list):
            raise ValueError(f"Example {example_id}: choices must be a list or mapping")
        examples.append(
            Example(
                id=example_id,
                question=question,
                answer=answer,
                choices=tuple(str(choice) for choice in raw_choices),
                metadata=dict(row.get("metadata", {})),
            )
        )
        if limit is not None and len(examples) >= limit:
            break
    if not examples:
        raise ValueError(f"No examples found in {path}")
    return examples


def render_question(example: Example) -> str:
    if not example.choices:
        return example.question
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    choices = "\n".join(f"{labels[index]}. {choice}" for index, choice in enumerate(example.choices))
    return f"{example.question}\n\nChoices:\n{choices}"


def initial_messages(example: Example, task: str, system_prompt: str | None = None) -> list[dict[str, str]]:
    if system_prompt is None:
        if task == "multiple_choice":
            system_prompt = (
                "You are a careful reasoning assistant. End with exactly one final answer tag "
                "<answer>X</answer>, where X is the option letter. Do not write after the tag."
            )
        else:
            system_prompt = (
                "You are a careful reasoning assistant. Solve the problem and end with exactly one final answer tag "
                "<answer>X</answer>, where X is the final answer. Do not write after the tag."
            )
    if task == "multiple_choice":
        instruction = "Think briefly, solve the multiple-choice question, and follow the required answer format."
    else:
        instruction = "Solve the problem step by step and follow the required answer format."
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{render_question(example)}\n\n{instruction}"},
    ]
