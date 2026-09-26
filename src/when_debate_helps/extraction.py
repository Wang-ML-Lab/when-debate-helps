from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

_ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_BOXED = re.compile(r"\\boxed\s*\{\s*([^{}]+?)\s*\}", re.IGNORECASE)
_MC_TRAILING = re.compile(r"(?:^|[\s:(])([A-Z])(?:[\s.)\]}]|$)", re.IGNORECASE)
_NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")


def extract_answer(text: str, task: str, num_choices: int = 4) -> str | None:
    if not text or not text.strip():
        return None
    tagged = _ANSWER_TAG.findall(text)
    boxed = _BOXED.findall(text)
    candidates = tagged or boxed
    if task == "multiple_choice":
        allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:num_choices])
        for candidate in reversed(candidates):
            letter = _extract_choice(candidate, allowed)
            if letter is not None:
                return letter
        for match in reversed(list(_MC_TRAILING.finditer(text))):
            letter = match.group(1).upper()
            if letter in allowed:
                return letter
        return None
    for candidate in reversed(candidates):
        normalized = normalize_free_form(candidate)
        if normalized is not None:
            return normalized
    numbers = _NUMBER.findall(text)
    return normalize_free_form(numbers[-1]) if numbers else None


def _extract_choice(value: str, allowed: set[str]) -> str | None:
    cleaned = value.strip().upper()
    if cleaned in allowed:
        return cleaned
    match = re.search(r"(?:OPTION|ANSWER)?\s*[:(]?\s*([A-Z])", cleaned)
    if match and match.group(1) in allowed:
        return match.group(1)
    return None


def normalize_free_form(value: str) -> str | None:
    cleaned = value.strip().strip("$ ").replace(",", "")
    if not cleaned:
        return None
    try:
        decimal = Decimal(cleaned)
    except InvalidOperation:
        return re.sub(r"\s+", " ", cleaned).lower()
    if decimal == decimal.to_integral():
        return str(decimal.quantize(Decimal(1)))
    return format(decimal.normalize(), "f")


def normalize_gold(answer: str, task: str, num_choices: int = 4) -> str:
    if task == "multiple_choice":
        extracted = extract_answer(f"<answer>{answer}</answer>", task, num_choices)
        if extracted is None:
            raise ValueError(f"Invalid multiple-choice gold answer: {answer}")
        return extracted
    normalized = normalize_free_form(answer)
    if normalized is None:
        raise ValueError("Gold answer cannot be empty")
    return normalized


def answer_is_correct(prediction: str | None, gold: str, task: str, num_choices: int = 4) -> bool:
    return prediction is not None and prediction == normalize_gold(gold, task, num_choices)
