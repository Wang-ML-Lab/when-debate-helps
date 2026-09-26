from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .data import Example, initial_messages
from .extraction import answer_is_correct, extract_answer
from .io import iter_jsonl, write_jsonl
from .model import GenerationConfig, TransformersGenerator
from .perturbation import GaussianPerturber


def profile_candidates(
    generator: TransformersGenerator,
    perturber: GaussianPerturber,
    examples: list[Example],
    candidates_path: str | Path,
    output_path: str | Path,
    task: str,
    generation: GenerationConfig,
    system_prompt: str | None = None,
    shard_id: int = 0,
    num_shards: int = 1,
    resume: bool = True,
) -> None:
    if num_shards <= 0 or not 0 <= shard_id < num_shards:
        raise ValueError("Require num_shards > 0 and 0 <= shard_id < num_shards")
    candidates = list(iter_jsonl(candidates_path))
    selected = [candidate for index, candidate in enumerate(candidates) if index % num_shards == shard_id]
    completed: set[str] = set()
    output = Path(output_path)
    if resume and output.exists():
        completed = {str(row["candidate_id"]) for row in iter_jsonl(output)}
    elif not resume and output.exists():
        output.unlink()
    selected = [candidate for candidate in selected if str(candidate["candidate_id"]) not in completed]
    conversations = [initial_messages(example, task, system_prompt) for example in examples]
    for candidate in tqdm(selected, desc=f"profile shard {shard_id}/{num_shards}"):
        started = time.time()
        seed = int(candidate["seed"])
        sigma = float(candidate["sigma"])
        with perturber.perturb(seed, sigma):
            texts = generator.generate(conversations, generation, seed=seed)
        predictions: list[dict[str, Any]] = []
        for example, text in zip(examples, texts, strict=True):
            extracted = extract_answer(text, task, len(example.choices) or 4)
            predictions.append(
                {
                    "example_id": example.id,
                    "prediction": extracted,
                    "correct": answer_is_correct(extracted, example.answer, task, len(example.choices) or 4),
                    "response": text,
                }
            )
        correct = sum(bool(prediction["correct"]) for prediction in predictions)
        write_jsonl(
            output,
            [
                {
                    "candidate_id": str(candidate["candidate_id"]),
                    "seed": seed,
                    "sigma": sigma,
                    "score": correct / len(predictions),
                    "correct": correct,
                    "examples": len(predictions),
                    "elapsed_seconds": time.time() - started,
                    "predictions": predictions,
                }
            ],
            append=output.exists(),
        )
        generator.clear_cache()


def merge_profile_shards(inputs: list[str | Path], output: str | Path) -> None:
    rows: dict[str, dict[str, Any]] = {}
    for path in inputs:
        for row in iter_jsonl(path):
            candidate_id = str(row["candidate_id"])
            if candidate_id in rows and rows[candidate_id] != row:
                raise ValueError(f"Conflicting duplicate candidate: {candidate_id}")
            rows[candidate_id] = row
    ordered = sorted(rows.values(), key=lambda row: row["candidate_id"])
    write_jsonl(output, ordered)
