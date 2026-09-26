from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .io import write_jsonl


def build_seed_bank(sigmas: list[float], seeds_per_sigma: int, global_seed: int) -> list[dict[str, Any]]:
    if not sigmas or seeds_per_sigma <= 0:
        raise ValueError("sigmas must be non-empty and seeds_per_sigma must be positive")
    rng = random.Random(global_seed)
    rows: list[dict[str, Any]] = []
    candidate_index = 0
    for sigma in sigmas:
        if sigma < 0:
            raise ValueError("sigmas must be non-negative")
        for _ in range(seeds_per_sigma):
            rows.append(
                {
                    "candidate_id": f"candidate-{candidate_index:06d}",
                    "seed": rng.randrange(0, 2**31),
                    "sigma": float(sigma),
                }
            )
            candidate_index += 1
    return rows


def save_seed_bank(path: str | Path, sigmas: list[float], seeds_per_sigma: int, global_seed: int) -> None:
    write_jsonl(path, build_seed_bank(sigmas, seeds_per_sigma, global_seed))
