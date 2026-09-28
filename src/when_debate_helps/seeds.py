from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .io import write_jsonl


def build_seed_bank(
    sigmas: list[float],
    seeds_per_sigma: int | None,
    global_seed: int,
    counts_per_sigma: list[int] | None = None,
) -> list[dict[str, Any]]:
    if not sigmas:
        raise ValueError("sigmas must be non-empty")
    if counts_per_sigma is not None:
        if len(counts_per_sigma) != len(sigmas) or any(count <= 0 for count in counts_per_sigma):
            raise ValueError("counts_per_sigma must contain one positive count for each sigma")
        counts = counts_per_sigma
    elif seeds_per_sigma is not None and seeds_per_sigma > 0:
        counts = [seeds_per_sigma] * len(sigmas)
    else:
        raise ValueError("seeds_per_sigma must be positive when counts_per_sigma is not set")
    rng = random.Random(global_seed)
    rows: list[dict[str, Any]] = []
    candidate_index = 0
    for sigma, count in zip(sigmas, counts, strict=True):
        if sigma < 0:
            raise ValueError("sigmas must be non-negative")
        for _ in range(count):
            rows.append(
                {
                    "candidate_id": f"candidate-{candidate_index:06d}",
                    "seed": rng.randrange(0, 2**31),
                    "sigma": float(sigma),
                }
            )
            candidate_index += 1
    return rows


def save_seed_bank(
    path: str | Path,
    sigmas: list[float],
    seeds_per_sigma: int | None,
    global_seed: int,
    counts_per_sigma: list[int] | None = None,
) -> None:
    write_jsonl(path, build_seed_bank(sigmas, seeds_per_sigma, global_seed, counts_per_sigma))
