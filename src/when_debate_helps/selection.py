from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import iter_jsonl, write_json


@dataclass(frozen=True)
class CandidateProfile:
    candidate_id: str
    seed: int
    sigma: float
    score: float
    predictions: dict[str, str | None]
    correctness: dict[str, bool]


def load_profiles(path: str | Path) -> list[CandidateProfile]:
    profiles: list[CandidateProfile] = []
    example_ids: tuple[str, ...] | None = None
    for row in iter_jsonl(path):
        predictions = {str(item["example_id"]): item.get("prediction") for item in row["predictions"]}
        correctness = {str(item["example_id"]): bool(item["correct"]) for item in row["predictions"]}
        current_ids = tuple(sorted(predictions))
        if example_ids is None:
            example_ids = current_ids
        elif current_ids != example_ids:
            raise ValueError("All candidate profiles must cover the same example ids")
        profiles.append(
            CandidateProfile(
                candidate_id=str(row["candidate_id"]),
                seed=int(row["seed"]),
                sigma=float(row["sigma"]),
                score=float(row["score"]),
                predictions=predictions,
                correctness=correctness,
            )
        )
    if not profiles:
        raise ValueError(f"No profiles found in {path}")
    return profiles


def select_team(
    profiles: list[CandidateProfile], method: str, team_size: int, tie_break_seed: int = 42
) -> dict[str, Any]:
    if not 0 < team_size <= len(profiles):
        raise ValueError("team_size must be between 1 and the number of profiles")
    if method == "top_accuracy":
        ordered = sorted(profiles, key=lambda profile: (-profile.score, profile.candidate_id))[:team_size]
        steps = [
            {"candidate_id": profile.candidate_id, "individual_score": profile.score, "objective": None}
            for profile in ordered
        ]
    elif method in {"ac_greedy", "sc_greedy"}:
        proxy = _ac_proxy(profiles) if method == "ac_greedy" else _sc_proxy(profiles)
        ordered, steps = _greedy_coverage(
            profiles,
            proxy,
            team_size,
            use_accuracy_tie_break=method == "ac_greedy",
            tie_break_seed=tie_break_seed,
        )
    else:
        raise ValueError(f"Unknown selection method: {method}")
    by_id = {profile.candidate_id: profile for profile in profiles}
    result = {
        "method": method,
        "team_size": team_size,
        "members": [
            {
                "candidate_id": profile.candidate_id,
                "seed": profile.seed,
                "sigma": profile.sigma,
                "construction_score": profile.score,
            }
            for profile in ordered
        ],
        "steps": steps,
        "pool_size": len(profiles),
        "construction_examples": len(next(iter(by_id.values())).predictions),
    }
    if method == "sc_greedy":
        result["tie_break_seed"] = tie_break_seed
    return result


def select_and_save(
    profile_path: str | Path,
    output_path: str | Path,
    method: str,
    team_size: int,
    tie_break_seed: int = 42,
) -> None:
    write_json(output_path, select_team(load_profiles(profile_path), method, team_size, tie_break_seed))


def _ac_proxy(profiles: list[CandidateProfile]) -> dict[str, dict[str, float]]:
    return {
        profile.candidate_id: {example_id: float(correct) for example_id, correct in profile.correctness.items()}
        for profile in profiles
    }


def _sc_proxy(profiles: list[CandidateProfile]) -> dict[str, dict[str, float]]:
    example_ids = sorted(profiles[0].predictions)
    support: dict[str, Counter[str]] = {}
    valid_counts: dict[str, int] = {}
    for example_id in example_ids:
        valid = [profile.predictions[example_id] for profile in profiles if profile.predictions[example_id] is not None]
        support[example_id] = Counter(str(answer) for answer in valid)
        valid_counts[example_id] = len(valid)
    proxy: dict[str, dict[str, float]] = {}
    for profile in profiles:
        scores: dict[str, float] = {}
        for example_id, prediction in profile.predictions.items():
            denominator = valid_counts[example_id]
            scores[example_id] = (
                support[example_id][str(prediction)] / denominator if prediction is not None and denominator else 0.0
            )
        proxy[profile.candidate_id] = scores
    return proxy


def _greedy_coverage(
    profiles: list[CandidateProfile],
    proxy: dict[str, dict[str, float]],
    team_size: int,
    use_accuracy_tie_break: bool,
    tie_break_seed: int,
) -> tuple[list[CandidateProfile], list[dict[str, Any]]]:
    examples = sorted(next(iter(proxy.values())))
    current = {example_id: 0.0 for example_id in examples}
    remaining = {profile.candidate_id: profile for profile in profiles}
    rng = random.Random(tie_break_seed)
    selected: list[CandidateProfile] = []
    steps: list[dict[str, Any]] = []
    for _ in range(team_size):
        scored: list[tuple[float, CandidateProfile]] = []
        for candidate_id, profile in remaining.items():
            new_values = [max(current[example_id], proxy[candidate_id][example_id]) for example_id in examples]
            objective = sum(new_values) / len(examples)
            scored.append((objective, profile))
        objective = max(item[0] for item in scored)
        tied = [profile for score, profile in scored if score == objective]
        if use_accuracy_tie_break:
            winner = max(tied, key=lambda profile: (profile.score, _reverse_id(profile.candidate_id)))
        else:
            winner = rng.choice(sorted(tied, key=lambda profile: profile.candidate_id))
        previous = sum(current.values()) / len(examples)
        for example_id in examples:
            current[example_id] = max(current[example_id], proxy[winner.candidate_id][example_id])
        selected.append(winner)
        steps.append(
            {
                "candidate_id": winner.candidate_id,
                "individual_score": winner.score,
                "marginal_gain": objective - previous,
                "objective": objective,
            }
        )
        del remaining[winner.candidate_id]
    return selected, steps


def _reverse_id(candidate_id: str) -> tuple[int, ...]:
    return tuple(-ord(character) for character in candidate_id)
