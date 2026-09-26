from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any


def majority_answer(answers: Iterable[str | None]) -> str | None:
    counts = Counter(answer for answer in answers if answer is not None)
    if not counts:
        return None
    most_common = counts.most_common()
    best_count = most_common[0][1]
    winners = [answer for answer, count in most_common if count == best_count]
    return winners[0] if len(winners) == 1 else None


def compute_readout_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("At least one record is required")
    n = len(records)
    proposal_hits = 0
    vote_correct = 0
    final_correct = 0
    headroom = 0
    recovered = 0
    new_correct = 0
    damaged = 0
    for record in records:
        initial_correct = [bool(value) for value in record["initial_correct"]]
        proposal = any(initial_correct)
        vote = bool(record["initial_vote_correct"])
        final = bool(record["final_correct"])
        proposal_hits += proposal
        vote_correct += vote
        final_correct += final
        headroom_event = proposal and not vote
        headroom += headroom_event
        recovered += headroom_event and final
        new_correct += (not proposal) and (not vote) and final
        damaged += vote and not final
    recovery_mass = recovered / n
    new_correct_mass = new_correct / n
    damage_mass = damaged / n
    vote_accuracy = vote_correct / n
    final_accuracy = final_correct / n
    decomposition_gain = recovery_mass + new_correct_mass - damage_mass
    observed_gain = final_accuracy - vote_accuracy
    if abs(decomposition_gain - observed_gain) > 1e-12:
        raise AssertionError("Recovery/new-correct/damage decomposition does not match the observed gain")
    return {
        "examples": n,
        "proposal_hit": proposal_hits / n,
        "initial_vote_accuracy": vote_accuracy,
        "final_accuracy": final_accuracy,
        "recoverable_headroom": headroom / n,
        "recovery_rate": recovered / headroom if headroom else None,
        "recovery_mass": recovery_mass,
        "new_correctness_mass": new_correct_mass,
        "damage_rate": damaged / vote_correct if vote_correct else None,
        "damage_mass": damage_mass,
        "gain": observed_gain,
        "decomposition_gain": decomposition_gain,
        "counts": {
            "proposal_hit": proposal_hits,
            "initial_vote_correct": vote_correct,
            "final_correct": final_correct,
            "headroom": headroom,
            "recovered": recovered,
            "new_correct": new_correct,
            "damaged": damaged,
        },
    }


def round_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    max_rounds = max(len(record["rounds"]) for record in records)
    output: list[dict[str, Any]] = []
    for round_index in range(max_rounds):
        available = [record["rounds"][round_index] for record in records if round_index < len(record["rounds"])]
        output.append(
            {
                "round": round_index,
                "examples": len(available),
                "proposal_hit": sum(bool(item["proposal_hit"]) for item in available) / len(available),
                "vote_accuracy": sum(bool(item["vote_correct"]) for item in available) / len(available),
                "valid_vote_rate": sum(item["vote"] is not None for item in available) / len(available),
            }
        )
    return output
