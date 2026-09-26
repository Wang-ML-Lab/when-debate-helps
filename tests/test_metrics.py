from when_debate_helps.metrics import compute_readout_metrics, majority_answer


def test_majority_requires_unique_winner() -> None:
    assert majority_answer(["A", "A", "B", None]) == "A"
    assert majority_answer(["A", "B"]) is None
    assert majority_answer([None, None]) is None


def test_recovery_damage_decomposition() -> None:
    records = [
        {"initial_correct": [True, False], "initial_vote_correct": False, "final_correct": True},
        {"initial_correct": [False, False], "initial_vote_correct": False, "final_correct": True},
        {"initial_correct": [True, True], "initial_vote_correct": True, "final_correct": False},
        {"initial_correct": [True, True], "initial_vote_correct": True, "final_correct": True},
    ]
    metrics = compute_readout_metrics(records)
    assert metrics["recoverable_headroom"] == 0.25
    assert metrics["recovery_rate"] == 1.0
    assert metrics["new_correctness_mass"] == 0.25
    assert metrics["damage_mass"] == 0.25
    assert metrics["gain"] == 0.25
    assert metrics["gain"] == metrics["decomposition_gain"]
