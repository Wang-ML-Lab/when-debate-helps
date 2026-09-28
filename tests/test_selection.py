from when_debate_helps.selection import CandidateProfile, _sc_proxy, select_team


def _profile(candidate_id, score, correct, predictions=None):
    predictions = predictions or {key: ("A" if value else "B") for key, value in correct.items()}
    return CandidateProfile(candidate_id, int(candidate_id[-1]), 0.001, score, predictions, correct)


def test_ac_greedy_selects_complementary_lower_accuracy_candidate() -> None:
    profiles = [
        _profile("c1", 2 / 3, {"q1": True, "q2": True, "q3": False}),
        _profile("c2", 2 / 3, {"q1": True, "q2": True, "q3": False}),
        _profile("c3", 1 / 3, {"q1": False, "q2": False, "q3": True}),
    ]
    result = select_team(profiles, "ac_greedy", 2)
    assert [member["candidate_id"] for member in result["members"]] == ["c1", "c3"]
    assert result["steps"][-1]["objective"] == 1.0


def test_sc_proxy_gives_invalid_extractions_zero_coverage() -> None:
    profiles = [
        _profile("c1", 1.0, {"q": True}, {"q": "A"}),
        _profile("c2", 0.0, {"q": False}, {"q": "A"}),
        _profile("c3", 0.0, {"q": False}, {"q": None}),
    ]
    proxy = _sc_proxy(profiles)
    assert proxy["c1"]["q"] == 1.0
    assert proxy["c2"]["q"] == 1.0
    assert proxy["c3"]["q"] == 0.0


def test_sc_greedy_random_tie_break_is_reproducible_and_label_free() -> None:
    predictions = {"q": "A"}
    first_labels = [
        _profile("c1", 1.0, {"q": True}, predictions),
        _profile("c2", 0.0, {"q": False}, predictions),
    ]
    reversed_labels = [
        _profile("c1", 0.0, {"q": False}, predictions),
        _profile("c2", 1.0, {"q": True}, predictions),
    ]

    first = select_team(first_labels, "sc_greedy", 1, tie_break_seed=7)
    repeated = select_team(first_labels, "sc_greedy", 1, tie_break_seed=7)
    relabeled = select_team(reversed_labels, "sc_greedy", 1, tie_break_seed=7)

    assert first["members"][0]["candidate_id"] == repeated["members"][0]["candidate_id"]
    assert first["members"][0]["candidate_id"] == relabeled["members"][0]["candidate_id"]
    assert first["tie_break_seed"] == 7


def test_top_accuracy_tie_break_is_stable() -> None:
    profiles = [
        _profile("c2", 0.5, {"q": True}),
        _profile("c1", 0.5, {"q": True}),
    ]
    result = select_team(profiles, "top_accuracy", 1)
    assert result["members"][0]["candidate_id"] == "c1"
