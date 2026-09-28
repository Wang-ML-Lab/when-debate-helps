from collections import Counter

from when_debate_helps.seeds import build_seed_bank


def test_seed_bank_supports_paper_scale_counts() -> None:
    rows = build_seed_bank(
        [0.0005, 0.001, 0.002],
        seeds_per_sigma=None,
        global_seed=42,
        counts_per_sigma=[160, 174, 166],
    )

    assert len(rows) == 500
    assert Counter(row["sigma"] for row in rows) == {0.0005: 160, 0.001: 174, 0.002: 166}
    assert rows == build_seed_bank(
        [0.0005, 0.001, 0.002],
        seeds_per_sigma=None,
        global_seed=42,
        counts_per_sigma=[160, 174, 166],
    )
