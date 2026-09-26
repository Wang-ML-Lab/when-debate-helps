from contextlib import nullcontext

from when_debate_helps.data import Example
from when_debate_helps.io import iter_jsonl, write_jsonl
from when_debate_helps.model import GenerationConfig
from when_debate_helps.profile import profile_candidates


class FakeGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, conversations, generation, seed):
        self.calls += 1
        return ["<answer>A</answer>" for _ in conversations]

    def clear_cache(self) -> None:
        pass


class FakePerturber:
    def perturb(self, seed, sigma):
        return nullcontext()


def test_no_resume_overwrites_existing_profile(tmp_path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    output = tmp_path / "profiles.jsonl"
    write_jsonl(candidates, [{"candidate_id": "candidate-000000", "seed": 1, "sigma": 0.1}])
    write_jsonl(output, [{"candidate_id": "stale"}])

    profile_candidates(
        generator=FakeGenerator(),
        perturber=FakePerturber(),
        examples=[Example(id="q", question="Question?", answer="A", choices=("yes", "no"))],
        candidates_path=candidates,
        output_path=output,
        task="multiple_choice",
        generation=GenerationConfig(max_new_tokens=1),
        resume=False,
    )

    rows = list(iter_jsonl(output))
    assert [row["candidate_id"] for row in rows] == ["candidate-000000"]
