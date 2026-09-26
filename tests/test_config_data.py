from pathlib import Path

from when_debate_helps.config import load_config
from when_debate_helps.data import load_examples, render_question


def test_config_dot_overrides() -> None:
    config = load_config("configs/debug.yaml", ["debate.rounds=3", "model.device=cpu"])
    assert config["debate"]["rounds"] == 3
    assert config["model"]["device"] == "cpu"


def test_example_fixture_loads() -> None:
    path = Path("examples/construction.jsonl")
    examples = load_examples(path)
    assert len(examples) == 4
    assert "A. 4" in render_question(examples[0])
