from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import Example, initial_messages
from .extraction import answer_is_correct, extract_answer
from .io import read_json, write_json, write_jsonl
from .metrics import compute_readout_metrics, majority_answer, round_metrics
from .model import GenerationConfig, TransformersGenerator
from .perturbation import GaussianPerturber


@dataclass(frozen=True)
class TeamMember:
    candidate_id: str
    seed: int
    sigma: float


def load_team(path: str | Path) -> list[TeamMember]:
    payload = read_json(path)
    return [
        TeamMember(
            candidate_id=str(member["candidate_id"]),
            seed=int(member["seed"]),
            sigma=float(member["sigma"]),
        )
        for member in payload["members"]
    ]


def evaluate_thicket(
    generator: TransformersGenerator,
    perturber: GaussianPerturber,
    examples: list[Example],
    team: list[TeamMember],
    task: str,
    generation: GenerationConfig,
    output_dir: str | Path,
    rounds: int = 3,
    peer_max_chars: int = 800,
    system_prompt: str | None = None,
    debate_prompt: str = "verify",
    judge_generation: GenerationConfig | None = None,
    global_seed: int = 42,
) -> dict[str, Any]:
    if not team:
        raise ValueError("The team cannot be empty")
    if rounds < 0:
        raise ValueError("rounds must be non-negative")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base_messages = [initial_messages(example, task, system_prompt) for example in examples]
    responses: list[list[str]] = []
    for member_index, member in enumerate(team):
        with perturber.perturb(member.seed, member.sigma):
            responses.append(generator.generate(base_messages, generation, global_seed + member_index * 100_003))
        generator.clear_cache()
    all_rounds: list[list[list[str]]] = [[list(agent_responses) for agent_responses in responses]]
    for round_index in range(1, rounds + 1):
        previous = all_rounds[-1]
        current: list[list[str]] = []
        for member_index, member in enumerate(team):
            conversations = [
                _debate_messages(
                    base_messages[example_index],
                    own_response=previous[member_index][example_index],
                    peer_responses=[
                        previous[peer_index][example_index]
                        for peer_index in range(len(team))
                        if peer_index != member_index
                    ],
                    peer_max_chars=peer_max_chars,
                    prompt_style=debate_prompt,
                )
                for example_index in range(len(examples))
            ]
            with perturber.perturb(member.seed, member.sigma):
                current.append(
                    generator.generate(
                        conversations,
                        generation,
                        global_seed + round_index * 1_000_003 + member_index * 100_003,
                    )
                )
            generator.clear_cache()
        all_rounds.append(current)
    records = _build_records(examples, team, all_rounds, task)
    tied_indices = [index for index, record in enumerate(records) if record["rounds"][-1]["vote"] is None]
    if tied_indices:
        judge_conversations = [
            _judge_messages(
                base_messages[index],
                [all_rounds[-1][agent_index][index] for agent_index in range(len(team))],
                peer_max_chars,
            )
            for index in tied_indices
        ]
        judge_outputs = generator.generate(
            judge_conversations,
            judge_generation or generation,
            global_seed + 9_000_001,
        )
        for example_index, judge_output in zip(tied_indices, judge_outputs, strict=True):
            example = examples[example_index]
            judged = extract_answer(judge_output, task, len(example.choices) or 4)
            records[example_index]["judge"] = {"response": judge_output, "answer": judged}
            records[example_index]["final_answer"] = judged
    for example, record in zip(examples, records, strict=True):
        if "final_answer" not in record:
            record["final_answer"] = record["rounds"][-1]["vote"]
        record["final_correct"] = answer_is_correct(
            record["final_answer"], example.answer, task, len(example.choices) or 4
        )
    write_jsonl(output / "traces.jsonl", records)
    summary = compute_readout_metrics(records)
    summary["rounds"] = round_metrics(records)
    summary["team"] = [member.__dict__ for member in team]
    summary["settings"] = {
        "debate_rounds": rounds,
        "peer_max_chars": peer_max_chars,
        "debate_prompt": debate_prompt,
        "finalizer": "unique_majority_then_base_judge",
        "global_seed": global_seed,
    }
    write_json(output / "summary.json", summary)
    return summary


def evaluate_base(
    generator: TransformersGenerator,
    examples: list[Example],
    agents: int,
    task: str,
    generation: GenerationConfig,
    output_dir: str | Path,
    rounds: int = 3,
    peer_max_chars: int = 800,
    system_prompt: str | None = None,
    debate_prompt: str = "verify",
    judge_generation: GenerationConfig | None = None,
    judge_ties: bool = True,
    global_seed: int = 42,
) -> dict[str, Any]:
    """Evaluate sampled base-model agents for self-consistency or Vanilla Debate."""
    if agents <= 0:
        raise ValueError("agents must be positive")
    if rounds < 0:
        raise ValueError("rounds must be non-negative")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    team = [
        TeamMember(candidate_id=f"base-sample-{index}", seed=global_seed + index, sigma=0.0) for index in range(agents)
    ]
    base_messages = [initial_messages(example, task, system_prompt) for example in examples]
    initial = [
        generator.generate(base_messages, generation, global_seed + member_index * 100_003)
        for member_index in range(agents)
    ]
    all_rounds: list[list[list[str]]] = [initial]
    for round_index in range(1, rounds + 1):
        previous = all_rounds[-1]
        current: list[list[str]] = []
        for member_index in range(agents):
            conversations = [
                _debate_messages(
                    base_messages[example_index],
                    own_response=previous[member_index][example_index],
                    peer_responses=[
                        previous[peer_index][example_index]
                        for peer_index in range(agents)
                        if peer_index != member_index
                    ],
                    peer_max_chars=peer_max_chars,
                    prompt_style=debate_prompt,
                )
                for example_index in range(len(examples))
            ]
            current.append(
                generator.generate(
                    conversations,
                    generation,
                    global_seed + round_index * 1_000_003 + member_index * 100_003,
                )
            )
        all_rounds.append(current)
        generator.clear_cache()
    records = _build_records(examples, team, all_rounds, task)
    tied_indices = [index for index, record in enumerate(records) if record["rounds"][-1]["vote"] is None]
    if judge_ties and tied_indices:
        judge_conversations = [
            _judge_messages(
                base_messages[index],
                [all_rounds[-1][agent_index][index] for agent_index in range(agents)],
                peer_max_chars,
            )
            for index in tied_indices
        ]
        judge_outputs = generator.generate(
            judge_conversations,
            judge_generation or generation,
            global_seed + 9_000_001,
        )
        for example_index, judge_output in zip(tied_indices, judge_outputs, strict=True):
            example = examples[example_index]
            judged = extract_answer(judge_output, task, len(example.choices) or 4)
            records[example_index]["judge"] = {"response": judge_output, "answer": judged}
            records[example_index]["final_answer"] = judged
    for example, record in zip(examples, records, strict=True):
        if "final_answer" not in record:
            record["final_answer"] = record["rounds"][-1]["vote"]
        record["final_correct"] = answer_is_correct(
            record["final_answer"], example.answer, task, len(example.choices) or 4
        )
    write_jsonl(output / "traces.jsonl", records)
    summary = compute_readout_metrics(records)
    summary["rounds"] = round_metrics(records)
    summary["settings"] = {
        "agents": agents,
        "debate_rounds": rounds,
        "peer_max_chars": peer_max_chars,
        "debate_prompt": debate_prompt,
        "finalizer": "unique_majority_then_base_judge" if judge_ties else "unique_majority",
        "global_seed": global_seed,
    }
    write_json(output / "summary.json", summary)
    return summary


def _build_records(
    examples: list[Example],
    team: list[TeamMember],
    all_rounds: list[list[list[str]]],
    task: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for example_index, example in enumerate(examples):
        record_rounds: list[dict[str, Any]] = []
        for round_index, agent_responses in enumerate(all_rounds):
            agents: list[dict[str, Any]] = []
            for member, responses in zip(team, agent_responses, strict=True):
                response = responses[example_index]
                answer = extract_answer(response, task, len(example.choices) or 4)
                agents.append(
                    {
                        "candidate_id": member.candidate_id,
                        "response": response,
                        "answer": answer,
                        "correct": answer_is_correct(answer, example.answer, task, len(example.choices) or 4),
                    }
                )
            vote = majority_answer(agent["answer"] for agent in agents)
            record_rounds.append(
                {
                    "round": round_index,
                    "agents": agents,
                    "proposal_hit": any(agent["correct"] for agent in agents),
                    "vote": vote,
                    "vote_correct": answer_is_correct(vote, example.answer, task, len(example.choices) or 4),
                }
            )
        records.append(
            {
                "example_id": example.id,
                "gold": example.answer,
                "initial_correct": [agent["correct"] for agent in record_rounds[0]["agents"]],
                "initial_vote": record_rounds[0]["vote"],
                "initial_vote_correct": record_rounds[0]["vote_correct"],
                "rounds": record_rounds,
            }
        )
    return records


def _debate_messages(
    base: list[dict[str, str]],
    own_response: str,
    peer_responses: list[str],
    peer_max_chars: int,
    prompt_style: str,
) -> list[dict[str, str]]:
    peers = "\n\n".join(
        f"Peer {index + 1}:\n{response[:peer_max_chars]}" for index, response in enumerate(peer_responses)
    )
    if prompt_style == "verify":
        instruction = (
            "Other assistants answered the same question. Verify your previous answer against the peer answers below. "
            "First write a short verification note identifying the decisive agreement or disagreement, then return a "
            "final "
            "answer in the original required format."
        )
    elif prompt_style == "standard":
        instruction = (
            "Other assistants answered the same question. Compare their solutions with yours, revise if needed, and "
            "return "
            "a final answer in the original required format."
        )
    else:
        raise ValueError("prompt_style must be 'verify' or 'standard'")
    return [
        *base,
        {"role": "assistant", "content": own_response},
        {"role": "user", "content": f"{instruction}\n\n{peers}"},
    ]


def _judge_messages(
    base: list[dict[str, str]],
    candidate_responses: list[str],
    peer_max_chars: int,
) -> list[dict[str, str]]:
    candidates = "\n\n".join(
        f"Candidate {index + 1}:\n{response[:peer_max_chars]}" for index, response in enumerate(candidate_responses)
    )
    return [
        *base,
        {
            "role": "user",
            "content": (
                "The candidate answers are tied. Verify which candidate is best supported. Write a short verification "
                "note, "
                "then return exactly one final answer in the original required format.\n\n"
                f"{candidates}"
            ),
        },
    ]
