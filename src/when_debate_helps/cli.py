from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_config, require
from .data import load_examples
from .debate import evaluate_base, evaluate_thicket, load_team
from .io import iter_jsonl, write_json
from .metrics import compute_readout_metrics, round_metrics
from .model import TransformersGenerator, generation_config
from .perturbation import GaussianPerturber
from .profile import merge_profile_shards, profile_candidates
from .runtime import parameter_counts, save_manifest
from .seeds import save_seed_bank
from .selection import select_and_save


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wdh", description="When Debate Helps reproducibility CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-data", help="Validate normalized JSONL input")
    validate.add_argument("path")
    validate.add_argument("--limit", type=int)

    seed_bank = subparsers.add_parser("seed-bank", help="Create a deterministic perturbation bank")
    _add_config(seed_bank)
    seed_bank.add_argument("--output", required=True)

    profile = subparsers.add_parser("profile", help="Profile perturbation candidates")
    _add_config(profile)
    profile.add_argument("--candidates", required=True)
    profile.add_argument("--output", required=True)
    profile.add_argument("--shard-id", type=int, default=0)
    profile.add_argument("--num-shards", type=int, default=1)
    profile.add_argument("--no-resume", action="store_true")

    merge = subparsers.add_parser("merge-profiles", help="Merge profile JSONL shards")
    merge.add_argument("--inputs", nargs="+", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--candidates", help="Validate merged candidate IDs against this seed bank")

    select = subparsers.add_parser("select", help="Select a fixed debate team")
    select.add_argument("--profiles", required=True)
    select.add_argument("--output", required=True)
    select.add_argument("--method", choices=["top_accuracy", "ac_greedy", "sc_greedy"], required=True)
    select.add_argument("--team-size", type=int, default=4)
    select.add_argument("--tie-break-seed", type=int, default=42)

    evaluate = subparsers.add_parser("evaluate", help="Run thicket vote/debate with full traces")
    _add_config(evaluate)
    evaluate.add_argument("--team", required=True)
    evaluate.add_argument("--output-dir", required=True)

    evaluate_base_parser = subparsers.add_parser(
        "evaluate-base", help="Run base-model self-consistency or sampled Vanilla Debate"
    )
    _add_config(evaluate_base_parser)
    evaluate_base_parser.add_argument("--agents", type=int, required=True)
    evaluate_base_parser.add_argument("--output-dir", required=True)
    evaluate_base_parser.add_argument("--no-judge-ties", action="store_true")

    analyze = subparsers.add_parser("analyze", help="Recompute metrics from saved traces")
    analyze.add_argument("--traces", required=True)
    analyze.add_argument("--output")
    return parser


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")


def _load_runtime(config: dict[str, Any]) -> tuple[TransformersGenerator, GaussianPerturber]:
    generator = TransformersGenerator.from_config(require(config, "model"))
    perturbation = config.get("perturbation", {})
    perturber = GaussianPerturber(
        generator.model,
        include=perturbation.get("include"),
        exclude=perturbation.get("exclude"),
        scale_mode=str(perturbation.get("scale_mode", "absolute")),
        verify_restore=bool(perturbation.get("verify_restore", True)),
    )
    counts = parameter_counts(generator.model, perturber.stats.parameters)
    print(json.dumps({"parameter_counts": counts, "perturbed_tensors": perturber.stats.tensors}, indent=2))
    return generator, perturber


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "validate-data":
        examples = load_examples(args.path, args.limit)
        print(json.dumps({"path": args.path, "examples": len(examples)}, indent=2))
        return
    if args.command == "merge-profiles":
        merge_profile_shards(args.inputs, args.output, args.candidates)
        return
    if args.command == "select":
        select_and_save(args.profiles, args.output, args.method, args.team_size, args.tie_break_seed)
        return
    if args.command == "analyze":
        records = list(iter_jsonl(args.traces))
        summary = compute_readout_metrics(records)
        summary["rounds"] = round_metrics(records)
        if args.output:
            write_json(args.output, summary)
        print(json.dumps(summary, indent=2))
        return

    config = load_config(args.config, args.set)
    if args.command == "seed-bank":
        bank = require(config, "seed_bank")
        counts = bank.get("counts_per_sigma")
        save_seed_bank(
            args.output,
            list(bank["sigmas"]),
            int(bank["seeds_per_sigma"]) if bank.get("seeds_per_sigma") is not None else None,
            int(bank["global_seed"]),
            [int(count) for count in counts] if counts is not None else None,
        )
        return
    if args.command == "profile":
        generator, perturber = _load_runtime(config)
        examples = load_examples(
            require(config, "data.construction_path"),
            config.get("data", {}).get("construction_limit"),
        )
        profile_candidates(
            generator=generator,
            perturber=perturber,
            examples=examples,
            candidates_path=args.candidates,
            output_path=args.output,
            task=require(config, "data.task"),
            generation=generation_config(require(config, "generation.profile")),
            system_prompt=config.get("prompts", {}).get("system"),
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            resume=not args.no_resume,
        )
        save_manifest(
            Path(args.output).with_suffix(".manifest.json"),
            config,
            parameter_counts(generator.model, perturber.stats.parameters),
        )
        return
    if args.command == "evaluate":
        generator, perturber = _load_runtime(config)
        examples = load_examples(require(config, "data.test_path"), config.get("data", {}).get("test_limit"))
        debate = require(config, "debate")
        summary = evaluate_thicket(
            generator=generator,
            perturber=perturber,
            examples=examples,
            team=load_team(args.team),
            task=require(config, "data.task"),
            generation=generation_config(require(config, "generation.eval")),
            judge_generation=generation_config(
                config.get("generation", {}).get("judge", require(config, "generation.eval"))
            ),
            output_dir=args.output_dir,
            rounds=int(debate.get("rounds", 3)),
            peer_max_chars=int(debate.get("peer_max_chars", 800)),
            system_prompt=config.get("prompts", {}).get("system"),
            debate_prompt=str(debate.get("prompt", "verify")),
            global_seed=int(debate.get("global_seed", 42)),
        )
        save_manifest(
            Path(args.output_dir) / "manifest.json",
            config,
            parameter_counts(generator.model, perturber.stats.parameters),
        )
        print(json.dumps(summary, indent=2))
        return
    if args.command == "evaluate-base":
        generator = TransformersGenerator.from_config(require(config, "model"))
        examples = load_examples(require(config, "data.test_path"), config.get("data", {}).get("test_limit"))
        debate = require(config, "debate")
        summary = evaluate_base(
            generator=generator,
            examples=examples,
            agents=args.agents,
            task=require(config, "data.task"),
            generation=generation_config(config.get("generation", {}).get("base", require(config, "generation.eval"))),
            judge_generation=generation_config(
                config.get("generation", {}).get("judge", require(config, "generation.eval"))
            ),
            output_dir=args.output_dir,
            rounds=int(debate.get("rounds", 3)),
            peer_max_chars=int(debate.get("peer_max_chars", 800)),
            system_prompt=config.get("prompts", {}).get("system"),
            debate_prompt=str(debate.get("prompt", "verify")),
            judge_ties=not args.no_judge_ties,
            global_seed=int(debate.get("global_seed", 42)),
        )
        save_manifest(Path(args.output_dir) / "manifest.json", config, parameter_counts(generator.model))
        print(json.dumps(summary, indent=2))
        return
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
