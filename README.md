# When Debate Helps

Independent reference implementation for **When Debate Helps: Proposal Supply and Verification-Aware Readout in Multi-Agent Reasoning**.

The code reproduces the experiment pipeline without depending on the earlier unlicensed RandOpt implementation:

1. sample deterministic full-weight Gaussian perturbations;
2. profile each perturbation on a construction split;
3. select a fixed society with Top-Accuracy, AC-Greedy, or label-free SC-Greedy;
4. run majority vote or multi-round verification-aware debate;
5. save every response and judge trace, then measure proposal hit, recoverable headroom, recovery, and damage.

The author list and canonical citation will be added after the arXiv record is public.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Python 3.10+, PyTorch, and Hugging Face Transformers are required. Model weights and benchmark data are not included. See [docs/DATA.md](docs/DATA.md) for the normalized JSONL schema.

## Debug run

The included arithmetic fixture exercises the complete pipeline with ten perturbations. It is for software validation, not a research result.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_debug.sh
```

To use a local checkpoint or reduce the run further:

```bash
CUDA_VISIBLE_DEVICES=0 wdh seed-bank \
  --config configs/debug.yaml \
  --set seed_bank.seeds_per_sigma=1 \
  --output runs/smoke/seed_bank.jsonl

CUDA_VISIBLE_DEVICES=0 wdh profile \
  --config configs/debug.yaml \
  --set model.name_or_path=/path/to/Qwen2.5-0.5B-Instruct \
  --set model.device=cuda:0 \
  --candidates runs/smoke/seed_bank.jsonl \
  --output runs/smoke/profiles.jsonl
```

## Paper-scale workflow

First prepare disjoint construction and test JSONL files, update their paths in `configs/qwen3_4b_paper.yaml`, and create the seed bank:

```bash
wdh validate-data data/benchmark/train.jsonl
wdh validate-data data/benchmark/test.jsonl
wdh seed-bank \
  --config configs/qwen3_4b_paper.yaml \
  --output runs/qwen3-4b/seed_bank.jsonl
```

Profile on four GPUs. Each process loads one model, evaluates a deterministic shard, and can resume its own JSONL file:

```bash
bash scripts/launch_profile_shards.sh \
  configs/qwen3_4b_paper.yaml \
  runs/qwen3-4b/seed_bank.jsonl \
  runs/qwen3-4b/profile \
  0 1 2 3
```

Select the three fixed teams:

```bash
for method in top_accuracy ac_greedy sc_greedy; do
  wdh select \
    --profiles runs/qwen3-4b/profile/profiles.jsonl \
    --output "runs/qwen3-4b/team-${method}.json" \
    --method "${method}" \
    --team-size 4
done
```

Run five-round debate with all traces and the vote-then-judge tiebreak readout:

```bash
CUDA_VISIBLE_DEVICES=0 wdh evaluate \
  --config configs/qwen3_4b_paper.yaml \
  --set model.device=cuda:0 \
  --team runs/qwen3-4b/team-sc_greedy.json \
  --output-dir runs/qwen3-4b/sc-greedy-r5
```

The same trace format supports the base-model baselines. For example, run SC-30 with no judge, or four-agent Vanilla Debate:

```bash
CUDA_VISIBLE_DEVICES=0 wdh evaluate-base \
  --config configs/qwen3_4b_paper.yaml \
  --set debate.rounds=0 \
  --agents 30 \
  --no-judge-ties \
  --output-dir runs/qwen3-4b/sc30

CUDA_VISIBLE_DEVICES=0 wdh evaluate-base \
  --config configs/qwen3_4b_paper.yaml \
  --agents 4 \
  --output-dir runs/qwen3-4b/vanilla-debate-r5
```

Base-model agents use the stochastic `generation.base` block, while thicket agents and the tie-break judge use their separate deterministic generation blocks.

`summary.json` contains aggregate and round-level metrics. `traces.jsonl` retains initial responses, every debate response, extracted answers, correctness, votes, and tie-break judge outputs. Metrics can be recomputed without generation:

```bash
wdh analyze \
  --traces runs/qwen3-4b/sc-greedy-r5/traces.jsonl \
  --output runs/qwen3-4b/sc-greedy-r5/summary-recomputed.json
```

## Reproducibility controls

- Candidate seeds and scales are generated from the configured global seed and saved to `seed_bank.jsonl` for each run.
- Profile sharding is deterministic by candidate index.
- Existing candidate IDs are skipped when profiling resumes.
- Full-weight perturbations restore from a CPU snapshot, not by subtracting low-precision noise.
- Exact tensor restoration is checked after every candidate by default.
- Invalid answer extractions contribute zero SC-Greedy coverage.
- Run manifests record the resolved config, package versions, device, commit, and parameter counts.

See [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) for method details and memory tradeoffs.

## License

Code and documentation are licensed under Apache-2.0. Model weights and datasets are not covered by this license; see [THIRD_PARTY.md](THIRD_PARTY.md).
