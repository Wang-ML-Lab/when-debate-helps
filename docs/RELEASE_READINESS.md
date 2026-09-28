# Release-Readiness Audit

Reviewed and updated September 27, 2026 against the authors' NeurIPS 2026 camera-ready `main.tex`.

**Assessment: ready as an independent reference implementation of the core profiling, selection, and R3 debate workflow.** This repository does not claim to be a complete archival artifact for every paper analysis.

## Verified Foundations

- Apache-2.0 license, contribution instructions, third-party notice, and exclusions for generated data and runs are present.
- A fresh editable installation, Ruff, the unit and regression tests, and package builds pass on the audited revision.
- The README contains the NeurIPS 2026 title and acceptance status, all six authors and affiliations from the paper TeX, and Figure 1.
- The reference Qwen configuration creates 500 candidates across noise scales `0.0005`, `0.001`, and `0.002`, with camera-ready counts 160, 174, and 166.
- The documented workflow uses SC@20 and three debate rounds. Output budgets match the camera-ready setup: 8192 tokens for AMC12/MATH500 and 4096 for GPQA/MMLU-Redux.

## Resolved Findings

### Label-Free SC-Greedy Tie-Breaking

SC-Greedy previously shared AC-Greedy's labeled-accuracy tie-break. It now collects candidates tied on the coverage objective and chooses among them with a seeded pseudorandom generator. `--tie-break-seed` defaults to 42 and is recorded in the selected-team JSON. A regression test holds predictions fixed, reverses accuracy and correctness labels, and verifies that the selected candidate does not change.

This makes the selection path label-free while keeping runs reproducible. AC-Greedy continues to use labeled construction accuracy because AC-Greedy is the label-based method.

### Profiling Worker Failure and Merge Completeness

The multi-GPU launcher now records every background process, checks every exit status, and exits before merging if any shard fails. The merge command receives the seed bank and requires the merged candidate-ID set to match it exactly. Regression tests cover both a failed worker and an incomplete merge.

## Public Release Scope

The public repository covers the core experiment path:

1. deterministic perturbation-bank construction;
2. candidate profiling and validated multi-GPU sharding;
3. Top-Accuracy, AC-Greedy, and label-free SC-Greedy society selection;
4. SC@20, majority vote, and R3 verification-aware debate;
5. complete response traces and recovery/damage metrics.

Figure-generation scripts, controlled fixed-proposal Probe runners, internal table assembly, and private experiment-management utilities are outside this code release. They are not required to run the public core pipeline. Model weights, licensed benchmark data, and raw paper runs are also not redistributed.

Users preparing paper-comparable runs must supply the benchmark splits described in the camera-ready paper, use the matching dataset prompt and output budget, and retain the resolved configuration and generated seed bank with each run.

## Token-Budget Note

The token budget matters whenever generation reaches the limit. Reducing AMC12 or MATH500 from 8192 to 4096 can truncate long solutions and change extracted answers, coverage profiles, selected societies, and downstream debate accuracy. For responses that finish before 4096 tokens, the larger cap alone does not alter deterministic decoding. GPQA and MMLU-Redux remain at the paper's 4096-token budget.

## Validation Boundary

Software tests verify selection independence from labels, deterministic seed-bank size and scale counts, worker-failure propagation, merge completeness, trace metrics, and package integrity. Full Qwen3-4B and OLMo-3-7B paper runs require the corresponding model weights, benchmark access, and GPU resources and were not rerun as part of this software audit.
