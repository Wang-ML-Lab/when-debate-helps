# Release-Readiness Audit

Reviewed September 27, 2026 against public commit `8b80ac682f5c4348fa1c5dc7355051d57d1b95bc` and the authors' local NeurIPS 2026 camera-ready `main.tex`.

**Assessment: usable as a reference implementation, but not ready to claim full reproduction of the paper.** Two correctness issues need attention, and the benchmark-to-result reproduction path is incomplete. This documentation update records those issues; it does not change experiment code.

## Verified Foundations

- Apache-2.0 license, contribution instructions, third-party notice, and exclusions for generated data and runs are present.
- Public CI passed on the reviewed commit: [tests run](https://github.com/neo1zh/when-debate-helps/actions/runs/36267874873).
- A fresh editable installation succeeded on macOS ARM64 with Python 3.12.14, PyTorch 2.14.0, and Transformers 5.17.0. Ruff passed and all 14 existing tests passed.
- Source distribution and wheel builds succeeded. The build emitted a deprecation warning for the TOML-table license declaration.
- The repository records a prior Qwen2.5-0.5B model smoke run in [RESEARCH_LOG.md](../RESEARCH_LOG.md). This audit did not rerun model generation or paper-scale GPU experiments.
- The README now contains the requested title, owner-confirmed NeurIPS 2026 acceptance, all six authors and affiliations from TeX, Figure 1, and a citation. [CITATION.cff](../CITATION.cff) supplies machine-readable citation metadata.

## Correctness Blockers

### 1. SC-Greedy Uses Labels to Break Ties

**Evidence:** [`selection.py`](https://github.com/neo1zh/when-debate-helps/blob/8b80ac682f5c4348fa1c5dc7355051d57d1b95bc/src/when_debate_helps/selection.py) uses the shared `_greedy_coverage` function for AC-Greedy and SC-Greedy. Candidates are ranked by coverage objective, then `profile.score`, which profiling computes from labeled correctness. Thus the SC-Greedy objective is label-free, but the implemented selector is not.

A minimal executed probe kept candidate predictions fixed at A and B, each with support 0.5. Changing only the gold answer and its derived accuracy scores changed the selected candidate from `c1` to `c2` at the same coverage objective of 0.5.

**Required before claiming label-free selection:** use a label-independent tie rule for SC-Greedy and add a regression test that changing gold labels and accuracy scores cannot change the selected society. Review any results produced with this implementation after the correction.

### 2. Failed Profile Workers Do Not Stop the Shard Launcher

**Evidence:** [`launch_profile_shards.sh`](https://github.com/neo1zh/when-debate-helps/blob/8b80ac682f5c4348fa1c5dc7355051d57d1b95bc/scripts/launch_profile_shards.sh) launches background workers and uses bare `wait`, which does not propagate an individual worker's failure. Merging checks duplicate conflicts but does not check completeness against the seed bank.

An executed shell probe made one worker exit 17 and the other succeed. The launcher still called the merge command and exited 0 when that command succeeded. In a real run, a missing file can make merging fail, but an existing partial or stale shard can allow an incomplete pool to pass through.

**Required:** retain each worker PID, check every exit status, stop before merging after any failure, and verify the merged candidate IDs against the expected seed bank. Test a failure after a partial output has been written.

## Missing Paper-Reproduction Materials

| Priority | Gap and Evidence | Completion Criterion |
| --- | --- | --- |
| High | **Exact pools and configurations.** `configs/qwen3_4b_paper.yaml` creates 300 candidates (100 per scale); the camera-ready seed-profiling appendix reports 500 Qwen candidates with scale counts 160/174/166. There is no OLMo configuration. | Publish the original seed banks, selected teams, model/tokenizer revisions, and configurations for both backbones and each benchmark. Explicitly separate regenerated reference pools from original paper pools. |
| High | **Dataset preparation and splits.** `docs/DATA.md` specifies a generic JSONL schema, and the public configuration points at placeholder paths. There are no benchmark conversion/download scripts or split manifests. | Document official sources, access requirements, versions, split IDs or reproducible split logic, and checksums for AMC12 (360/739 construction/test), MATH500 (200/300), GPQA (200/346), and MMLU-Redux (40/60 per subject across five subjects). Add OlympiadBench (200/474) for appendix reproduction and verify construction/test disjointness. Redistributing restricted data is unnecessary. |
| High | **Budgets, prompts, and reported metrics.** The example workflow uses SC-30 and final R5 readout. The camera-ready main table uses SC@20 and R3 majority vote from five-round traces. The public config gives every task 4096 output tokens; the paper uses 8192 for AMC12/MATH500 and 4096 for GPQA/MMLU-Redux. Prompt text also differs from the camera-ready templates. | Release dataset-specific prompts and budgets, commands selecting the correct round/readout, and aggregation scripts for all eight main-table columns, including the five MMLU subjects. Existing traces contain round-level votes, but final accuracy is a different statistic. |
| High | **Results and mechanism experiments.** No public run artifacts, table/figure regeneration scripts, controlled fixed-proposal LVD probe runner, or complete five-seed robustness recipe are present. | Publish allowed traces or sufficient per-example summaries, manifests, expected aggregate values/tolerances, and scripts reproducing the main table, dynamics figures, controlled probes, and stochastic intervals. Map each paper result to its inputs and command. |
| Medium | **Run provenance and safe resume.** `runtime.py` records Python/PyTorch/CUDA but omits Transformers/tokenizer versions, model revision, and input hashes. Profiling resumes by candidate ID alone; generated candidate IDs can recur across different pools/configurations. | Record the full tested environment and input identities. Refuse resume when model, dataset, generation settings, or seed/sigma mapping differs. Bind profiles and selected teams to their construction data and seed bank. |
| Medium | **Integration tests and environment coverage.** Existing tests cover helpers, profiling, and perturbation restoration; they do not exercise a complete debate/judge/CLI run. CI tests Python 3.11 despite a Python >=3.10 claim, and dependencies have open lower bounds. | Add a small deterministic integration fixture covering tied and untied votes, debate rounds, trace serialization, and offline recomputation. Record a tested dependency lock or constraints file, test the advertised minimum Python version, and document measured GPU/host-RAM requirements and runtime. |
| Medium | **Versioned distribution and paper links.** GitHub has no tags or releases as of the audit. There is no public paper/proceedings link. Package builds omit configs, scripts, examples, and figure assets from the source archive, so README workflows do not work from that archive alone. | Publish a tagged release with release notes after blockers are resolved; supply a self-contained reproduction bundle or clearly require a Git checkout. Add canonical paper identifiers when public. Modernize the deprecated package license metadata before publishing packages. |

## Validation Scope

The installation, lint, 14 unit tests, package builds, SC-Greedy tie probe, and failed-worker probe were run during this audit. The Figure 1 PNG was rendered from the exact PDF referenced by the first figure in `main.tex` and visually checked. No benchmark accuracy, GPU resource estimate, or paper-result equivalence is asserted by these software checks.
