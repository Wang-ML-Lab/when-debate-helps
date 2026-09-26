# Research log

This file records completed public-repository validation runs. Paper results will be linked after the arXiv release and artifact export.

## 2026-09-26: independent release implementation

- Implemented the `profile -> select -> evaluate -> analyze` pipeline.
- Added exact restoration checks after every full-weight perturbation.
- Added deterministic sharding and resumable profile outputs.
- Added Top-Accuracy, AC-Greedy, and label-free SC-Greedy team selection.
- Added complete debate and judge traces plus recovery/damage diagnostics.

Validation completed with:

```bash
ruff check .
pytest -q
CUDA_VISIBLE_DEVICES=3 wdh profile ...
CUDA_VISIBLE_DEVICES=3 wdh evaluate ...
wdh analyze ...
```

- Ruff passed with no findings.
- All 13 unit tests passed.
- A real-model smoke run used the local Qwen2.5-0.5B-Instruct checkpoint, four perturbations, four construction examples, and three held-out examples.
- Profiling covered all 494,032,768 model parameters across 290 tensors and completed exact restoration checks for every candidate.
- SC-Greedy selection, one debate round, tie-break judging, trace serialization, and offline metric recomputation completed successfully.
- The fixture uses a 32-token generation limit, so its accuracy is not a research result.
