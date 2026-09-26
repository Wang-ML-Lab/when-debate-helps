# Contributing

Use Python 3.10 or newer. Install development dependencies with `pip install -e '.[dev]'`, then run `ruff check .` and `pytest -q` before opening a pull request.

Keep generated benchmark data, model weights, and raw runs out of Git. New experiment logic should include a focused unit test and a small smoke command. Bug reports should include the resolved configuration, `manifest.json`, and the smallest trace that reproduces the issue.
