# Repository Guidelines

## Project Structure

- `src/fhe_compare/` contains reusable inference code for CKKS, BGV, BFV, Paillier, shared model handling, and the OpenFHE protocol.
- `scripts/` contains experiments, benchmarks, reports, and demo CLIs. Exploratory scripts record how earlier parameters were selected.
- `tests/` contains pytest coverage. `docs/` explains architecture, decisions, experiment results, limitations, and run commands.
- `results/` stores generated benchmark JSON and tables. `data/`, `artifacts/`, and `demo/` are local or ignored; demo folders may contain secret keys.

## Setup and Common Commands

Use Python 3.12 and `uv`; do not create a separate pip environment.

```sh
uv sync
uv run python scripts/ckks_smoke.py
uv run python scripts/bgv_smoke.py
uv run pytest
uv run python scripts/benchmark.py --threads 1 --reps 5
uv run python scripts/benchmark_paillier.py --reps 3 --key-bits 3072
uv run python scripts/demo_client_server.py --runs ckks:1 bgv:1 bfv:1
```

Benchmarks can take several minutes. Regenerate tables with `uv run python scripts/report.py`; see `docs/runbook.md` for data setup and full regeneration steps.

## Style and Architecture

Use four-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and descriptive scheme module names. Keep cryptographic operations in `src/fhe_compare/`; scripts should call package code. Label cryptographic docstrings `CLIENT` or `SERVER`. No formatter or linter is configured.

## Testing and Results

Use pytest; test files and functions use the `test_*` pattern. No coverage threshold is configured. For scheme changes, compare results with the exact integer reference (BGV/BFV) or plaintext score (CKKS), including boundary inputs. Report commands, parameters, security settings, and machine details. Paillier is linear-only; its decimal size estimates are not OpenFHE binary sizes.

## Commits and Pull Requests

Recent commits use concise imperative subjects such as `Add ...`, sometimes naming the phase. PRs should summarize scope, validation commands and outcomes, and unmeasured limitations. Include relevant tables or plots. Never include patient rows, ignored artifacts, secret keys, or demo exchanges.

## Security and Data Handling

Never print or commit patient rows, secret keys, or `demo/` files. Results are a technical demonstration, not clinical guidance. Preserve the documented split, feature order, preprocessing, model, and polynomial coefficients across schemes; if changed, regenerate dependent results and explain why.
