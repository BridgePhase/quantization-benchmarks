# AGENTS.md

Guidance for coding agents working in this repository.

## Project Snapshot

- Language: Python 3.11+
- Package manager / runner: `uv`
- Main entrypoints:
  - `run_benchmarks.sh` (end-to-end quality + perf benchmarks)
  - `run_perf_benchmark.py` (performance benchmark)
  - `compare_results.py` (aggregate and print comparison tables)
- Output artifacts are written under `results/` (gitignored).

## Rule Files (Cursor/Copilot)

- `.cursorrules`: not found
- `.cursor/rules/`: not found
- `.github/copilot-instructions.md`: not found

If any of the above files are added later, treat them as higher-priority
agent instructions and update this file accordingly.

## Environment and Setup

1. Use Python 3.11 or newer.
2. Install dependencies with platform extras:
   - CUDA: `uv sync --extra cuda`
   - Apple Silicon (MPS): `uv sync`
3. For gated models, authenticate once:
   - `uv run huggingface-cli login`

## Build / Lint / Test Commands

This repo does not currently define dedicated lint/test tooling in
`pyproject.toml` (no configured `pytest`, `ruff`, `mypy`, etc.).

Use the commands below for validation and benchmarking.

### Build / Dependency Sync

- Install/update env (CUDA): `uv sync --extra cuda`
- Install/update env (MPS): `uv sync`
- Lockfile refresh (if needed): `uv lock`

### Run Benchmarks (Primary Validation)

- Auto-detect platform and run all configs:
  - `bash run_benchmarks.sh`
- Force platform:
  - `bash run_benchmarks.sh cuda`
  - `bash run_benchmarks.sh mps`
- Quality-only run (skip perf):
  - `bash run_benchmarks.sh mps --skip-perf`

### Run a Single "Test-Like" Unit of Work

There is no formal unit-test suite yet. For focused validation, run one task
or one script invocation:

- Single lm-eval task via benchmark script (edit `TASKS` first):
  - set `TASKS="hellaswag"` (or another single task) in `run_benchmarks.sh`
  - run `bash run_benchmarks.sh mps --skip-perf`
- Single performance benchmark invocation:
  - `uv run python run_perf_benchmark.py --model <model> --quant bf16 --device mps --output-path results/bf16`
- Note: MPS runs are BF16-only; INT8/INT4 quantization is unsupported on MPS.

### Compare Results

- `uv run python compare_results.py`

### If/When Pytest Tests Are Added

Agents should use these conventional commands:

- Run all tests: `uv run pytest`
- Run a file: `uv run pytest tests/test_file.py`
- Run a single test: `uv run pytest tests/test_file.py::test_case_name -q`
- Run by keyword: `uv run pytest -k "keyword"`

## Code Style Guidelines

Follow existing code patterns in this repository unless explicitly asked to
introduce a new standard.

### Imports

- Group imports as:
  1) standard library
  2) third-party
  3) local modules
- Keep one import per logical module line (avoid wildcard imports).
- Prefer explicit imports (e.g., `from pathlib import Path`).

### Formatting

- Use 4-space indentation.
- Prefer double quotes for strings (matches current files).
- Keep lines readable; target ~88-100 chars where practical.
- Use trailing commas in multiline literals/calls when it improves diffs.
- Preserve shebang + `set -euo pipefail` style in shell scripts.

### Types and Signatures

- Add type hints to new/modified Python functions.
- Use modern built-in generics (`list[str]`, `dict[str, float | None]`).
- Annotate return types on public/top-level functions.
- Keep argparse return type as `argparse.Namespace` where applicable.

### Naming Conventions

- Modules/files: `snake_case.py`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- CLI flags: `kebab-case` long options (e.g., `--output-path`)
- Keep quantization labels consistent: `bf16`, `int8`, `int4`.

### Data and Paths

- Use `pathlib.Path` for filesystem paths in Python.
- Create output directories with `mkdir(parents=True, exist_ok=True)`.
- Write JSON with indentation for readability (`indent=2`).
- Keep benchmark outputs under `results/` and do not commit them.

### Error Handling and Exits

- Fail fast with clear actionable messages.
- For CLI scripts, validate required runtime dependencies early.
- Use explicit non-zero exits for hard failures (`sys.exit(1)`).
- Catch specific exceptions (avoid bare `except:`).
- In shell scripts, retain strict mode: `set -euo pipefail`.

### Logging / Console Output

- Use concise progress prints for long-running operations.
- Include enough context in status messages (model, quant, device, step).
- Keep output deterministic and parse-friendly where possible.

### Benchmarking and Reproducibility

- Preserve current methodology unless intentionally changed:
  - warmup before measurement
  - explicit device synchronization
  - greedy decoding for perf (`do_sample=False`)
- If methodology changes, update `README.md` and relevant script help text.

### Documentation and Comments

- Keep top-level module docstrings for scripts.
- Add comments only for non-obvious logic or backend-specific workarounds.
- When changing CLI args or defaults, update `README.md` examples.

## Agent Workflow Expectations

- Before editing, read `README.md`, `pyproject.toml`, and target scripts.
- Prefer minimal, surgical changes over broad refactors.
- Do not introduce new dependencies unless necessary and justified.
- If adding lint/test tools, document commands in this file and `README.md`.
- Validate touched code paths with the narrowest practical command.

## Quick Command Reference

- Full benchmark: `bash run_benchmarks.sh`
- Quality only: `bash run_benchmarks.sh mps --skip-perf`
- Perf only: `uv run python run_perf_benchmark.py ...`
- Compare: `uv run python compare_results.py`
- Future single test pattern: `uv run pytest path/to/test.py::test_name -q`
