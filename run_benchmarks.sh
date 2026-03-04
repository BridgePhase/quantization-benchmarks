#!/usr/bin/env bash
set -euo pipefail

uv run python run_benchmarks.py "$@"
