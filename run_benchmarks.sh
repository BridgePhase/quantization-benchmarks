#!/usr/bin/env bash
set -euo pipefail

echo "Running NLTK preflight checks..."
uv run python - <<'PY'
from __future__ import annotations

import sys


def ensure_resource(resource_path: str, download_name: str) -> None:
    import nltk

    try:
        nltk.data.find(resource_path)
        return
    except LookupError:
        print(f"Missing NLTK resource '{download_name}', downloading...")
        ok = nltk.download(download_name, quiet=False)
        if not ok:
            print(
                f"Failed to download NLTK resource '{download_name}'. "
                "Set NLTK_DATA to a writable path and retry.",
                file=sys.stderr,
            )
            sys.exit(1)


try:
    import nltk  # noqa: F401
except Exception as exc:  # pragma: no cover - defensive import guard
    print(
        "Failed to import nltk. Run 'uv sync' to install dependencies before running benchmarks.",
        file=sys.stderr,
    )
    print(f"Import error: {exc}", file=sys.stderr)
    sys.exit(1)

ensure_resource("tokenizers/punkt_tab", "punkt_tab")
ensure_resource("tokenizers/punkt", "punkt")
PY

uv run python run_benchmarks.py "$@"
