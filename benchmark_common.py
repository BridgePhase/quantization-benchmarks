"""Shared helpers for benchmark scripts."""

import argparse
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_MODEL = "meta-llama/Llama-3.2-3B"
DEFAULT_TASKS = "ifeval,hellaswag"
DEFAULT_BATCH_SIZE = "1"


def now_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H:%M:%S")


def parse_tasks(tasks: str) -> list[str]:
    return [task.strip() for task in tasks.split(",") if task.strip()]


def model_output_dir(output_path: str | Path, model_id: str) -> Path:
    return Path(output_path) / model_id.replace("/", "__")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def detect_platform_or_exit() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"

    print("Error: No supported GPU detected (checked CUDA and MPS).")
    print(
        "Usage: bash run_benchmarks.sh [cuda|mps] [--skip-perf] "
        "[--model <model>] [--tasks <tasks>] [--batch-size <size>]"
    )
    sys.exit(1)


def validate_platform_dependencies_or_exit(platform: str) -> None:
    if platform == "cuda":
        try:
            importlib.import_module("bitsandbytes")
        except ModuleNotFoundError:
            print("Error: bitsandbytes is not installed. Install it with:")
            print("  uv sync --extra cuda")
            sys.exit(1)
    elif platform == "mps":
        try:
            importlib.import_module("optimum.quanto")
        except ModuleNotFoundError:
            print("Error: optimum-quanto is not installed. Install it with:")
            print("  uv sync --extra mps")
            sys.exit(1)


def add_model_arg(
    parser: argparse.ArgumentParser,
    *,
    required: bool = False,
    default: str | None = None,
) -> None:
    kwargs: dict = {
        "type": str,
        "help": "HuggingFace model ID.",
    }
    if required:
        kwargs["required"] = True
    else:
        kwargs["default"] = DEFAULT_MODEL if default is None else default
    parser.add_argument("--model", **kwargs)


def add_tasks_arg(
    parser: argparse.ArgumentParser,
    *,
    required: bool = False,
    default: str | None = None,
) -> None:
    kwargs: dict = {
        "type": str,
        "help": "Comma-separated lm-eval task list.",
    }
    if required:
        kwargs["required"] = True
    else:
        kwargs["default"] = DEFAULT_TASKS if default is None else default
    parser.add_argument("--tasks", **kwargs)


def add_batch_size_arg(
    parser: argparse.ArgumentParser,
    *,
    required: bool = False,
    default: str | None = None,
) -> None:
    kwargs: dict = {
        "dest": "batch_size",
        "type": str,
        "help": "Batch size for evaluation.",
    }
    if required:
        kwargs["required"] = True
    else:
        kwargs["default"] = DEFAULT_BATCH_SIZE if default is None else default
    parser.add_argument("--batch-size", "--batch_size", **kwargs)


def add_device_arg(
    parser: argparse.ArgumentParser,
    *,
    required: bool = False,
    default: str | None = None,
    choices: list[str] | None = None,
) -> None:
    kwargs: dict = {
        "type": str,
        "help": "Device to run on.",
    }
    if choices is not None:
        kwargs["choices"] = choices
    if required:
        kwargs["required"] = True
    elif default is not None:
        kwargs["default"] = default
    parser.add_argument("--device", **kwargs)


def add_output_path_arg(
    parser: argparse.ArgumentParser, *, required: bool = True
) -> None:
    kwargs: dict = {
        "dest": "output_path",
        "type": str,
        "help": "Output directory path.",
    }
    if required:
        kwargs["required"] = True
    parser.add_argument("--output-path", "--output_path", **kwargs)
