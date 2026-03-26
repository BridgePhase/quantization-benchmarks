"""Run quality and performance benchmarks across quantization configs.

This script orchestrates BF16/INT8/INT4 benchmark runs and writes one
run-level results file under results/<timestamp>-results.json. The file is
updated after each quality/performance step completes.
"""

import argparse
import getpass
import json
import socket
import subprocess
import sys
from pathlib import Path

from benchmark_common import (
    add_batch_size_arg,
    add_model_arg,
    add_tasks_arg,
    detect_platform_or_exit,
    model_output_dir,
    now_timestamp,
    parse_tasks,
    read_json,
    validate_platform_dependencies_or_exit,
)

RESULTS_DIR = Path("results")

CONFIGS: list[tuple[str, str]] = [
    ("BF16", "bf16"),
    ("INT8", "int8"),
    ("INT4", "int4"),
]
QUANT_CHOICES = [quant for _, quant in CONFIGS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run model quantization quality/performance benchmarks.",
    )
    parser.add_argument(
        "platform",
        nargs="?",
        choices=["cuda", "mps"],
        help="Target platform; if omitted, auto-detect CUDA then MPS.",
    )
    parser.add_argument(
        "--skip-perf",
        action="store_true",
        help="Skip performance benchmarks and run quality only.",
    )
    add_model_arg(parser)
    add_tasks_arg(parser)
    add_batch_size_arg(parser)
    parser.add_argument(
        "--quant",
        action="append",
        default=None,
        help=(
            "Quantization config(s) to run: bf16, int8, int4. "
            "Accepts repeated flags or comma-separated values."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional lm-eval sample limit for faster smoke tests.",
    )
    return parser.parse_args()


def write_run_file(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(json.dumps(data, indent=2, default=str))
    temp_path.replace(path)


def find_latest_quality_file(output_dir: Path) -> Path | None:
    candidates: list[Path] = []
    if output_dir.is_dir():
        candidates.extend(output_dir.rglob("*.json"))

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if candidate.name in {"perf_results.json"}:
            continue
        try:
            data = read_json(candidate)
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and "results" in data:
            return candidate
    return None


def run_command(command: list[str]) -> None:
    print(f"$ {' '.join(command)}")
    subprocess.run(command, check=True)


def parse_selected_quants(raw_values: list[str] | None) -> list[str]:
    if not raw_values:
        return QUANT_CHOICES

    selected: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        for value in raw.split(","):
            quant = value.strip().lower()
            if not quant:
                continue
            if quant not in QUANT_CHOICES:
                choices = ", ".join(QUANT_CHOICES)
                raise ValueError(
                    f"Invalid --quant '{quant}'. Expected one of: {choices}"
                )
            if quant not in seen:
                selected.append(quant)
                seen.add(quant)

    if not selected:
        choices = ", ".join(QUANT_CHOICES)
        raise ValueError(
            f"No valid --quant values provided. Expected one of: {choices}"
        )

    return selected


def build_lm_eval_quality_command(
    platform: str,
    model: str,
    tasks: str,
    batch_size: str,
    quant: str,
    output_path: Path,
    limit: int | None,
) -> list[str]:
    if quant == "bf16":
        if platform == "mps":
            command = [
                "uv",
                "run",
                "lm_eval",
                "--model",
                "hf",
                "--model_args",
                f"pretrained={model},dtype=bfloat16",
                "--tasks",
                tasks,
                "--batch_size",
                batch_size,
                "--output_path",
                str(output_path),
                "--log_samples",
                "--device",
                "mps",
            ]
            if limit is not None:
                command.extend(["--limit", str(limit)])
            return command
        command = [
            "uv",
            "run",
            "lm_eval",
            "--model",
            "hf",
            "--model_args",
            f"pretrained={model},dtype=bfloat16",
            "--tasks",
            tasks,
            "--batch_size",
            batch_size,
            "--output_path",
            str(output_path),
            "--log_samples",
        ]
        if limit is not None:
            command.extend(["--limit", str(limit)])
        return command

    load_flag = "load_in_8bit=True" if quant == "int8" else "load_in_4bit=True"
    command = [
        "uv",
        "run",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model},{load_flag}",
        "--tasks",
        tasks,
        "--batch_size",
        batch_size,
        "--output_path",
        str(output_path),
        "--log_samples",
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    return command


def build_perf_benchmark_command(
    model: str,
    quant: str,
    device: str,
    output_path: Path,
) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "run_perf_benchmark.py",
        "--model",
        model,
        "--quant",
        quant,
        "--device",
        device,
        "--output-path",
        str(output_path),
    ]


def main() -> None:
    args = parse_args()
    platform = args.platform

    try:
        selected_quants = parse_selected_quants(args.quant)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    if platform is None:
        print("No platform specified, auto-detecting...")
        platform = detect_platform_or_exit()
        print(f"Detected platform: {platform}")

    validate_platform_dependencies_or_exit(platform)

    started_at = now_timestamp()
    run_id = f"{getpass.getuser()}@{socket.gethostname()}-{started_at}"
    run_file = RESULTS_DIR / f"{started_at}-results.json"
    task_list = parse_tasks(args.tasks)

    run_data: dict = {
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": None,
        "status": "running",
        "device": platform,
        "model": args.model,
        "tasks": task_list,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "quants": selected_quants,
        "include_performance_metrics": not args.skip_perf,
        "configs": {},
    }

    for _, quant in CONFIGS:
        selected = quant in selected_quants
        unsupported_on_mps = platform == "mps" and quant in {"int8", "int4"}
        skipped_reason = None
        if not selected:
            skipped_reason = "Not selected via --quant."
        elif unsupported_on_mps:
            skipped_reason = "Quantization is disabled on MPS; BF16 baseline only."

        run_data["configs"][quant] = {
            "quality": {
                "status": "skipped" if skipped_reason else "pending",
                "reason": skipped_reason,
            },
            "performance": {
                "status": (
                    "skipped" if args.skip_perf or skipped_reason else "pending"
                ),
                "reason": (
                    "Skipped via --skip-perf." if args.skip_perf else skipped_reason
                ),
            },
        }

    write_run_file(run_file, run_data)

    print("============================================")
    print("  Model Quantization Benchmarks")
    print("============================================")
    print()
    print(f"Model:     {args.model}")
    print(f"Tasks:     {args.tasks}")
    print(f"Batch:     {args.batch_size}")
    print(f"Quant:     {','.join(selected_quants)}")
    print(f"Limit:     {args.limit if args.limit is not None else 'none'}")
    print(f"Platform:  {platform}")
    print(
        "Perf:      SKIPPED (--skip-perf)" if args.skip_perf else "Perf:      enabled"
    )
    print(f"Run file:  {run_file}")
    print()
    print("REMINDER: If your selected model is gated, accept its license on")
    print("huggingface.co and log in via: uv run huggingface-cli login")
    print()

    for index, (display_name, quant) in enumerate(CONFIGS, start=1):
        output_path = model_output_dir(RESULTS_DIR, args.model) / quant
        selected = quant in selected_quants
        unsupported_on_mps = platform == "mps" and quant in {"int8", "int4"}

        print("============================================")
        if quant == "bf16" and platform == "mps":
            print(f"  [{index}/3] Running BF16 (baseline)")
        elif quant == "bf16":
            print(f"  [{index}/3] Running {display_name} (baseline)")
        else:
            bits = "8-bit" if quant == "int8" else "4-bit"
            print(f"  [{index}/3] Running {display_name} ({bits} quantization)")
        print("============================================")

        if not selected:
            print("  Skipping: not selected via --quant.")
            print()
            continue

        if unsupported_on_mps:
            print("  Skipping: quantization is disabled on MPS (BF16 only).")
            print()
            continue

        run_data["configs"][quant]["quality"]["started_at"] = now_timestamp()
        write_run_file(run_file, run_data)

        quality_cmd: list[str] | None = None
        try:
            quality_cmd = build_lm_eval_quality_command(
                platform=platform,
                model=args.model,
                tasks=args.tasks,
                batch_size=args.batch_size,
                quant=quant,
                output_path=output_path,
                limit=args.limit,
            )
            run_command(quality_cmd)
            quality_file = find_latest_quality_file(output_path)
            quality_payload = read_json(quality_file) if quality_file else None
        except Exception as exc:
            run_data["configs"][quant]["quality"].update(
                {
                    "status": "failed",
                    "completed_at": now_timestamp(),
                    "command": quality_cmd,
                    "return_code": getattr(exc, "returncode", None),
                    "error": str(exc),
                }
            )
            run_data["status"] = "failed"
            run_data["completed_at"] = now_timestamp()
            write_run_file(run_file, run_data)
            raise

        run_data["configs"][quant]["quality"].update(
            {
                "status": "completed",
                "completed_at": now_timestamp(),
                "source_file": str(quality_file) if quality_file else None,
                "payload": quality_payload,
            }
        )
        write_run_file(run_file, run_data)

        if args.skip_perf:
            print()
            continue

        print()
        print(f"  Running {display_name} performance benchmark...")
        run_data["configs"][quant]["performance"]["started_at"] = now_timestamp()
        write_run_file(run_file, run_data)

        perf_cmd = build_perf_benchmark_command(
            model=args.model,
            quant=quant,
            device=platform,
            output_path=output_path,
        )
        perf_file = output_path / "perf_results.json"

        try:
            # Run each perf benchmark in a fresh subprocess so CUDA allocator
            # state and loaded model weights from earlier quant runs cannot
            # inflate the next run's VRAM measurement.
            run_command(perf_cmd)
            perf_payload = read_json(perf_file)
        except Exception as exc:
            run_data["configs"][quant]["performance"].update(
                {
                    "status": "failed",
                    "completed_at": now_timestamp(),
                    "command": perf_cmd,
                    "return_code": getattr(exc, "returncode", None),
                    "error": str(exc),
                }
            )
            run_data["status"] = "failed"
            run_data["completed_at"] = now_timestamp()
            write_run_file(run_file, run_data)
            raise

        run_data["configs"][quant]["performance"].update(
            {
                "status": "completed",
                "completed_at": now_timestamp(),
                "source_file": str(perf_file),
                "payload": perf_payload,
            }
        )
        write_run_file(run_file, run_data)
        print()

    run_data["status"] = "completed"
    run_data["completed_at"] = now_timestamp()
    write_run_file(run_file, run_data)

    print("============================================")
    print("  All benchmarks complete!")
    print("  Compare results: uv run python compare_results.py")
    print(f"  Run artifact: {run_file}")
    print("============================================")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as err:
        sys.exit(err.returncode)
