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
from run_benchmark_quanto import run_quanto_quality
from run_perf_benchmark import run_performance_benchmark

RESULTS_DIR = Path("results")

CONFIGS: list[tuple[str, str]] = [
    ("BF16", "bf16"),
    ("INT8", "int8"),
    ("INT4", "int4"),
]


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


def find_latest_quality_file(config_dir: Path, model_id: str) -> Path | None:
    out_dir = model_output_dir(config_dir, model_id)
    if not out_dir.is_dir():
        return None

    explicit = out_dir / "results.json"
    if explicit.is_file():
        return explicit

    candidates = sorted(
        out_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if candidate.name == "perf_results.json":
            continue
        try:
            data = read_json(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "results" in data:
            return candidate
    return None


def run_command(command: list[str]) -> None:
    print(f"$ {' '.join(command)}")
    subprocess.run(command, check=True)


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
                f"pretrained={model},dtype=float16",
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


def main() -> None:
    args = parse_args()
    platform = args.platform

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
        "include_performance_metrics": not args.skip_perf,
        "configs": {},
    }

    for _, quant in CONFIGS:
        run_data["configs"][quant] = {
            "quality": {
                "status": "pending",
            },
            "performance": {
                "status": "skipped" if args.skip_perf else "pending",
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
        output_path = RESULTS_DIR / quant

        print("============================================")
        if quant == "bf16" and platform == "mps":
            print(f"  [{index}/3] Running FP16 (baseline)")
        elif quant == "bf16":
            print(f"  [{index}/3] Running {display_name} (baseline)")
        else:
            bits = "8-bit" if quant == "int8" else "4-bit"
            print(f"  [{index}/3] Running {display_name} ({bits} quantization)")
        print("============================================")

        run_data["configs"][quant]["quality"]["started_at"] = now_timestamp()
        write_run_file(run_file, run_data)

        quality_cmd: list[str] | None = None
        try:
            if platform == "mps" and quant in {"int8", "int4"}:
                quality_file, quality_payload = run_quanto_quality(
                    model_id=args.model,
                    tasks=args.tasks,
                    batch_size=args.batch_size,
                    output_path=output_path,
                    weights=quant,
                    device="mps",
                    limit=args.limit,
                )
            else:
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
                quality_file = find_latest_quality_file(output_path, args.model)
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

        try:
            perf_file, perf_payload = run_performance_benchmark(
                model_id=args.model,
                quant=quant,
                device=platform,
                output_path=output_path,
            )
        except Exception as exc:
            run_data["configs"][quant]["performance"].update(
                {
                    "status": "failed",
                    "completed_at": now_timestamp(),
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
