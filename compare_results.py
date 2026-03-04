"""Print quality/performance comparison tables from the latest run artifact."""

import argparse
import json
import sys
from pathlib import Path

RESULTS_DIR = Path("results")

CONFIGS = [
    ("BF16", "bf16"),
    ("INT8", "int8"),
    ("INT4", "int4"),
]

PERF_METRICS = [
    ("model_vram_mb", "VRAM Model (MB)"),
    ("peak_vram_mb", "VRAM Peak (MB)"),
    ("avg_tokens_per_sec", "Avg Tokens/sec"),
    ("median_tokens_per_sec", "Median Tokens/sec"),
    ("avg_latency_s", "Avg Latency (s)"),
    ("median_latency_s", "Median Latency (s)"),
]


def find_latest_run_file(results_dir: Path) -> Path | None:
    candidates = [path for path in results_dir.glob("*-results.json") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def task_metric_label(task_name: str, metric_name: str) -> str:
    return f"{task_name} {metric_name}"


def discover_quality_metrics(quality_payloads: list[dict]) -> list[tuple[str, str]]:
    """Discover quality metrics dynamically from lm-eval result payloads."""
    discovered: set[tuple[str, str]] = set()
    for payload in quality_payloads:
        results = payload.get("results")
        if not isinstance(results, dict):
            continue
        for task_name, task_data in results.items():
            if not isinstance(task_data, dict):
                continue
            for metric_key, value in task_data.items():
                if not isinstance(value, (int, float)):
                    continue
                metric_name = metric_key.split(",", 1)[0]
                if metric_name.endswith("_stderr"):
                    continue
                discovered.add((str(task_name), metric_name))
    return sorted(discovered)


def extract_quality_metrics(
    payload: dict,
    metric_specs: list[tuple[str, str]],
) -> dict[str, float | None]:
    """Extract dynamic quality metrics from a quality payload."""
    results = payload.get("results", {})
    metrics: dict[str, float | None] = {}

    for task, metric_name in metric_specs:
        task_data = results.get(task, {}) if isinstance(results, dict) else {}
        display_name = task_metric_label(task, metric_name)
        if not isinstance(task_data, dict):
            metrics[display_name] = None
            continue

        value = task_data.get(metric_name)
        if not isinstance(value, (int, float)):
            value = None
            for key, candidate in task_data.items():
                if key.startswith(f"{metric_name},") and isinstance(
                    candidate, (int, float)
                ):
                    value = candidate
                    break
        metrics[display_name] = value

    return metrics


def extract_perf_metrics(perf_payload: dict) -> dict[str, float | None]:
    """Extract performance metrics from a perf payload."""
    metrics: dict[str, float | None] = {}
    for metric_key, display_name in PERF_METRICS:
        value = perf_payload.get(metric_key)
        metrics[display_name] = value if isinstance(value, (int, float)) else None
    return metrics


def print_table(
    title: str,
    metric_names: list[str],
    all_metrics: dict[str, dict[str, float | None]],
    config_names: list[str],
    fmt: str = ".4f",
) -> None:
    col_width = 14
    metric_col_width = max(len(name) for name in metric_names) + 2

    header = f"{'Metric':<{metric_col_width}}"
    for config_name in config_names:
        header += f"{config_name:>{col_width}}"

    print()
    print("=" * len(header))
    print(f"  {title}")
    print("=" * len(header))
    print()
    print(header)
    print("-" * len(header))

    for metric_name in metric_names:
        row = f"{metric_name:<{metric_col_width}}"
        for config_name in config_names:
            value = all_metrics.get(config_name, {}).get(metric_name)
            if value is not None:
                row += f"{value:>{col_width}{fmt}}"
            else:
                row += f"{'N/A':>{col_width}}"
        print(row)

    print("-" * len(header))
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare benchmark quality and performance results from the latest run.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model ID label to display in table titles.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not RESULTS_DIR.is_dir():
        print(f"Error: Results directory '{RESULTS_DIR}' not found.")
        print("Run the benchmarks first: bash run_benchmarks.sh")
        sys.exit(1)

    run_file = find_latest_run_file(RESULTS_DIR)
    if run_file is None:
        print(
            "No run artifacts found. Run the benchmarks first: bash run_benchmarks.sh"
        )
        sys.exit(1)

    try:
        run_data = json.loads(run_file.read_text())
    except json.JSONDecodeError:
        print(f"Error: Latest run artifact is not valid JSON: {run_file}")
        sys.exit(1)

    if not isinstance(run_data, dict):
        print(f"Error: Latest run artifact has unexpected format: {run_file}")
        sys.exit(1)

    configs_data = run_data.get("configs")
    if not isinstance(configs_data, dict):
        print(f"Error: Latest run artifact is missing 'configs': {run_file}")
        sys.exit(1)

    run_id = run_data.get("run_id")
    model_label = args.model or run_data.get("model") or "Model"
    config_names = [name for name, _ in CONFIGS]

    print(f"Run ID:   {run_id if isinstance(run_id, str) else 'unknown'}")
    print(f"Run file: {run_file}")

    quality_payloads: dict[str, dict] = {}
    perf_payloads: dict[str, dict] = {}

    for config_name, quant in CONFIGS:
        config_data = configs_data.get(quant)
        if not isinstance(config_data, dict):
            continue

        quality = config_data.get("quality")
        if isinstance(quality, dict):
            payload = quality.get("payload")
            if isinstance(payload, dict):
                quality_payloads[config_name] = payload

        performance = config_data.get("performance")
        if isinstance(performance, dict):
            payload = performance.get("payload")
            if isinstance(payload, dict):
                perf_payloads[config_name] = payload

    discovered_metrics = discover_quality_metrics(list(quality_payloads.values()))
    if discovered_metrics:
        quality_metric_names = [
            task_metric_label(task, metric_name)
            for task, metric_name in discovered_metrics
        ]
        all_quality: dict[str, dict[str, float | None]] = {}
        for config_name in config_names:
            payload = quality_payloads.get(config_name)
            if payload is not None:
                all_quality[config_name] = extract_quality_metrics(
                    payload, discovered_metrics
                )
            else:
                all_quality[config_name] = {}

        print_table(
            f"{model_label} Quality Metrics",
            quality_metric_names,
            all_quality,
            config_names,
        )
    else:
        print("No quality metrics found in the latest run artifact.")

    perf_metric_names = [display_name for _, display_name in PERF_METRICS]
    all_perf: dict[str, dict[str, float | None]] = {}
    for config_name in config_names:
        payload = perf_payloads.get(config_name)
        if payload is not None:
            all_perf[config_name] = extract_perf_metrics(payload)
        else:
            all_perf[config_name] = {}

    print_table(
        f"{model_label} Performance Metrics",
        perf_metric_names,
        all_perf,
        config_names,
        fmt=".1f",
    )


if __name__ == "__main__":
    main()
