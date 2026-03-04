"""Parse lm-eval results and print a comparison table across quantization configs."""

import json
import sys
from pathlib import Path

RESULTS_DIR = Path("results")

CONFIGS = [
    ("BF16", "bf16"),
    ("INT8", "int8"),
    ("INT4", "int4"),
]

# Metrics to extract per task
TASK_METRICS = {
    "hellaswag": [
        ("acc_norm", "HellaSwag acc_norm"),
    ],
    "ifeval": [
        ("prompt_level_strict_acc", "IFEval prompt_strict"),
        ("inst_level_strict_acc", "IFEval inst_strict"),
        ("prompt_level_loose_acc", "IFEval prompt_loose"),
        ("inst_level_loose_acc", "IFEval inst_loose"),
    ],
}

PERF_METRICS = [
    ("model_vram_mb", "VRAM Model (MB)"),
    ("peak_vram_mb", "VRAM Peak (MB)"),
    ("avg_tokens_per_sec", "Avg Tokens/sec"),
    ("median_tokens_per_sec", "Median Tokens/sec"),
    ("avg_latency_s", "Avg Latency (s)"),
    ("median_latency_s", "Median Latency (s)"),
]


def find_results_file(config_dir: Path) -> Path | None:
    """Find the results JSON file inside a config's output directory.

    lm-eval writes results to: <output_path>/<model_name>/<filename>.json
    We look for any JSON file containing a "results" key.
    """
    if not config_dir.is_dir():
        return None
    for json_file in sorted(config_dir.rglob("results.json")):
        return json_file
    # Fallback: look for any JSON with a "results" key
    for json_file in sorted(config_dir.rglob("*.json")):
        try:
            data = json.loads(json_file.read_text())
            if "results" in data:
                return json_file
        except (json.JSONDecodeError, KeyError):
            continue
    return None


def find_perf_results_file(config_dir: Path) -> Path | None:
    """Find perf_results.json inside a config's output directory."""
    if not config_dir.is_dir():
        return None
    for json_file in sorted(config_dir.rglob("perf_results.json")):
        return json_file
    return None


def extract_metrics(results_file: Path) -> dict[str, float | None]:
    """Extract all tracked metrics from a results JSON file."""
    data = json.loads(results_file.read_text())
    results = data.get("results", {})
    metrics = {}

    for task, task_metrics in TASK_METRICS.items():
        task_data = results.get(task, {})
        for metric_key, display_name in task_metrics:
            value = task_data.get(metric_key)
            # lm-eval sometimes nests metrics with a comma-separated filter suffix
            if value is None:
                for key, val in task_data.items():
                    if key.startswith(metric_key) and isinstance(val, (int, float)):
                        value = val
                        break
            metrics[display_name] = value

    return metrics


def extract_perf_metrics(perf_file: Path) -> dict[str, float | None]:
    """Extract performance metrics from a perf_results.json file."""
    data = json.loads(perf_file.read_text())
    metrics = {}
    for metric_key, display_name in PERF_METRICS:
        metrics[display_name] = data.get(metric_key)
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


def main() -> None:
    if not RESULTS_DIR.is_dir():
        print(f"Error: Results directory '{RESULTS_DIR}' not found.")
        print("Run the benchmarks first: bash run_benchmarks.sh")
        sys.exit(1)

    # Collect metrics for each config
    all_quality: dict[str, dict[str, float | None]] = {}
    all_perf: dict[str, dict[str, float | None]] = {}
    found_quality = False
    found_perf = False

    for config_name, config_dir_name in CONFIGS:
        config_dir = RESULTS_DIR / config_dir_name
        results_file = find_results_file(config_dir)
        if results_file:
            all_quality[config_name] = extract_metrics(results_file)
            found_quality = True
        else:
            all_quality[config_name] = {}

        perf_file = find_perf_results_file(config_dir)
        if perf_file:
            all_perf[config_name] = extract_perf_metrics(perf_file)
            found_perf = True
        else:
            all_perf[config_name] = {}

    if not found_quality and not found_perf:
        print("No results found. Run the benchmarks first: bash run_benchmarks.sh")
        sys.exit(1)

    config_names = [name for name, _ in CONFIGS]

    # Quality metrics table
    if found_quality:
        quality_metric_names = []
        for _task, task_metrics in TASK_METRICS.items():
            for _metric_key, display_name in task_metrics:
                quality_metric_names.append(display_name)

        print_table(
            "Llama-3.2-3B Quality Metrics",
            quality_metric_names,
            all_quality,
            config_names,
        )

    # Performance metrics table
    if found_perf:
        perf_metric_names = [display_name for _, display_name in PERF_METRICS]

        print_table(
            "Llama-3.2-3B Performance Metrics",
            perf_metric_names,
            all_perf,
            config_names,
            fmt=".1f",
        )


if __name__ == "__main__":
    main()
