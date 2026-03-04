"""Parse lm-eval results and print a comparison table across quantization configs."""

import argparse
import json
import re
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


def find_results_file(config_dir: Path) -> Path | None:
    """Find the results JSON file inside a config's output directory.

    lm-eval writes results to: <output_path>/<model_name>/<filename>.json
    We look for any JSON file containing a "results" key.
    """
    if not config_dir.is_dir():
        return None

    # Prefer the most recent timestamped quality file.
    timestamped_files = sorted(
        config_dir.rglob("*-results.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for json_file in timestamped_files:
        if json_file.name == "perf_results.json":
            continue
        try:
            data = json.loads(json_file.read_text())
            if "results" in data:
                return json_file
        except (json.JSONDecodeError, KeyError):
            continue

    # Fallback to legacy file name.
    for json_file in sorted(config_dir.rglob("results.json"), reverse=True):
        return json_file

    # Fallback: look for any JSON with a "results" key.
    for json_file in sorted(config_dir.rglob("*.json"), reverse=True):
        if json_file.name == "perf_results.json":
            continue
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


def task_metric_label(task_name: str, metric_name: str) -> str:
    return f"{task_name} {metric_name}"


def discover_quality_metrics(quality_files: list[Path]) -> list[tuple[str, str]]:
    """Discover quality metrics dynamically from lm-eval result payloads."""
    discovered: set[tuple[str, str]] = set()
    for quality_file in quality_files:
        try:
            data = json.loads(quality_file.read_text())
        except json.JSONDecodeError:
            continue
        results = data.get("results")
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


def extract_metrics(
    results_file: Path,
    metric_specs: list[tuple[str, str]],
) -> dict[str, float | None]:
    """Extract dynamic quality metrics from a results JSON file."""
    data = json.loads(results_file.read_text())
    results = data.get("results", {})
    metrics = {}

    for task, metric_name in metric_specs:
        task_data = results.get(task, {})
        display_name = task_metric_label(task, metric_name)
        if not isinstance(task_data, dict):
            metrics[display_name] = None
            continue

        value = task_data.get(metric_name)
        # lm-eval often uses filter suffixes such as ",none".
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


def extract_perf_metrics(perf_file: Path) -> dict[str, float | None]:
    """Extract performance metrics from a perf_results.json file."""
    data = json.loads(perf_file.read_text())
    metrics = {}
    for metric_key, display_name in PERF_METRICS:
        metrics[display_name] = data.get(metric_key)
    return metrics


def infer_model_label(result_file: Path) -> str | None:
    """Infer model label from a result/perf JSON file path/payload."""
    model_dir_name = result_file.parent.name
    if "__" in model_dir_name:
        return model_dir_name.replace("__", "/")

    try:
        data = json.loads(result_file.read_text())
    except json.JSONDecodeError:
        return None

    config = data.get("config")
    if isinstance(config, dict):
        model = config.get("model")
        if isinstance(model, str) and model:
            return model

        model_args = config.get("model_args")
        if isinstance(model_args, str):
            match = re.search(r"pretrained=([^,]+)", model_args)
            if match:
                return match.group(1)

    return None


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
        description="Compare benchmark quality and performance results across quant configs.",
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

    # Collect metrics for each config
    all_quality: dict[str, dict[str, float | None]] = {}
    all_perf: dict[str, dict[str, float | None]] = {}
    quality_files: dict[str, Path | None] = {}
    found_quality = False
    found_perf = False
    inferred_model_label: str | None = None

    for config_name, config_dir_name in CONFIGS:
        config_dir = RESULTS_DIR / config_dir_name
        results_file = find_results_file(config_dir)
        quality_files[config_name] = results_file
        if results_file:
            found_quality = True
            if inferred_model_label is None:
                inferred_model_label = infer_model_label(results_file)

        perf_file = find_perf_results_file(config_dir)
        if perf_file:
            all_perf[config_name] = extract_perf_metrics(perf_file)
            found_perf = True
            if inferred_model_label is None:
                inferred_model_label = infer_model_label(perf_file)
        else:
            all_perf[config_name] = {}

    if not found_quality and not found_perf:
        print("No results found. Run the benchmarks first: bash run_benchmarks.sh")
        sys.exit(1)

    config_names = [name for name, _ in CONFIGS]
    model_label = args.model or inferred_model_label or "Model"

    # Quality metrics table
    if found_quality:
        discovered_metrics = discover_quality_metrics(
            [path for path in quality_files.values() if path is not None],
        )
        if not discovered_metrics:
            print("No quality metrics found in result payloads.")
            found_quality = False

    if found_quality:
        quality_metric_names = [
            task_metric_label(task, metric_name)
            for task, metric_name in discovered_metrics
        ]

        for config_name in config_names:
            results_file = quality_files.get(config_name)
            if results_file is not None:
                all_quality[config_name] = extract_metrics(
                    results_file, discovered_metrics
                )
            else:
                all_quality[config_name] = {}

        print_table(
            f"{model_label} Quality Metrics",
            quality_metric_names,
            all_quality,
            config_names,
        )

    # Performance metrics table
    if found_perf:
        perf_metric_names = [display_name for _, display_name in PERF_METRICS]

        print_table(
            f"{model_label} Performance Metrics",
            perf_metric_names,
            all_perf,
            config_names,
            fmt=".1f",
        )


if __name__ == "__main__":
    main()
