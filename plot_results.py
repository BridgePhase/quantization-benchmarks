"""Generate PNG charts from benchmark run artifacts."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("results")
CONFIGS = [
    ("BF16", "bf16"),
    ("INT8", "int8"),
    ("INT4", "int4"),
]
QUALITY_METRICS = [
    ("MMLU", "mmlu", "acc,none"),
    ("IFEval Prompt Strict", "ifeval", "prompt_level_strict_acc,none"),
    ("IFEval Inst Strict", "ifeval", "inst_level_strict_acc,none"),
    ("IFEval Prompt Loose", "ifeval", "prompt_level_loose_acc,none"),
    ("IFEval Inst Loose", "ifeval", "inst_level_loose_acc,none"),
]
PERF_METRICS = [
    ("Model VRAM (MB)", "model_vram_mb"),
    ("Avg Tokens/sec", "avg_tokens_per_sec"),
    ("Avg Latency (s)", "avg_latency_s"),
]
COLORS = {
    "BF16": "#4c78a8",
    "INT8": "#f58518",
    "INT4": "#54a24b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate quality and performance charts from benchmark results.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory containing run artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "graphs",
        help="Directory for generated PNG charts.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def find_latest_completed_runs(results_dir: Path) -> list[dict]:
    latest_by_model: dict[str, tuple[Path, dict]] = {}

    for path in sorted(results_dir.glob("*-results.json")):
        data = _read_json(path)
        if data.get("status") != "completed":
            continue

        model = data.get("model")
        if not isinstance(model, str):
            continue

        previous = latest_by_model.get(model)
        if previous is None or path.stat().st_mtime > previous[0].stat().st_mtime:
            latest_by_model[model] = (path, data)

    return [item[1] for item in latest_by_model.values()]


def short_model_name(model_id: str) -> str:
    return model_id.split("/", 1)[-1]


def extract_quality_matrix(
    runs: list[dict], task: str, key: str
) -> tuple[list[str], dict[str, list[float]]]:
    labels = [short_model_name(run["model"]) for run in runs]
    series: dict[str, list[float]] = {}

    for display_name, quant in CONFIGS:
        values: list[float] = []
        for run in runs:
            payload = run["configs"][quant]["quality"]["payload"]["results"]
            values.append(payload[task][key])
        series[display_name] = values

    return labels, series


def extract_perf_matrix(
    runs: list[dict], key: str
) -> tuple[list[str], dict[str, list[float]]]:
    labels = [short_model_name(run["model"]) for run in runs]
    series: dict[str, list[float]] = {}

    for display_name, quant in CONFIGS:
        values: list[float] = []
        for run in runs:
            payload = run["configs"][quant]["performance"]["payload"]
            values.append(payload[key])
        series[display_name] = values

    return labels, series


def add_grouped_bars(
    ax: plt.Axes, labels: list[str], series: dict[str, list[float]]
) -> None:
    x = np.arange(len(labels))
    width = 0.24
    offsets = [-width, 0.0, width]

    for offset, config_name in zip(offsets, series.keys(), strict=True):
        ax.bar(
            x + offset,
            series[config_name],
            width=width,
            label=config_name,
            color=COLORS[config_name],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)


def save_quality_chart(runs: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes_flat = axes.flatten()

    for index, (title, task, key) in enumerate(QUALITY_METRICS):
        labels, series = extract_quality_matrix(runs, task, key)
        ax = axes_flat[index]
        add_grouped_bars(ax, labels, series)
        ax.set_title(title)
        ax.set_ylim(bottom=0)

    axes_flat[-1].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Quality Benchmarks by Model and Precision", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / "quality_benchmarks.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_performance_chart(runs: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, (title, key) in zip(axes, PERF_METRICS, strict=True):
        labels, series = extract_perf_matrix(runs, key)
        add_grouped_bars(ax, labels, series)
        ax.set_title(title)
        ax.set_ylim(bottom=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Performance Benchmarks by Model and Precision", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output_dir / "performance_benchmarks.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_normalized_delta_chart(runs: list[dict], output_dir: Path) -> None:
    labels = [short_model_name(run["model"]) for run in runs]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    quality_deltas: dict[str, list[float]] = {"INT8": [], "INT4": []}
    perf_vram_deltas: dict[str, list[float]] = {"INT8": [], "INT4": []}

    for run in runs:
        bf16_quality = run["configs"]["bf16"]["quality"]["payload"]["results"]
        bf16_perf = run["configs"]["bf16"]["performance"]["payload"]
        for display_name, quant in (("INT8", "int8"), ("INT4", "int4")):
            quant_quality = run["configs"][quant]["quality"]["payload"]["results"]
            quant_perf = run["configs"][quant]["performance"]["payload"]
            quality_deltas[display_name].append(
                quant_quality["mmlu"]["acc,none"] - bf16_quality["mmlu"]["acc,none"]
            )
            perf_vram_deltas[display_name].append(
                (quant_perf["model_vram_mb"] - bf16_perf["model_vram_mb"])
                / bf16_perf["model_vram_mb"]
                * 100
            )

    for ax, title, series, ylabel in [
        (axes[0], "MMLU Delta vs BF16", quality_deltas, "Absolute score delta"),
        (axes[1], "Model VRAM Delta vs BF16", perf_vram_deltas, "Percent delta"),
    ]:
        x = np.arange(len(labels))
        width = 0.32
        for offset, config_name in zip(
            (-width / 2, width / 2), series.keys(), strict=True
        ):
            ax.bar(
                x + offset,
                series[config_name],
                width=width,
                label=config_name,
                color=COLORS[config_name],
            )
        ax.axhline(0, color="#444444", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Quantized Delta vs BF16", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output_dir / "quantization_deltas.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    runs = sorted(
        find_latest_completed_runs(args.results_dir), key=lambda run: run["model"]
    )
    if not runs:
        raise SystemExit("No completed run artifacts found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_quality_chart(runs, args.output_dir)
    save_performance_chart(runs, args.output_dir)
    save_normalized_delta_chart(runs, args.output_dir)

    print(f"Generated charts in {args.output_dir}")


if __name__ == "__main__":
    main()
