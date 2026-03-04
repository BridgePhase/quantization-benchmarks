# Model Quantization Comparison

Compare any HuggingFace model at full precision (BF16) vs 8-bit and 4-bit quantization using [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness).

Supports both **NVIDIA CUDA GPUs** and **Apple Silicon Macs (MPS)**.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- One of the following:
  - **CUDA GPU** (Linux/Windows) — uses [BitsAndBytes](https://github.com/bitsandbytes-foundation/bitsandbytes) for quantization
  - **Apple Silicon Mac** (M1/M2/M3/M4) — runs BF16 baseline on MPS (quantization disabled)
- HuggingFace account (if using gated models, accept the license on HuggingFace first)

## Setup

Install dependencies for your platform:

```bash
# CUDA GPU
uv sync --extra cuda

# Apple Silicon Mac (MPS)
uv sync
```

Log in to HuggingFace (required for gated model access):

```bash
uv run huggingface-cli login
```

## Configuration

`run_benchmarks.py` has defaults near the top of the script:

```bash
MODEL="meta-llama/Llama-3.2-3B"   # Any HuggingFace model ID
TASKS="ifeval,hellaswag"           # Comma-separated lm-eval task names
BATCH_SIZE="1"                     # Shared eval batch size (CUDA + MPS)
```

You can override either value at runtime via CLI flags:

- `--model <huggingface-model-id>`
- `--tasks <comma-separated-task-list>`
- `--batch-size <size>` (defaults to `1`)
- `--limit <n>` (optional; no limit by default)

## Usage

### Run benchmarks

`run_benchmarks.sh` is a thin wrapper around `run_benchmarks.py`.

Common benchmark CLI defaults/flags are shared internally via `benchmark_common.py`.
For compatibility, helper scripts accept both hyphen and underscore variants for shared flags (for example `--batch-size` / `--batch_size`, `--output-path` / `--output_path`).

```bash
# Auto-detect platform
bash run_benchmarks.sh

# Or specify explicitly
bash run_benchmarks.sh cuda
bash run_benchmarks.sh mps

# Override model and/or tasks
bash run_benchmarks.sh mps --model "google/gemma-2-2b" --tasks "ifeval"
bash run_benchmarks.sh cuda --tasks "hellaswag,ifeval"

# Set an explicit batch size for apples-to-apples comparisons
bash run_benchmarks.sh mps --batch-size 1
bash run_benchmarks.sh cuda --batch-size 1

# Skip performance benchmarks (quality-only)
bash run_benchmarks.sh mps --skip-perf

# Fast smoke test using only first 10 eval samples per task
bash run_benchmarks.sh mps --skip-perf --tasks ifeval --batch-size 1 --limit 10
```

On CUDA, this runs three configurations sequentially:

1. **BF16** — full precision baseline
2. **INT8** — 8-bit quantization
3. **INT4** — 4-bit quantization

On MPS, quantization is disabled, so only **BF16** executes. INT8/INT4 remain
in run artifacts and comparison tables with `N/A` values for consistency.

Why quantization is disabled on MPS in this project:

- **No native BitsAndBytes Metal kernels**: BitsAndBytes quantization is built around highly tuned CUDA kernels. Equivalent native kernels are not available on MPS.
- **Dequantization overhead**: without native low-bit kernels, weights are often expanded to higher precision during compute, which adds memory traffic and extra compute overhead.
- **Observed memory behavior**: in practice, VRAM usage on MPS may not decrease (and can slightly increase) due to runtime/bookkeeping overhead instead of true low-bit execution.

Because of this, this repo treats MPS as a BF16 baseline path, and reserves INT8/INT4 comparisons for CUDA.

For each configuration, both a **quality benchmark** (lm-eval) and a **performance benchmark** (throughput/latency/VRAM) are run. Use `--skip-perf` to skip the performance benchmarks.

Results are saved to `results/{bf16,int8,int4}/`. Each directory contains:
- `results.json` (or lm-eval JSON variant) — quality metrics from the backend runner
- `perf_results.json` — performance metrics (VRAM, throughput, latency)

Each benchmark invocation also creates one run-level artifact:
- `results/<YYYY-mm-dd-HH:MM:SS>-results.json`

The run artifact is updated after each completed step and includes:
- run metadata (`device`, `model`, `tasks`, `batch_size`, `include_performance_metrics`)
- per-config quality payloads (`bf16`, `int8`, `int4`)
- per-config performance payloads when performance benchmarks are enabled

### What `--limit` does

`--limit` passes through to lm-eval's sample limiter for quality evaluation. It caps how many examples are evaluated per task.

- `--limit` is **not enabled by default** (full dataset/task split is used)
- it speeds up quality runs significantly for debugging/smoke tests
- it does **not** change the performance benchmark phase (`run_perf_benchmark.py`)
- it should generally **not** be used for final quality comparisons, because smaller sample counts are noisier and less representative

### Run performance benchmarks standalone

```bash
uv run python run_perf_benchmark.py \
  --model meta-llama/Llama-3.2-3B \
  --quant bf16 \
  --device mps \
  --output-path results/bf16
```

Options: `--max-new-tokens` (default 128), `--warmup` (default 1), `--iterations` (default 5).

### Compare results

```bash
uv run python compare_results.py
```

Prints a formatted table comparing metrics across all configurations.

## Metrics

### Quality metrics

The default task configuration evaluates:

- **HellaSwag**: `acc_norm` (normalized accuracy)
- **IFEval**: `prompt_level_strict_acc`, `inst_level_strict_acc`, `prompt_level_loose_acc`, `inst_level_loose_acc`

`compare_results.py` discovers task metrics dynamically from the result payloads.

### Performance metrics

The performance benchmark (`run_perf_benchmark.py`) measures:

- **VRAM Model (MB)** — GPU memory allocated after model load
- **VRAM Peak (MB)** — Peak GPU memory during inference (CUDA only; shows N/A on MPS)
- **Avg / Median Tokens/sec** — Inference throughput across 5 standardized prompts
- **Avg / Median Latency (s)** — Generation time per prompt

Methodology:
- Uses greedy decoding (`do_sample=False`) for reproducibility
- Runs 1 warmup iteration before measurement
- Generates 128 tokens per prompt by default
- Uses `torch.cuda.synchronize()` / `torch.mps.synchronize()` for accurate timing

## Notes

- Each benchmark run takes 30-60+ minutes depending on hardware and model size.
- CUDA supports BF16/INT8/INT4. MPS supports BF16 only.
- BitsAndBytes low-bit quantization is CUDA-optimized; this project does not run quantized paths on MPS to avoid misleading performance/memory conclusions.
- Peak VRAM tracking is only available on CUDA (`torch.cuda.max_memory_allocated`). On MPS, the "VRAM Peak" column shows N/A.
