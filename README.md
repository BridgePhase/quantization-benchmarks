# Model Quantization Comparison

Compare any HuggingFace model at full precision (BF16) vs 8-bit and 4-bit quantization using [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness).

Supports both **NVIDIA CUDA GPUs** (via BitsAndBytes) and **Apple Silicon Macs** (via optimum-quanto).

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- One of the following:
  - **CUDA GPU** (Linux/Windows) — uses [BitsAndBytes](https://github.com/bitsandbytes-foundation/bitsandbytes) for quantization
  - **Apple Silicon Mac** (M1/M2/M3/M4) — uses [optimum-quanto](https://github.com/huggingface/optimum-quanto) for quantization
- HuggingFace account (if using gated models, accept the license on HuggingFace first)

## Setup

Install dependencies for your platform:

```bash
# CUDA GPU
uv sync --extra cuda

# Apple Silicon Mac (MPS)
uv sync --extra mps
```

Log in to HuggingFace (required for gated model access):

```bash
uv run huggingface-cli login
```

## Configuration

`run_benchmarks.sh` has defaults at the top of the script:

```bash
MODEL="meta-llama/Llama-3.2-3B"   # Any HuggingFace model ID
TASKS="ifeval,hellaswag"           # Comma-separated lm-eval task names
```

You can override either value at runtime via CLI flags:

- `--model <huggingface-model-id>`
- `--tasks <comma-separated-task-list>`

## Usage

### Run benchmarks

```bash
# Auto-detect platform
bash run_benchmarks.sh

# Or specify explicitly
bash run_benchmarks.sh cuda
bash run_benchmarks.sh mps

# Override model and/or tasks
bash run_benchmarks.sh mps --model "google/gemma-2-2b" --tasks "ifeval"
bash run_benchmarks.sh cuda --tasks "hellaswag,ifeval"

# Skip performance benchmarks (quality-only)
bash run_benchmarks.sh mps --skip-perf
```

This runs three configurations sequentially:

1. **BF16** — full precision baseline
2. **INT8** — 8-bit quantization
3. **INT4** — 4-bit quantization

For each configuration, both a **quality benchmark** (lm-eval) and a **performance benchmark** (throughput/latency/VRAM) are run. Use `--skip-perf` to skip the performance benchmarks.

Results are saved to `results/{bf16,int8,int4}/`. Each directory contains:
- `results.json` — quality metrics from lm-eval
- `perf_results.json` — performance metrics (VRAM, throughput, latency)

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

If you use different tasks (via script defaults or `--tasks`), update the `TASK_METRICS` dict in `compare_results.py` to match.

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
- CUDA uses BitsAndBytes for quantization; MPS uses optimum-quanto. These are different quantization backends, so results are **not directly comparable across platforms** — only compare configurations within the same platform.
- On MPS, quantized runs (INT8/INT4) use a Python wrapper (`run_benchmark_quanto.py`) instead of the `lm_eval` CLI, because `lm_eval`'s CLI cannot pass the `QuantoConfig` object required by optimum-quanto.
- Peak VRAM tracking is only available on CUDA (`torch.cuda.max_memory_allocated`). On MPS, the "VRAM Peak" column shows N/A.
