#!/usr/bin/env bash
set -euo pipefail

MODEL="meta-llama/Llama-3.2-3B"
TASKS="ifeval,hellaswag"
BATCH_SIZE="1"

# --- Argument parsing ---
PLATFORM=""
SKIP_PERF=false

while [ "$#" -gt 0 ]; do
  case "$1" in
    --skip-perf)
      SKIP_PERF=true
      shift
      ;;
    cuda|mps)
      PLATFORM="$1"
      shift
      ;;
    --model)
      if [ "$#" -lt 2 ]; then
        echo "Error: --model requires a value."
        echo "Usage: bash run_benchmarks.sh [cuda|mps] [--skip-perf] [--model <model>] [--tasks <tasks>]"
        exit 1
      fi
      MODEL="$2"
      shift 2
      ;;
    --model=*)
      MODEL="${1#*=}"
      shift
      ;;
    --tasks)
      if [ "$#" -lt 2 ]; then
        echo "Error: --tasks requires a value."
        echo "Usage: bash run_benchmarks.sh [cuda|mps] [--skip-perf] [--model <model>] [--tasks <tasks>]"
        exit 1
      fi
      TASKS="$2"
      shift 2
      ;;
    --tasks=*)
      TASKS="${1#*=}"
      shift
      ;;
    --batch-size)
      if [ "$#" -lt 2 ]; then
        echo "Error: --batch-size requires a value."
        echo "Usage: bash run_benchmarks.sh [cuda|mps] [--skip-perf] [--model <model>] [--tasks <tasks>] [--batch-size <size>]"
        exit 1
      fi
      BATCH_SIZE="$2"
      shift 2
      ;;
    --batch-size=*)
      BATCH_SIZE="${1#*=}"
      shift
      ;;
    *)
      echo "Error: Unknown argument '$1'."
      echo "Usage: bash run_benchmarks.sh [cuda|mps] [--skip-perf] [--model <model>] [--tasks <tasks>] [--batch-size <size>]"
      exit 1
      ;;
  esac
done

# --- Platform detection / selection ---
if [ -z "$PLATFORM" ]; then
  echo "No platform specified, auto-detecting..."
  if uv run python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    PLATFORM="cuda"
  elif uv run python -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
    PLATFORM="mps"
  else
    echo "Error: No supported GPU detected (checked CUDA and MPS)."
    echo "Usage: bash run_benchmarks.sh [cuda|mps] [--skip-perf] [--model <model>] [--tasks <tasks>] [--batch-size <size>]"
    exit 1
  fi
  echo "Detected platform: $PLATFORM"
fi

if [ "$PLATFORM" != "cuda" ] && [ "$PLATFORM" != "mps" ]; then
  echo "Error: Invalid platform '$PLATFORM'. Must be 'cuda' or 'mps'."
  echo "Usage: bash run_benchmarks.sh [cuda|mps] [--skip-perf] [--model <model>] [--tasks <tasks>] [--batch-size <size>]"
  exit 1
fi

# --- Validate required packages ---
if [ "$PLATFORM" = "cuda" ]; then
  if ! uv run python -c "import bitsandbytes" 2>/dev/null; then
    echo "Error: bitsandbytes is not installed. Install it with:"
    echo "  uv sync --extra cuda"
    exit 1
  fi
elif [ "$PLATFORM" = "mps" ]; then
  if ! uv run python -c "import optimum.quanto" 2>/dev/null; then
    echo "Error: optimum-quanto is not installed. Install it with:"
    echo "  uv sync --extra mps"
    exit 1
  fi
fi

echo "============================================"
echo "  Model Quantization Benchmarks"
echo "============================================"
echo ""
echo "Model:     $MODEL"
echo "Tasks:     $TASKS"
echo "Batch:     $BATCH_SIZE"
echo "Platform:  $PLATFORM"
if [ "$SKIP_PERF" = true ]; then
  echo "Perf:      SKIPPED (--skip-perf)"
else
  echo "Perf:      enabled"
fi
echo ""
echo "REMINDER: If your selected model is gated, accept its license on"
echo "huggingface.co and log in via: uv run huggingface-cli login"
echo ""

# --- FP16/BF16 (baseline) ---
echo "============================================"
if [ "$PLATFORM" = "mps" ]; then
  echo "  [1/3] Running FP16 (baseline)"
else
  echo "  [1/3] Running BF16 (baseline)"
fi
echo "============================================"
if [ "$PLATFORM" = "mps" ]; then
  # MPS does not support bfloat16; use float16 instead
  uv run lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL,dtype=float16" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --output_path results/bf16 \
    --log_samples \
    --device mps
else
  uv run lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL,dtype=bfloat16" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --output_path results/bf16 \
    --log_samples
fi

if [ "$SKIP_PERF" = false ]; then
  echo ""
  echo "  Running BF16 performance benchmark..."
  uv run python run_perf_benchmark.py \
    --model "$MODEL" \
    --quant bf16 \
    --device "$PLATFORM" \
    --output-path results/bf16
fi

echo ""

# --- INT8 ---
echo "============================================"
echo "  [2/3] Running INT8 (8-bit quantization)"
echo "============================================"
if [ "$PLATFORM" = "mps" ]; then
  uv run python run_benchmark_quanto.py \
    --model "$MODEL" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --output_path results/int8 \
    --weights int8 \
    --device mps
else
  uv run lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL,load_in_8bit=True" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --output_path results/int8 \
    --log_samples
fi

if [ "$SKIP_PERF" = false ]; then
  echo ""
  echo "  Running INT8 performance benchmark..."
  uv run python run_perf_benchmark.py \
    --model "$MODEL" \
    --quant int8 \
    --device "$PLATFORM" \
    --output-path results/int8
fi

echo ""

# --- INT4 ---
echo "============================================"
echo "  [3/3] Running INT4 (4-bit quantization)"
echo "============================================"
if [ "$PLATFORM" = "mps" ]; then
  uv run python run_benchmark_quanto.py \
    --model "$MODEL" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --output_path results/int4 \
    --weights int4 \
    --device mps
else
  uv run lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL,load_in_4bit=True" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --output_path results/int4 \
    --log_samples
fi

if [ "$SKIP_PERF" = false ]; then
  echo ""
  echo "  Running INT4 performance benchmark..."
  uv run python run_perf_benchmark.py \
    --model "$MODEL" \
    --quant int4 \
    --device "$PLATFORM" \
    --output-path results/int4
fi

echo ""
echo "============================================"
echo "  All benchmarks complete!"
echo "  Compare results: uv run python compare_results.py"
echo "============================================"
