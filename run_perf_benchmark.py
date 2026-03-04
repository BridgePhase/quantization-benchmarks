"""Measure inference performance (VRAM, throughput, latency) for a quantized model."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "The theory of general relativity describes",
    "In a large-scale distributed system, consistency can be achieved by",
    "The French Revolution began in 1789 when",
    "To implement a binary search tree in Python, you would",
    "The mitochondria is often called the powerhouse of the cell because",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure inference performance for a quantized model."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g. meta-llama/Llama-3.2-3B)",
    )
    parser.add_argument(
        "--quant",
        type=str,
        required=True,
        choices=["bf16", "int8", "int4"],
        help="Quantization level",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        choices=["cuda", "mps"],
        help="Device to run on",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output directory (perf_results.json saved under <output-path>/<model>/)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Number of tokens to generate per prompt (default: 128)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup iterations (default: 1)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of measurement iterations (default: 5)",
    )
    return parser.parse_args()


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def load_model(model_id: str, quant: str, device: str):
    kwargs = {}

    if quant == "bf16":
        if device == "mps":
            kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = torch.bfloat16
        kwargs["device_map"] = device
    elif device == "cuda":
        # BitsAndBytes quantization for CUDA
        if quant == "int8":
            kwargs["load_in_8bit"] = True
        elif quant == "int4":
            kwargs["load_in_4bit"] = True
    elif device == "mps":
        # optimum-quanto for MPS
        from transformers import QuantoConfig

        kwargs["quantization_config"] = QuantoConfig(weights=quant)
        kwargs["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def measure_vram(device: str) -> dict:
    vram = {}
    if device == "cuda":
        vram["model_vram_mb"] = round(
            torch.cuda.memory_allocated() / 1024 / 1024, 1
        )
    elif device == "mps":
        vram["model_vram_mb"] = round(
            torch.mps.current_allocated_memory() / 1024 / 1024, 1
        )
    return vram


def run_benchmark(
    model,
    tokenizer,
    device: str,
    max_new_tokens: int,
    warmup: int,
    iterations: int,
) -> dict:
    prompts = PROMPTS[:iterations]

    # Warmup
    for i in range(warmup):
        print(f"  Warmup {i + 1}/{warmup}...")
        inputs = tokenizer(prompts[0], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        synchronize(device)

    # Reset peak memory after warmup (CUDA only)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Measurement
    latencies = []
    tokens_per_sec_list = []

    for i, prompt in enumerate(prompts):
        print(f"  Iteration {i + 1}/{iterations}: {prompt[:50]}...")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        synchronize(device)
        start = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )

        synchronize(device)
        elapsed = time.perf_counter() - start

        generated_tokens = outputs.shape[1] - input_len
        tps = generated_tokens / elapsed

        latencies.append(elapsed)
        tokens_per_sec_list.append(tps)
        print(f"    {generated_tokens} tokens in {elapsed:.2f}s ({tps:.1f} tok/s)")

    # Peak VRAM (CUDA only)
    peak_vram = {}
    if device == "cuda":
        peak_vram["peak_vram_mb"] = round(
            torch.cuda.max_memory_allocated() / 1024 / 1024, 1
        )

    return {
        **peak_vram,
        "avg_tokens_per_sec": round(statistics.mean(tokens_per_sec_list), 2),
        "median_tokens_per_sec": round(statistics.median(tokens_per_sec_list), 2),
        "avg_latency_s": round(statistics.mean(latencies), 3),
        "median_latency_s": round(statistics.median(latencies), 3),
        "min_latency_s": round(min(latencies), 3),
        "max_latency_s": round(max(latencies), 3),
    }


def main() -> None:
    args = parse_args()

    print(f"Loading {args.model} ({args.quant}) on {args.device}...")
    model, tokenizer = load_model(args.model, args.quant, args.device)

    # Measure VRAM after model load
    vram_info = measure_vram(args.device)
    print(f"  Model VRAM: {vram_info.get('model_vram_mb', 'N/A')} MB")

    print(f"Running performance benchmark ({args.iterations} iterations, "
          f"{args.max_new_tokens} tokens each)...")
    perf_metrics = run_benchmark(
        model,
        tokenizer,
        args.device,
        args.max_new_tokens,
        args.warmup,
        args.iterations,
    )

    results = {
        "model": args.model,
        "quantization": args.quant,
        "device": args.device,
        "max_new_tokens": args.max_new_tokens,
        "warmup_iterations": args.warmup,
        "measurement_iterations": args.iterations,
        **vram_info,
        **perf_metrics,
    }

    # Save to <output-path>/<model_name>/perf_results.json
    output_dir = Path(args.output_path) / args.model.replace("/", "__")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "perf_results.json"
    output_file.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output_file}")

    # Print summary
    print("\n--- Performance Summary ---")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
