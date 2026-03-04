"""Run quantized benchmarks using optimum-quanto (for MPS / non-CUDA platforms).

lm_eval's CLI cannot pass complex Python objects like QuantoConfig through
--model_args, so this thin wrapper uses the Python API directly.
"""

import argparse
from pathlib import Path

import lm_eval
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig

from benchmark_common import (
    add_batch_size_arg,
    add_device_arg,
    add_model_arg,
    add_output_path_arg,
    add_tasks_arg,
    model_output_dir,
    now_timestamp,
    parse_tasks,
    write_json,
)


WEIGHTS_MAP = {
    "int8": "int8",
    "int4": "int4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lm-eval benchmarks with optimum-quanto quantization."
    )
    add_model_arg(parser, required=True)
    add_tasks_arg(parser, required=True)
    add_batch_size_arg(parser, default="1")
    add_output_path_arg(parser, required=True)
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        choices=list(WEIGHTS_MAP.keys()),
        help="Quantization weight type (int8 or int4)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional lm-eval sample limit for faster smoke tests.",
    )
    add_device_arg(parser, default="mps")
    return parser.parse_args()


def run_quanto_quality(
    *,
    model_id: str,
    tasks: str,
    batch_size: str,
    output_path: str | Path,
    weights: str,
    device: str = "mps",
    limit: int | None = None,
) -> tuple[Path, dict]:
    quantization_config = QuantoConfig(weights=WEIGHTS_MAP[weights])

    print(f"Loading model {model_id} with quanto {weights} quantization...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map=device,
    )

    lm = HFLM(pretrained=model, tokenizer=tokenizer, device=device)

    task_list = parse_tasks(tasks)

    print(f"Running evaluation on tasks: {task_list}")
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=task_list,
        batch_size=batch_size,
        limit=limit,
        log_samples=True,
    )

    # Save results in the format compare_results.py expects
    output_dir = model_output_dir(output_path, model_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "results.json"
    timestamp = now_timestamp()

    # lm_eval.simple_evaluate returns a dict with "results", "samples", etc.
    # We need the "results" key at top level of the JSON for compare_results.py.
    serializable = {
        "results": results.get("results", {}),
        "config": {
            "model": model_id,
            "tasks": task_list,
            "batch_size": batch_size,
            "limit": limit,
            "quantization": f"quanto_{weights}",
            "device": device,
            "generated_at": timestamp,
        },
    }
    write_json(output_file, serializable)
    print(f"Results saved to {output_file}")
    return output_file, serializable


def main() -> None:
    args = parse_args()
    run_quanto_quality(
        model_id=args.model,
        tasks=args.tasks,
        batch_size=args.batch_size,
        output_path=args.output_path,
        weights=args.weights,
        device=args.device,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
