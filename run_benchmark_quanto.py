"""Run quantized benchmarks using optimum-quanto (for MPS / non-CUDA platforms).

lm_eval's CLI cannot pass complex Python objects like QuantoConfig through
--model_args, so this thin wrapper uses the Python API directly.
"""

import argparse
import json
from pathlib import Path

import lm_eval
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig


WEIGHTS_MAP = {
    "int8": "int8",
    "int4": "int4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lm-eval benchmarks with optimum-quanto quantization."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g. meta-llama/Llama-3.2-3B)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        required=True,
        help="Comma-separated list of lm-eval tasks",
    )
    parser.add_argument(
        "--batch_size",
        type=str,
        default="auto",
        help="Batch size for evaluation (default: auto)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Directory to save results JSON",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        choices=list(WEIGHTS_MAP.keys()),
        help="Quantization weight type (int8 or int4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to run on (default: mps)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    quantization_config = QuantoConfig(weights=WEIGHTS_MAP[args.weights])

    print(f"Loading model {args.model} with quanto {args.weights} quantization...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        device_map=args.device,
    )

    lm = HFLM(pretrained=model, tokenizer=tokenizer, device=args.device)

    task_list = [t.strip() for t in args.tasks.split(",")]

    print(f"Running evaluation on tasks: {task_list}")
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=task_list,
        batch_size=args.batch_size,
        log_samples=True,
    )

    # Save results in the format compare_results.py expects
    output_dir = Path(args.output_path) / args.model.replace("/", "__")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "results.json"

    # lm_eval.simple_evaluate returns a dict with "results", "samples", etc.
    # We need the "results" key at top level of the JSON for compare_results.py.
    serializable = {
        "results": results.get("results", {}),
        "config": {
            "model": args.model,
            "quantization": f"quanto_{args.weights}",
            "device": args.device,
        },
    }
    output_file.write_text(json.dumps(serializable, indent=2, default=str))
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
