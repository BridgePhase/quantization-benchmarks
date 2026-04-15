# Results Analysis

This analysis uses only the remaining post-fix benchmark artifacts in `results/`.
Older runs generated before the `2026-03-26` fix for the INT4 memory accounting/loading bug were removed, so the conclusions below reflect the corrected quantization behavior.

## Runs Included

- `microsoft/Phi-4-mini-instruct` (`2026-03-26-23:25:55-results.json`)
- `meta-llama/Llama-3.2-3B` (`2026-03-27-15:41:19-results.json`)
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (`2026-03-28-17:14:58-results.json`)
- `google/gemma-3-4b-pt` (`2026-03-29-15:39:55-results.json`)

All four runs were executed on CUDA and include both `ifeval` and `mmlu`.

## Benchmarks Used

| Benchmark | Full Name | What It Measures | Format | Main Metrics In This Repo | Why It Matters |
|---|---|---|---|---|---|
| `mmlu` | Massive Multitask Language Understanding | General knowledge and reasoning across many subjects | Multiple-choice QA | `acc` / `acc,none` | Shows whether quantization affects broad knowledge and answer selection |
| `ifeval` | Instruction Following Evaluation | How well the model follows prompt instructions and output constraints | Instruction-following generation | `prompt_level_strict_acc`, `inst_level_strict_acc`, `prompt_level_loose_acc`, `inst_level_loose_acc` | Shows whether quantization affects instruction-following reliability |

| IFEval Metric | Meaning |
|---|---|
| `prompt_level_strict_acc` | Whether the full prompt was followed under strict checking |
| `inst_level_strict_acc` | Whether individual instructions were followed under strict checking |
| `prompt_level_loose_acc` | Whether the full prompt was followed under more permissive checking |
| `inst_level_loose_acc` | Whether individual instructions were followed under more permissive checking |

## Executive Summary

- `INT8` is the best quality-preserving compromise.
- `INT4` provides the largest VRAM savings and, after the bug fix, now clearly uses less memory than `INT8`.
- In these benchmarks, quantization reduced memory usage but did not improve throughput.
- `BF16` remained the fastest mode across all retained runs.

## Quality Impact

### INT8 vs BF16

`INT8` stayed very close to `BF16` overall.

- `MMLU`: average delta `-0.0026`
- `IFEval prompt strict`: average delta `+0.0046`
- `IFEval inst strict`: average delta `+0.0006`
- `IFEval prompt loose`: average delta `+0.0042`
- `IFEval inst loose`: average delta `+0.0003`

Interpretation:

- `MMLU` was effectively unchanged.
- `IFEval` moved slightly in either direction depending on model, but the average change was near zero.
- Small gains for `INT8` on some `IFEval` metrics should be treated as normal benchmark variation rather than evidence that quantization improved the model.

### INT4 vs BF16

`INT4` showed a clearer quality tradeoff.

- `MMLU`: average delta `-0.0283`
- `IFEval prompt strict`: average delta `-0.0300`
- `IFEval inst strict`: average delta `-0.0279`
- `IFEval prompt loose`: average delta `-0.0337`
- `IFEval inst loose`: average delta `-0.0297`

Interpretation:

- `INT4` consistently degraded quality more than `INT8`.
- The strongest quality regressions appeared on larger/stronger models such as `Phi-4-mini-instruct` and `Llama-3.2-3B`.
- Tiny models were less consistent, but the aggregate trend still favored `BF16` and `INT8` over `INT4`.

## Memory Impact

This is the clearest benefit of quantization in the retained runs.

### INT8 vs BF16

- model VRAM reduction: `41.5%` to `43.8%`
- average reduction: `42.6%`

### INT4 vs BF16

- model VRAM reduction: `60.3%` to `63.1%`
- average reduction: `61.6%`

Interpretation:

- After removing the buggy pre-fix runs, `INT4` now clearly and consistently uses less memory than `INT8`.
- The post-fix results now match the expected ordering: `BF16` > `INT8` > `INT4` for VRAM usage.

## Performance Impact

Quantization improved memory efficiency, but not speed.

### INT8 vs BF16

- average tokens/sec delta: `-70.2%`
- average latency delta: `+246.1%`

### INT4 vs BF16

- average tokens/sec delta: `-41.9%`
- average latency delta: `+71.3%`

Interpretation:

- `BF16` was the fastest option in every retained run.
- `INT8` was the slowest mode.
- `INT4` was still slower than `BF16`, but it was consistently less slow than `INT8`.

This does not mean 4-bit inference is inherently faster than 8-bit inference in general. It means that, in this specific stack and benchmark setup, the `INT8` path appears to pay more end-to-end overhead than the `INT4` path. Likely contributors include kernel differences, dequantization overhead, memory-bandwidth effects, and the fact that these measurements were taken at batch size `1` during autoregressive generation, where per-token overhead matters a lot.

## Per-Model Notes

### `microsoft/Phi-4-mini-instruct`

- `INT8` preserved `MMLU` closely and was slightly better on some `IFEval` metrics.
- `INT4` introduced the largest quality regressions in the dataset, especially on `IFEval`.
- Memory behavior followed expectations after the fix: `INT4` < `INT8` < `BF16`.

### `meta-llama/Llama-3.2-3B`

- `INT8` was nearly lossless on `MMLU` but regressed `IFEval` somewhat.
- `INT4` degraded both `MMLU` and `IFEval` more noticeably.
- Performance followed the same pattern seen elsewhere: `BF16` fastest, `INT8` slowest.

### `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

- Quality changes were small and somewhat noisy across quantization levels.
- Memory still improved substantially with quantization.
- Despite the small model size, the quantized paths were still slower than `BF16`.

### `google/gemma-3-4b-pt`

- `INT8` slightly reduced `MMLU` but improved `IFEval` metrics.
- `INT4` hurt `MMLU` more clearly while leaving `IFEval` closer to baseline.
- The memory reductions were strong and aligned with the other post-fix runs.

## Conclusions

If the priority is preserving quality while cutting VRAM, `INT8` is the best default choice in these results.

If the priority is maximum memory reduction, `INT4` is now the clear winner after the bug fix, but that benefit comes with a larger quality cost.

If the priority is raw generation speed, `BF16` remains the best option in this benchmark suite.

Overall:

- Best quality / speed baseline: `BF16`
- Best memory / quality tradeoff: `INT8`
- Best memory savings: `INT4`
