from __future__ import annotations

import argparse
from .analysis import cost_vs_accuracy
from .inference import batching_optimizer, latency_profiler, quantization_benchmark
from .training import full_finetune, lora_finetune, qlora_finetune


MODES = {
    "lora": (lora_finetune.main, lora_finetune.parse_args),
    "qlora": (qlora_finetune.main, qlora_finetune.parse_args),
    "full": (full_finetune.main, full_finetune.parse_args),
    "quantization": (quantization_benchmark.main, quantization_benchmark.parse_args),
    "batching": (batching_optimizer.main, batching_optimizer.parse_args),
    "latency": (latency_profiler.main, latency_profiler.parse_args),
    "analysis": (cost_vs_accuracy.main, cost_vs_accuracy.parse_args),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM Efficiency Lab entrypoint.")
    parser.add_argument(
        "mode",
        choices=MODES.keys(),
        help="Select which experiment or utility to run.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file for the selected mode.",
    )
    parser.add_argument(
        "--additional",
        nargs=argparse.REMAINDER,
        help="Additional CLI arguments passed to the underlying script.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    handler, parser = MODES[args.mode]
    extra: list[str] = []
    if args.config:
        extra.extend(["--config", args.config])
    if args.additional:
        extra.extend(args.additional)
    cli_args = parser(extra) if extra else None
    handler(cli_args)


if __name__ == "__main__":
    main()
