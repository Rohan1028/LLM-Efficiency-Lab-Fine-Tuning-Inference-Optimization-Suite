from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from ..utils.logger import get_logger, setup_logger
from ..utils.metrics import latency_quantiles


logger = get_logger("inference.latency_profiler")


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile latency distribution during generation.")
    parser.add_argument("--config", type=str, required=True, help="Inference config path.")
    parser.add_argument("--iters", type=int, default=10, help="Number of runs to profile.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    return parser.parse_args(args=args)


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_pipeline(model_name: str, dtype: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=getattr(torch, dtype) if hasattr(torch, dtype) else torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=model.dtype,
    )


def profile_latency(config: Dict[str, Any], num_iterations: int) -> Dict[str, Any]:
    prompt = config["prompt"]
    dtype = config.get("quantization", {}).get("compute_dtype", "float16")
    text_pipe = load_pipeline(config["model_name"], dtype)
    per_token_latencies: List[float] = []
    per_request_latencies: List[float] = []

    for idx in range(num_iterations):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        outputs = text_pipe(prompt, max_new_tokens=config.get("max_new_tokens", 128), do_sample=False)
        request_latency = time.perf_counter() - start
        per_request_latencies.append(request_latency)
        tokens = outputs[0]["generated_text"].split()
        if len(tokens) > 0:
            per_token_latencies.append(request_latency / len(tokens))
        logger.debug("Iteration %s latency %.4fs", idx, request_latency)

    result = {
        "model": config["model_name"],
        "prompt_length": len(prompt.split()),
        "iterations": num_iterations,
        "per_request_latency_s": per_request_latencies,
        "request_latency_summary": {
            "avg": sum(per_request_latencies) / len(per_request_latencies),
            **latency_quantiles(
                per_request_latencies, config.get("metrics", {}).get("latency_quantiles", [0.5, 0.95])
            ),
        },
        "per_token_latency_s": per_token_latencies,
    }
    if torch.cuda.is_available():
        result["gpu_memory_gb"] = torch.cuda.max_memory_allocated() / 1024**3
        torch.cuda.reset_peak_memory_stats()
    return result


def main(cli_args: argparse.Namespace | None = None) -> None:
    args = cli_args or parse_args()
    setup_logger()
    config = load_config(Path(args.config))
    report = profile_latency(config, args.iters)
    output_path = Path(args.output or config.get("output_dir", "results") + "/latency_profile.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved latency profile to %s", output_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
