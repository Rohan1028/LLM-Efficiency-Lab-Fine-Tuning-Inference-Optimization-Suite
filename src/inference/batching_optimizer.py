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
from ..utils.metrics import latency_quantiles, throughput


logger = get_logger("inference.batching")


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate batching strategies for LLM inference.")
    parser.add_argument("--config", type=str, required=True, help="Path to inference YAML config.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    return parser.parse_args(args=args)


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_text_pipeline(model_name: str, dtype: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=getattr(torch, dtype) if hasattr(torch, dtype) else torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    text_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=model.dtype,
    )
    return text_pipe, tokenizer


def evaluate_batches(config: Dict[str, Any]) -> Dict[str, Any]:
    prompt = config["prompt"]
    batch_sizes: List[int] = config["batch_sizes"]
    dtype = config.get("quantization", {}).get("compute_dtype", "float16")
    text_pipe, tokenizer = load_text_pipeline(config["model_name"], dtype)

    results = []
    for batch_size in batch_sizes:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        prompts = [prompt for _ in range(batch_size)]
        latencies = []
        token_counts = []
        generated_text = ""
        for _ in range(3):
            start = time.perf_counter()
            outputs = text_pipe(
                prompts, max_new_tokens=config.get("max_new_tokens", 128), do_sample=False
            )
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)
            generated_text = outputs[0]["generated_text"]
            total_tokens = sum(len(tokenizer.encode(out["generated_text"])) for out in outputs)
            token_counts.append(total_tokens)

        total_tokens = sum(token_counts)
        total_time = sum(latencies)
        metrics = {
            "batch_size": batch_size,
            "latency_quantiles": latency_quantiles(
                latencies, config.get("metrics", {}).get("latency_quantiles", [0.5, 0.95])
            ),
            "avg_latency_s": total_time / len(latencies),
            "throughput_tokens_per_s": throughput(total_tokens, total_time),
            "output_sample": generated_text[-200:],
        }
        if torch.cuda.is_available():
            metrics["gpu_memory_gb"] = torch.cuda.max_memory_allocated() / 1024**3
            torch.cuda.empty_cache()
        results.append(metrics)
        logger.info("Batch size %s metrics: %s", batch_size, metrics)

    return {
        "model": config["model_name"],
        "prompt": prompt,
        "results": results,
    }


def main(cli_args: argparse.Namespace | None = None) -> None:
    args = cli_args or parse_args()
    setup_logger()
    config = load_config(Path(args.config))
    report = evaluate_batches(config)
    output_path = Path(args.output or config.get("output_dir", "results") + "/batching_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved batching metrics to %s", output_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
