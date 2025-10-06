from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

from ..utils.logger import get_logger, setup_logger
from ..utils.metrics import latency_quantiles, throughput


logger = get_logger("inference.quantization")


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark quantized inference for LLMs.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to inference YAML configuration.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save metrics JSON.",
    )
    return parser.parse_args(args=args)


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_model(model_name: str, bits: int, dtype: str, backend: str):
    quant_args: Dict[str, Any] = {"trust_remote_code": False}
    if bits in {4, 8} and backend == "bitsandbytes":
        quant_args["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=bits == 4,
            load_in_8bit=bits == 8,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=getattr(torch, dtype),
        )
        quant_args["device_map"] = "auto"
    else:
        quant_args["device_map"] = "auto"
        quant_args["torch_dtype"] = getattr(torch, dtype)
    logger.info("Loading %s at %s-bit precision.", model_name, bits)
    model = AutoModelForCausalLM.from_pretrained(model_name, **quant_args)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def benchmark(config: Dict[str, Any]) -> Dict[str, Any]:
    model_name = config["model_name"]
    dtype = config.get("quantization", {}).get("compute_dtype", "bfloat16")
    backend = config.get("quantization", {}).get("backend", "bitsandbytes")
    prompt = config["prompt"]
    bits_list: List[int] = config["quantization"]["bits"]
    max_new_tokens = config.get("max_new_tokens", 128)
    results = []

    for bits in bits_list:
        model, tokenizer = load_model(model_name, bits, dtype, backend)
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=getattr(torch, dtype) if hasattr(torch, dtype) else torch.float16,
        )
        warmup = generator(prompt, max_new_tokens=32, do_sample=False)
        logger.debug("Warmup output (%s-bit): %s", bits, warmup[0]["generated_text"])

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated() / 1024**3
        else:
            start_mem = 0.0

        token_counts: List[int] = []
        latencies = []
        generated_text = ""
        for _ in range(3):
            start = time.perf_counter()
            output = generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)
            generated_text = output[0]["generated_text"]
            token_counts.append(len(tokenizer.encode(generated_text)))

        total_tokens = sum(token_counts)
        total_time = sum(latencies)
        metric_throughput = throughput(total_tokens, total_time)

        max_mem = (
            torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else start_mem
        )
        results.append(
            {
                "bits": bits,
                "avg_latency_s": sum(latencies) / len(latencies),
                "latency_quantiles": latency_quantiles(latencies, config["metrics"]["latency_quantiles"]),
                "throughput_tokens_per_s": metric_throughput,
                "gpu_memory_gb": max_mem,
                "output_sample": generated_text[-200:],
            }
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "model": model_name,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "results": results,
    }


def main(cli_args: argparse.Namespace | None = None) -> None:
    args = cli_args or parse_args()
    setup_logger()
    config = load_config(Path(args.config))
    report = benchmark(config)
    output_path = Path(args.output or config.get("output_dir", "results") + "/quantization_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved quantization benchmark to %s", output_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
