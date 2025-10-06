from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from ..utils.logger import get_logger, setup_logger
from .trainer_base import BaseFinetuningTrainer


logger = get_logger("training.qlora")


class QLoRAFineTuner(BaseFinetuningTrainer):
    """
    QLoRA fine-tuning that loads models in 4bit precision and applies LoRA adapters.
    """

    def _load_model(self) -> AutoModelForCausalLM:
        load_in_4bit = self.model_config.get("load_in_4bit", True)
        logger_extra: Dict[str, Any] = {
            "load_in_4bit": load_in_4bit,
            "bnb4bit_compute_dtype": self.model_config.get("dtype", "bfloat16"),
        }
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type=self.model_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=self.model_config.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=getattr(torch, self.model_config.get("dtype", "bfloat16")),
        )
        model_kwargs = {
            "device_map": "auto",
            "quantization_config": bnb_config,
            "trust_remote_code": self.model_config.get("trust_remote_code", False),
        }
        model_name = self.model_config["model_name"]
        logger.info("Loading QLoRA base model %s with %s", model_name, logger_extra)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        lora_params: Dict[str, Any] = self.model_config.get("peft", {})
        lora_config = LoraConfig(
            r=lora_params.get("r", 8),
            lora_alpha=lora_params.get("lora_alpha", 16),
            lora_dropout=lora_params.get("lora_dropout", 0.05),
            target_modules=lora_params.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for LLMs.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML training configuration.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to YAML model configuration.",
    )
    return parser.parse_args(args=args)


def main(cli_args: argparse.Namespace | None = None) -> None:
    args = cli_args or parse_args()
    setup_logger()
    tuner = QLoRAFineTuner.from_config_files(
        model_config_path=Path(args.model_config),
        training_config_path=Path(args.config),
    )
    torch.manual_seed(tuner.training_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tuner.training_config.seed)
    tuner.prepare()
    metrics = tuner.run()
    for key, value in metrics.items():
        if isinstance(value, (float, int)):
            value = round(float(value), 4)
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
