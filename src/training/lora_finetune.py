from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from ..utils.logger import setup_logger
from .trainer_base import BaseFinetuningTrainer


class LoRAFineTuner(BaseFinetuningTrainer):
    """
    Fine-tunes causal LMs using the PEFT LoRA strategy.
    """

    def _load_model(self):
        model = super()._load_model()
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
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for LLMs.")
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
    tuner = LoRAFineTuner.from_config_files(
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
