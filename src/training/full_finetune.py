from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..utils.logger import get_logger, setup_logger
from .trainer_base import BaseFinetuningTrainer


logger = get_logger("training.full")


class FullFineTuner(BaseFinetuningTrainer):
    """
    Performs full model fine-tuning without adapters.
    """

    def _load_model(self):
        model = super()._load_model()
        logger.info("Loaded full-precision model with %s trainable parameters.", model.num_parameters())
        return model


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full fine-tuning for causal LMs.")
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
    tuner = FullFineTuner.from_config_files(
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
