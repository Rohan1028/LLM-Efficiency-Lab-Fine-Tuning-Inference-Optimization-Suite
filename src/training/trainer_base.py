from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from ..utils.logger import get_logger


logger = get_logger("training.base")


PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""


@dataclass
class DatasetConfig:
    path: str
    split: str = "train"
    text_field: str = "output"
    instruction_field: str = "instruction"
    input_field: str = "input"
    max_samples: Optional[int] = None


@dataclass
class TrainingConfig:
    seed: int = 42
    output_dir: str = "outputs/default"
    logging_dir: str = "logs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 0.3
    bf16: bool = True
    gradient_checkpointing: bool = True
    evaluation_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    logging_steps: int = 10
    push_to_hub: bool = False
    report_to: list[str] = field(default_factory=lambda: ["tensorboard"])
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    validation: Optional[DatasetConfig] = None
    resume_from_checkpoint: Optional[str] = None


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class BaseFinetuningTrainer:
    """
    Base class that encapsulates shared logic across fine-tuning strategies.
    """

    def __init__(self, model_config: Dict[str, Any], training_config: TrainingConfig) -> None:
        self.model_config = model_config
        self.training_config = training_config
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.train_dataset: Optional[Dataset] = None
        self.eval_dataset: Optional[Dataset] = None
        self.trainer: Optional[Trainer] = None

    @classmethod
    def from_config_files(cls, model_config_path: Path, training_config_path: Path):
        model_config = load_yaml(model_config_path)
        training_dict = load_yaml(training_config_path)
        logger.info("Loaded model config from %s", model_config_path)
        logger.info("Loaded training config from %s", training_config_path)

        dataset_cfg = DatasetConfig(**training_dict.pop("dataset"))
        validation_cfg = (
            DatasetConfig(**training_dict.pop("validation"))
            if training_dict.get("validation")
            else None
        )
        training_conf = TrainingConfig(dataset=dataset_cfg, validation=validation_cfg, **training_dict)
        return cls(model_config=model_config, training_config=training_conf)

    def prepare(self) -> None:
        logger.info("Preparing tokenizer and datasets.")
        self.tokenizer = self._load_tokenizer()
        self.train_dataset, self.eval_dataset = self._load_datasets()
        logger.info("Preparing model for fine-tuning.")
        self.model = self._load_model()
        if self.training_config.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        self.trainer = self._create_trainer()

    def _load_tokenizer(self) -> AutoTokenizer:
        tokenizer_name = self.model_config.get("tokenizer_name") or self.model_config["model_name"]
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    def _load_datasets(self) -> tuple[Dataset, Optional[Dataset]]:
        dataset_cfg = self.training_config.dataset
        data_files = dataset_cfg.path if dataset_cfg.path.endswith(".json") else None
        if data_files:
            dataset = load_dataset("json", data_files=dataset_cfg.path)
        else:
            dataset = load_dataset(dataset_cfg.path)

        def _select_split(ds: DatasetDict | Dataset, cfg: DatasetConfig) -> Dataset:
            if isinstance(ds, DatasetDict):
                split = cfg.split if cfg.split in ds else "train"
                selected = ds[split]
            else:
                selected = ds
            if cfg.max_samples:
                selected = selected.select(range(min(len(selected), cfg.max_samples)))
            return selected

        train_dataset = _select_split(dataset, dataset_cfg)
        eval_dataset = None
        if self.training_config.validation:
            validation_cfg = self.training_config.validation
            eval_dataset = _select_split(dataset, validation_cfg)

        def format_example(example: Dict[str, Any]) -> Dict[str, str]:
            instruction = example.get(dataset_cfg.instruction_field, "")
            input_text = example.get(dataset_cfg.input_field, "")
            output_text = example.get(dataset_cfg.text_field, "")
            example["text"] = PROMPT_TEMPLATE.format(
                instruction=instruction.strip(),
                input=input_text.strip(),
                output=output_text.strip(),
            )
            return example

        train_dataset = train_dataset.map(format_example)
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(format_example)
        return train_dataset, eval_dataset

    def _load_model(self) -> AutoModelForCausalLM:
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": getattr(torch, self.model_config.get("dtype", "float16")),
            "trust_remote_code": self.model_config.get("trust_remote_code", False),
        }
        logger.info("Loading base model %s", self.model_config["model_name"])
        model = AutoModelForCausalLM.from_pretrained(self.model_config["model_name"], **model_kwargs)
        return model

    def _create_trainer(self) -> Trainer:
        assert self.tokenizer is not None
        assert self.model is not None
        assert self.train_dataset is not None

        training_args = TrainingArguments(
            output_dir=self.training_config.output_dir,
            logging_dir=self.training_config.logging_dir,
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            warmup_ratio=self.training_config.warmup_ratio,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            max_grad_norm=self.training_config.max_grad_norm,
            bf16=self.training_config.bf16,
            gradient_checkpointing=self.training_config.gradient_checkpointing,
            evaluation_strategy=self.training_config.evaluation_strategy,
            eval_steps=self.training_config.eval_steps,
            save_strategy=self.training_config.save_strategy,
            save_steps=self.training_config.save_steps,
            save_total_limit=self.training_config.save_total_limit,
            logging_steps=self.training_config.logging_steps,
            report_to=self.training_config.report_to,
            push_to_hub=self.training_config.push_to_hub,
            seed=self.training_config.seed,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False, pad_to_multiple_of=8
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        return trainer

    def add_callback(self, callback: TrainerCallback) -> None:
        if not self.trainer:
            raise RuntimeError("Trainer not initialized. Call prepare() first.")
        self.trainer.add_callback(callback)

    def run(self) -> Dict[str, Any]:
        if not self.trainer:
            raise RuntimeError("Trainer not initialized. Call prepare() first.")
        logger.info("Starting fine-tuning.")
        train_result = self.trainer.train(
            resume_from_checkpoint=self.training_config.resume_from_checkpoint
        )
        metrics = train_result.metrics
        self.trainer.save_model()
        if self.eval_dataset is not None:
            eval_metrics = self.trainer.evaluate()
            metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})

        metrics_path = Path(self.training_config.output_dir) / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        logger.info("Training completed. Metrics saved to %s", metrics_path)
        return metrics
