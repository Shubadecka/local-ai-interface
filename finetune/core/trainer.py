"""Training orchestration with Unsloth + TRL."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Callable

from finetune.config.schema import FinetuneConfig
from finetune.utils.logging import get_logger
from finetune.utils.paths import ensure_dir
from finetune.utils.data_formatting_funcs import FORMATTING_FUNCS

logger = get_logger("trainer")


class TrainingError(Exception):
    """Error during training."""

    pass


def run_training(
    config: FinetuneConfig,
    resume_from: Optional[Path] = None,
) -> Path:
    """Run the finetuning training loop.

    Args:
        config: Finetuning configuration
        resume_from: Optional checkpoint path to resume from

    Returns:
        Path to the final checkpoint directory

    Raises:
        TrainingError: If training fails
    """
    # Lazy imports to avoid loading heavy dependencies unnecessarily
    # Unsloth must be imported before trl/transformers/peft for optimizations
    try:
        import unsloth  # noqa: F401
        from unsloth import FastLanguageModel
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        raise TrainingError(
            f"Training dependencies not installed. "
            f"Install with: uv pip install llm-finetune[train]\n"
            f"Missing: {e}"
        ) from e

    output_dir = config.get_output_path()
    ensure_dir(output_dir)

    # Save config copy for reproducibility
    _save_run_config(config, output_dir)
    _save_run_metadata(config, output_dir, resume_from)

    logger.info(f"Loading model: {config.model.name}")

    # Load model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model.name,
        max_seq_length=config.model.max_seq_length,
        load_in_4bit=config.model.load_in_4bit,
        load_in_8bit=config.model.load_in_8bit,
        dtype=_get_dtype(config.model.dtype),
        trust_remote_code=config.model.trust_remote_code,
    )

    logger.info("Applying LoRA adapters")

    # Apply LoRA with Unsloth
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        target_modules=config.lora.target_modules,
        bias=config.lora.bias,
        use_gradient_checkpointing=config.lora.use_gradient_checkpointing,
        random_state=config.lora.random_state,
        use_rslora=config.lora.use_rslora,
        loftq_config=config.lora.loftq_config,
    )

    logger.info(f"Loading dataset: {config.data.dataset}")

    # Load dataset
    dataset = _load_dataset(config)

    # Prepare training arguments
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.training.num_epochs,
        per_device_train_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps,
        warmup_ratio=config.training.warmup_ratio,
        max_steps=config.training.max_steps,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        lr_scheduler_type=config.training.lr_scheduler_type,
        optim=config.training.optim,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        seed=config.training.seed,
        gradient_checkpointing=config.training.gradient_checkpointing,
        max_grad_norm=config.training.max_grad_norm,
        max_length=config.model.max_seq_length,
        packing=config.data.packing,
        dataset_text_field=config.data.text_field,
        dataloader_num_workers=config.data.dataloader_num_workers,
    )

    logger.info("Initializing trainer")

    # Get formatting function
    def get_formatting_func(name: str) -> Callable:
        if name not in FORMATTING_FUNCS:
            raise ValueError(f"Formatting function {name} not found")
        return FORMATTING_FUNCS[name]

    formatting_func = get_formatting_func(config.data.process_func)

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        formatting_func=formatting_func,
        dataset_num_proc=config.data.dataset_num_proc,
    )

    # Resume from checkpoint if specified
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")

    logger.info("Starting training")

    try:
        trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        raise

    # Save final model
    final_dir = output_dir / "final"
    logger.info(f"Saving final model to: {final_dir}")

    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Update metadata
    _update_run_metadata(output_dir, status="completed")

    logger.info("Training complete!")
    return final_dir


def _get_dtype(dtype_str: Optional[str]) -> Any:
    """Convert dtype string to torch dtype."""
    if dtype_str is None:
        return None

    import torch

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str)


def _load_dataset(config: FinetuneConfig) -> Any:
    """Load and prepare the training dataset."""
    from datasets import load_dataset

    # Check if it's a local file
    dataset_path = config.data.dataset
    if Path(dataset_path).exists():
        # Local file
        if dataset_path.endswith(".json"):
            dataset = load_dataset("json", data_files=dataset_path, split="train")
        elif dataset_path.endswith(".jsonl"):
            dataset = load_dataset("json", data_files=dataset_path, split="train")
        elif dataset_path.endswith(".csv"):
            dataset = load_dataset("csv", data_files=dataset_path, split="train")
        else:
            raise TrainingError(f"Unsupported file format: {dataset_path}")
    else:
        # HuggingFace dataset
        dataset = load_dataset(dataset_path, split=config.data.split)

    # Limit samples if specified
    if config.data.max_samples:
        dataset = dataset.select(range(min(config.data.max_samples, len(dataset))))

    # Apply prompt template if specified
    if config.data.prompt_template:
        dataset = dataset.map(
            lambda x: _apply_prompt_template(x, config.data.prompt_template),
            remove_columns=dataset.column_names,
        )

    logger.info(f"Dataset loaded: {len(dataset)} samples")
    return dataset


def _apply_prompt_template(example: dict, template: str) -> dict:
    """Apply prompt template to a dataset example."""
    # Handle common dataset formats
    text = template.format(
        instruction=example.get("instruction", ""),
        input=example.get("input", ""),
        output=example.get("output", ""),
        text=example.get("text", ""),
        question=example.get("question", ""),
        answer=example.get("answer", ""),
        context=example.get("context", ""),
    )
    return {"text": text}


def _save_run_config(config: FinetuneConfig, output_dir: Path) -> None:
    """Save the configuration for this run."""
    import yaml

    config_path = output_dir / "run_config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(
            config.model_dump(mode="json", exclude_none=True),
            f,
            default_flow_style=False,
            sort_keys=False,
        )
    logger.debug(f"Saved run config to: {config_path}")


def _save_run_metadata(
    config: FinetuneConfig,
    output_dir: Path,
    resume_from: Optional[Path] = None,
) -> None:
    """Save metadata for this training run."""
    metadata = {
        "run_name": config.output.run_name,
        "model": config.model.name,
        "started_at": datetime.now().isoformat(),
        "status": "running",
        "resumed_from": str(resume_from) if resume_from else None,
    }

    metadata_path = output_dir / "run_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.debug(f"Saved run metadata to: {metadata_path}")


def _update_run_metadata(output_dir: Path, **updates: Any) -> None:
    """Update metadata for this training run."""
    metadata_path = output_dir / "run_metadata.json"

    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    metadata.update(updates)
    metadata["updated_at"] = datetime.now().isoformat()

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
