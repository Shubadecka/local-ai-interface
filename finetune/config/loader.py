"""YAML configuration loader."""

from pathlib import Path
from typing import Any, Union

import yaml
from pydantic import ValidationError

from finetune.config.schema import FinetuneConfig
from finetune.utils.logging import get_logger

logger = get_logger("config")


class ConfigError(Exception):
    """Error loading or validating configuration."""

    pass


def load_yaml(path: Union[str, Path]) -> dict[str, Any]:
    """Load a YAML file.

    Args:
        path: Path to YAML file

    Returns:
        Parsed YAML content as dictionary

    Raises:
        ConfigError: If file cannot be read or parsed
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")

    try:
        with open(path, "r") as f:
            content = yaml.safe_load(f)
            if content is None:
                return {}
            if not isinstance(content, dict):
                raise ConfigError(f"Configuration must be a YAML mapping, got {type(content)}")
            return content
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}: {e}") from e


def load_config(path: Union[str, Path]) -> FinetuneConfig:
    """Load and validate a finetuning configuration.

    Args:
        path: Path to YAML configuration file

    Returns:
        Validated FinetuneConfig object

    Raises:
        ConfigError: If configuration is invalid
    """
    logger.info(f"Loading configuration from {path}")
    raw_config = load_yaml(path)

    try:
        config = FinetuneConfig.model_validate(raw_config)
        logger.info(f"Configuration loaded successfully for run: {config.output.run_name}")
        return config
    except ValidationError as e:
        error_messages = []
        for error in e.errors():
            location = ".".join(str(loc) for loc in error["loc"])
            message = error["msg"]
            error_messages.append(f"  - {location}: {message}")
        raise ConfigError(
            f"Configuration validation failed:\n" + "\n".join(error_messages)
        ) from e


def save_config(config: FinetuneConfig, path: Union[str, Path]) -> None:
    """Save a configuration to a YAML file.

    Args:
        config: Configuration to save
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.safe_dump(
            config.model_dump(mode="json", exclude_none=True),
            f,
            default_flow_style=False,
            sort_keys=False,
        )
    logger.info(f"Configuration saved to {path}")


def generate_default_config() -> dict[str, Any]:
    """Generate a default configuration template.

    Returns:
        Dictionary with default configuration values
    """
    return {
        "model": {
            "name": "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
            "max_seq_length": 2048,
            "load_in_4bit": True,
        },
        "lora": {
            "r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
        "training": {
            "num_epochs": 1,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "warmup_steps": 5,
            "logging_steps": 1,
            "save_steps": 100,
        },
        "data": {
            "dataset": "yahma/alpaca-cleaned",
            "split": "train",
            "max_samples": 1000,
        },
        "output": {
            "run_name": "my-finetune-run",
            "output_dir": "./checkpoints",
        },
    }
