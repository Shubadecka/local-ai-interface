"""Pytest fixtures for the test suite."""

from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def tmp_config_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for test configs."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def tmp_checkpoint_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for test checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


@pytest.fixture
def sample_config_dict() -> dict:
    """Return a sample configuration dictionary for testing."""
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
            "run_name": "test-run",
            "output_dir": "./checkpoints",
        },
    }
