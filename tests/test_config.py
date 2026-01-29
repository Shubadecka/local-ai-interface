"""Tests for configuration loading and validation."""

from pathlib import Path

import pytest
import yaml

from finetune.config.loader import (
    ConfigError,
    generate_default_config,
    load_config,
    load_yaml,
    save_config,
)
from finetune.config.schema import (
    DataConfig,
    FinetuneConfig,
    LoRAConfig,
    ModelConfig,
    OutputConfig,
    TrainingConfig,
)


class TestModelConfig:
    """Tests for ModelConfig schema."""

    def test_valid_model_config(self):
        """Test valid model configuration."""
        config = ModelConfig(name="unsloth/Meta-Llama-3.1-8B-bnb-4bit")
        assert config.name == "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
        assert config.max_seq_length == 2048
        assert config.load_in_4bit is True

    def test_model_name_validation(self):
        """Test that empty model name is rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ModelConfig(name="  ")

    def test_model_name_stripped(self):
        """Test that model name is stripped of whitespace."""
        config = ModelConfig(name="  model/name  ")
        assert config.name == "model/name"


class TestLoRAConfig:
    """Tests for LoRAConfig schema."""

    def test_default_lora_config(self):
        """Test default LoRA configuration."""
        config = LoRAConfig()
        assert config.r == 16
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.0
        assert "q_proj" in config.target_modules

    def test_lora_rank_bounds(self):
        """Test LoRA rank bounds validation."""
        with pytest.raises(ValueError):
            LoRAConfig(r=0)

        with pytest.raises(ValueError):
            LoRAConfig(r=300)

    def test_lora_dropout_bounds(self):
        """Test LoRA dropout bounds validation."""
        with pytest.raises(ValueError):
            LoRAConfig(lora_dropout=-0.1)

        with pytest.raises(ValueError):
            LoRAConfig(lora_dropout=1.5)


class TestTrainingConfig:
    """Tests for TrainingConfig schema."""

    def test_default_training_config(self):
        """Test default training configuration."""
        config = TrainingConfig()
        assert config.num_epochs == 1
        assert config.batch_size == 2
        assert config.learning_rate == 2e-4

    def test_learning_rate_positive(self):
        """Test that learning rate must be positive."""
        with pytest.raises(ValueError):
            TrainingConfig(learning_rate=0)

        with pytest.raises(ValueError):
            TrainingConfig(learning_rate=-0.001)


class TestDataConfig:
    """Tests for DataConfig schema."""

    def test_valid_data_config(self):
        """Test valid data configuration."""
        config = DataConfig(dataset="yahma/alpaca-cleaned")
        assert config.dataset == "yahma/alpaca-cleaned"
        assert config.split == "train"

    def test_max_samples_positive(self):
        """Test that max_samples must be positive if set."""
        with pytest.raises(ValueError):
            DataConfig(dataset="test", max_samples=0)


class TestOutputConfig:
    """Tests for OutputConfig schema."""

    def test_valid_output_config(self):
        """Test valid output configuration."""
        config = OutputConfig(run_name="my-test-run")
        assert config.run_name == "my-test-run"
        assert config.output_dir == Path("./checkpoints")

    def test_run_name_validation(self):
        """Test that empty run name is rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            OutputConfig(run_name="  ")

    def test_run_name_spaces_replaced(self):
        """Test that spaces in run name are replaced with hyphens."""
        config = OutputConfig(run_name="my test run")
        assert config.run_name == "my-test-run"


class TestFinetuneConfig:
    """Tests for complete FinetuneConfig."""

    def test_valid_config(self, sample_config_dict):
        """Test valid complete configuration."""
        config = FinetuneConfig.model_validate(sample_config_dict)
        assert config.model.name == "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
        assert config.output.run_name == "test-run"

    def test_get_output_path(self, sample_config_dict):
        """Test output path generation."""
        config = FinetuneConfig.model_validate(sample_config_dict)
        expected = Path("./checkpoints") / "test-run"
        assert config.get_output_path() == expected

    def test_extra_fields_forbidden(self, sample_config_dict):
        """Test that extra fields are rejected."""
        sample_config_dict["unknown_field"] = "value"
        with pytest.raises(ValueError):
            FinetuneConfig.model_validate(sample_config_dict)


class TestConfigLoader:
    """Tests for configuration file loading."""

    def test_load_yaml(self, tmp_config_dir):
        """Test YAML file loading."""
        config_path = tmp_config_dir / "test.yaml"
        config_path.write_text("key: value\nnested:\n  inner: data")

        result = load_yaml(config_path)
        assert result == {"key": "value", "nested": {"inner": "data"}}

    def test_load_yaml_not_found(self):
        """Test loading non-existent file."""
        with pytest.raises(ConfigError, match="not found"):
            load_yaml("/nonexistent/path.yaml")

    def test_load_yaml_invalid(self, tmp_config_dir):
        """Test loading invalid YAML."""
        config_path = tmp_config_dir / "invalid.yaml"
        config_path.write_text("key: [invalid yaml")

        with pytest.raises(ConfigError, match="Invalid YAML"):
            load_yaml(config_path)

    def test_load_config(self, tmp_config_dir, sample_config_dict):
        """Test full configuration loading."""
        config_path = tmp_config_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(sample_config_dict, f)

        config = load_config(config_path)
        assert isinstance(config, FinetuneConfig)
        assert config.model.name == "unsloth/Meta-Llama-3.1-8B-bnb-4bit"

    def test_load_config_validation_error(self, tmp_config_dir):
        """Test configuration validation errors."""
        config_path = tmp_config_dir / "invalid.yaml"
        config_path.write_text("model:\n  name: ''")

        with pytest.raises(ConfigError, match="validation failed"):
            load_config(config_path)

    def test_save_config(self, tmp_config_dir, sample_config_dict):
        """Test configuration saving."""
        config = FinetuneConfig.model_validate(sample_config_dict)
        output_path = tmp_config_dir / "output.yaml"

        save_config(config, output_path)

        assert output_path.exists()
        loaded = load_config(output_path)
        assert loaded.model.name == config.model.name

    def test_generate_default_config(self):
        """Test default configuration generation."""
        default = generate_default_config()
        assert "model" in default
        assert "lora" in default
        assert "training" in default
        assert "data" in default
        assert "output" in default

        # Should be valid
        config = FinetuneConfig.model_validate(default)
        assert config.model.name == "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
