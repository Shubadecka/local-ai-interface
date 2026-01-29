"""Tests for CLI commands."""

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from finetune.cli.main import app

runner = CliRunner()


class TestMainCLI:
    """Tests for main CLI app."""

    def test_help(self):
        """Test --help flag."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "LLM Finetuning CLI" in result.output

    def test_version(self):
        """Test --version flag."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "finetune" in result.output
        assert "0.1.0" in result.output

    def test_no_args(self):
        """Test that no args shows help."""
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        # Should show help message


class TestConfigCommands:
    """Tests for config subcommands."""

    def test_config_help(self):
        """Test config --help."""
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0
        assert "Configuration management" in result.output

    def test_config_generate(self, tmp_path):
        """Test config generate command."""
        output_path = tmp_path / "test-config.yaml"
        result = runner.invoke(app, ["config", "generate", "-o", str(output_path)])

        assert result.exit_code == 0
        assert output_path.exists()

        # Verify content
        with open(output_path) as f:
            config = yaml.safe_load(f)
        assert "model" in config
        assert "lora" in config

    def test_config_generate_no_overwrite(self, tmp_path):
        """Test that generate doesn't overwrite without --force."""
        output_path = tmp_path / "existing.yaml"
        output_path.write_text("existing content")

        result = runner.invoke(app, ["config", "generate", "-o", str(output_path)])

        assert result.exit_code == 1
        assert "already exists" in result.output

    def test_config_generate_force(self, tmp_path):
        """Test that generate overwrites with --force."""
        output_path = tmp_path / "existing.yaml"
        output_path.write_text("existing content")

        result = runner.invoke(
            app, ["config", "generate", "-o", str(output_path), "--force"]
        )

        assert result.exit_code == 0
        assert "model" in output_path.read_text()

    def test_config_validate_valid(self, tmp_path, sample_config_dict):
        """Test config validate with valid config."""
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(sample_config_dict, f)

        result = runner.invoke(app, ["config", "validate", str(config_path)])

        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_config_validate_invalid(self, tmp_path):
        """Test config validate with invalid config."""
        config_path = tmp_path / "invalid.yaml"
        config_path.write_text("model:\n  name: ''")

        result = runner.invoke(app, ["config", "validate", str(config_path)])

        assert result.exit_code == 1
        assert "failed" in result.output.lower()

    def test_config_show(self, tmp_path, sample_config_dict):
        """Test config show command."""
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(sample_config_dict, f)

        result = runner.invoke(app, ["config", "show", str(config_path)])

        assert result.exit_code == 0


class TestListCommands:
    """Tests for list subcommands."""

    def test_list_help(self):
        """Test list --help."""
        result = runner.invoke(app, ["list", "--help"])
        assert result.exit_code == 0

    def test_list_runs_empty(self, tmp_path):
        """Test list runs with empty directory."""
        result = runner.invoke(app, ["list", "runs", "--dir", str(tmp_path)])

        assert result.exit_code == 0
        assert "No training runs found" in result.output

    def test_list_runs(self, tmp_path):
        """Test list runs with runs present."""
        run_dir = tmp_path / "test-run"
        run_dir.mkdir()
        (run_dir / "run_config.yaml").write_text(
            yaml.dump({"model": {"name": "test-model"}})
        )
        (run_dir / "final").mkdir()

        result = runner.invoke(app, ["list", "runs", "--dir", str(tmp_path)])

        assert result.exit_code == 0
        assert "test-run" in result.output

    def test_list_checkpoints_not_found(self, tmp_path):
        """Test list checkpoints for non-existent run."""
        result = runner.invoke(
            app, ["list", "checkpoints", "nonexistent", "--dir", str(tmp_path)]
        )

        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_list_checkpoints(self, tmp_path):
        """Test list checkpoints for a run."""
        run_dir = tmp_path / "test-run"
        run_dir.mkdir()
        (run_dir / "checkpoint-100").mkdir()
        (run_dir / "checkpoint-100" / "adapter_model.safetensors").touch()

        result = runner.invoke(
            app, ["list", "checkpoints", "test-run", "--dir", str(tmp_path)]
        )

        assert result.exit_code == 0
        assert "checkpoint-100" in result.output


class TestTrainCommand:
    """Tests for train command."""

    def test_train_help(self):
        """Test train --help."""
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "Start or resume finetuning" in result.output

    def test_train_no_args(self):
        """Test train without required args."""
        result = runner.invoke(app, ["train"])

        assert result.exit_code == 1
        assert "must provide" in result.output.lower()

    def test_train_both_args(self, tmp_path, sample_config_dict):
        """Test train with both --config and --resume."""
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(sample_config_dict, f)

        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()

        result = runner.invoke(
            app,
            ["train", "--config", str(config_path), "--resume", str(checkpoint_dir)],
        )

        assert result.exit_code == 1
        assert "cannot use both" in result.output.lower()

    def test_train_dry_run(self, tmp_path, sample_config_dict):
        """Test train --dry-run validates without training."""
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(sample_config_dict, f)

        result = runner.invoke(
            app, ["train", "--config", str(config_path), "--dry-run"]
        )

        assert result.exit_code == 0
        assert "dry run" in result.output.lower()


class TestConvertCommand:
    """Tests for convert command."""

    def test_convert_help(self):
        """Test convert --help."""
        result = runner.invoke(app, ["convert", "--help"])
        assert result.exit_code == 0
        assert "Convert a finetuned checkpoint" in result.output

    def test_convert_invalid_quantization(self, tmp_path):
        """Test convert with invalid quantization method."""
        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()

        result = runner.invoke(
            app,
            ["convert", str(checkpoint), "--name", "test", "--quantize", "invalid"],
        )

        assert result.exit_code == 1
        assert "invalid quantization" in result.output.lower()

    def test_convert_invalid_name(self, tmp_path):
        """Test convert with invalid model name."""
        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()

        result = runner.invoke(
            app, ["convert", str(checkpoint), "--name", "invalid name!@#"]
        )

        assert result.exit_code == 1
        assert "alphanumeric" in result.output.lower()


@pytest.fixture
def sample_config_dict():
    """Return a sample configuration dictionary."""
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
            "target_modules": ["q_proj", "k_proj", "v_proj"],
        },
        "training": {
            "num_epochs": 1,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
        },
        "data": {
            "dataset": "yahma/alpaca-cleaned",
            "split": "train",
        },
        "output": {
            "run_name": "test-run",
            "output_dir": "./checkpoints",
        },
    }
