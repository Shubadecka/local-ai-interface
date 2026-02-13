"""Tests for checkpoint management."""

import json

import yaml

from finetune.core.checkpoint import CheckpointManager, find_checkpoint


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    def test_list_runs_empty(self, tmp_checkpoint_dir):
        """Test listing runs when directory is empty."""
        manager = CheckpointManager(tmp_checkpoint_dir)
        runs = manager.list_runs()
        assert runs == []

    def test_list_runs_nonexistent(self, tmp_path):
        """Test listing runs when directory doesn't exist."""
        manager = CheckpointManager(tmp_path / "nonexistent")
        runs = manager.list_runs()
        assert runs == []

    def test_list_runs_with_runs(self, tmp_checkpoint_dir):
        """Test listing runs with actual run directories."""
        # Create mock run directories
        run1 = tmp_checkpoint_dir / "run-1"
        run1.mkdir()
        (run1 / "run_config.yaml").write_text(yaml.dump({"model": {"name": "model-1"}}))
        (run1 / "run_metadata.json").write_text(json.dumps({"status": "completed"}))

        run2 = tmp_checkpoint_dir / "run-2"
        run2.mkdir()
        (run2 / "run_config.yaml").write_text(yaml.dump({"model": {"name": "model-2"}}))

        manager = CheckpointManager(tmp_checkpoint_dir)
        runs = manager.list_runs()

        assert len(runs) == 2
        names = [r["name"] for r in runs]
        assert "run-1" in names
        assert "run-2" in names

    def test_list_runs_ignores_hidden(self, tmp_checkpoint_dir):
        """Test that hidden directories are ignored."""
        (tmp_checkpoint_dir / ".hidden").mkdir()
        (tmp_checkpoint_dir / "visible").mkdir()

        manager = CheckpointManager(tmp_checkpoint_dir)
        runs = manager.list_runs()

        assert len(runs) == 1
        assert runs[0]["name"] == "visible"

    def test_list_checkpoints_not_found(self, tmp_checkpoint_dir):
        """Test listing checkpoints for non-existent run."""
        manager = CheckpointManager(tmp_checkpoint_dir)
        result = manager.list_checkpoints("nonexistent")
        assert result is None

    def test_list_checkpoints(self, tmp_checkpoint_dir):
        """Test listing checkpoints for a run."""
        run_dir = tmp_checkpoint_dir / "my-run"
        run_dir.mkdir()

        # Create checkpoint directories
        (run_dir / "checkpoint-100").mkdir()
        (run_dir / "checkpoint-100" / "adapter_model.safetensors").touch()

        (run_dir / "checkpoint-200").mkdir()
        (run_dir / "checkpoint-200" / "adapter_model.safetensors").touch()

        (run_dir / "final").mkdir()
        (run_dir / "final" / "adapter_model.safetensors").touch()

        manager = CheckpointManager(tmp_checkpoint_dir)
        checkpoints = manager.list_checkpoints("my-run")

        assert len(checkpoints) == 3
        # Should be sorted by step
        assert checkpoints[0]["step"] == 100
        assert checkpoints[1]["step"] == 200
        assert checkpoints[2]["name"] == "final"

    def test_get_latest_checkpoint(self, tmp_checkpoint_dir):
        """Test getting the latest checkpoint."""
        run_dir = tmp_checkpoint_dir / "my-run"
        run_dir.mkdir()

        (run_dir / "checkpoint-100").mkdir()
        (run_dir / "checkpoint-100" / "adapter_model.safetensors").touch()
        (run_dir / "checkpoint-200").mkdir()
        (run_dir / "checkpoint-200" / "adapter_model.safetensors").touch()

        manager = CheckpointManager(tmp_checkpoint_dir)
        latest = manager.get_latest_checkpoint("my-run")

        assert latest is not None
        assert latest.name == "checkpoint-200"

    def test_get_latest_checkpoint_prefers_final(self, tmp_checkpoint_dir):
        """Test that final checkpoint is returned if it exists."""
        run_dir = tmp_checkpoint_dir / "my-run"
        run_dir.mkdir()

        (run_dir / "checkpoint-100").mkdir()
        (run_dir / "checkpoint-100" / "adapter_model.safetensors").touch()
        (run_dir / "final").mkdir()
        (run_dir / "final" / "adapter_model.safetensors").touch()

        manager = CheckpointManager(tmp_checkpoint_dir)
        latest = manager.get_latest_checkpoint("my-run")

        assert latest is not None
        assert latest.name == "final"

    def test_get_run_config(self, tmp_checkpoint_dir):
        """Test getting run configuration."""
        run_dir = tmp_checkpoint_dir / "my-run"
        run_dir.mkdir()
        (run_dir / "run_config.yaml").write_text(
            yaml.dump({"model": {"name": "test-model"}})
        )

        manager = CheckpointManager(tmp_checkpoint_dir)
        config = manager.get_run_config("my-run")

        assert config is not None
        assert config["model"]["name"] == "test-model"

    def test_get_run_config_not_found(self, tmp_checkpoint_dir):
        """Test getting config for run without config file."""
        run_dir = tmp_checkpoint_dir / "my-run"
        run_dir.mkdir()

        manager = CheckpointManager(tmp_checkpoint_dir)
        config = manager.get_run_config("my-run")

        assert config is None

    def test_get_run_metadata(self, tmp_checkpoint_dir):
        """Test getting run metadata."""
        run_dir = tmp_checkpoint_dir / "my-run"
        run_dir.mkdir()
        (run_dir / "run_metadata.json").write_text(
            json.dumps({"status": "running", "model": "test"})
        )

        manager = CheckpointManager(tmp_checkpoint_dir)
        metadata = manager.get_run_metadata("my-run")

        assert metadata is not None
        assert metadata["status"] == "running"

    def test_infer_status_completed(self, tmp_checkpoint_dir):
        """Test status inference when final checkpoint exists."""
        run_dir = tmp_checkpoint_dir / "my-run"
        run_dir.mkdir()
        (run_dir / "final").mkdir()

        manager = CheckpointManager(tmp_checkpoint_dir)
        runs = manager.list_runs()

        assert runs[0]["status"] == "completed"

    def test_infer_status_in_progress(self, tmp_checkpoint_dir):
        """Test status inference when checkpoints exist but no final."""
        run_dir = tmp_checkpoint_dir / "my-run"
        run_dir.mkdir()
        (run_dir / "checkpoint-100").mkdir()

        manager = CheckpointManager(tmp_checkpoint_dir)
        runs = manager.list_runs()

        assert runs[0]["status"] == "in_progress"


class TestFindCheckpoint:
    """Tests for find_checkpoint function."""

    def test_find_direct_checkpoint(self, tmp_checkpoint_dir):
        """Test finding a direct checkpoint directory."""
        checkpoint = tmp_checkpoint_dir / "checkpoint-100"
        checkpoint.mkdir()
        (checkpoint / "adapter_model.safetensors").touch()

        result = find_checkpoint(checkpoint)
        assert result == checkpoint

    def test_find_checkpoint_from_run(self, tmp_checkpoint_dir):
        """Test finding latest checkpoint from run directory."""
        run_dir = tmp_checkpoint_dir / "my-run"
        run_dir.mkdir()
        (run_dir / "run_config.yaml").touch()

        (run_dir / "checkpoint-100").mkdir()
        (run_dir / "checkpoint-100" / "adapter_model.safetensors").touch()
        (run_dir / "checkpoint-200").mkdir()
        (run_dir / "checkpoint-200" / "adapter_model.safetensors").touch()

        result = find_checkpoint(run_dir)
        assert result is not None
        assert result.name == "checkpoint-200"

    def test_find_checkpoint_prefers_final(self, tmp_checkpoint_dir):
        """Test that final is returned from run directory."""
        run_dir = tmp_checkpoint_dir / "my-run"
        run_dir.mkdir()
        (run_dir / "run_config.yaml").touch()

        (run_dir / "checkpoint-100").mkdir()
        (run_dir / "checkpoint-100" / "adapter_model.safetensors").touch()
        (run_dir / "final").mkdir()
        (run_dir / "final" / "adapter_model.safetensors").touch()

        result = find_checkpoint(run_dir)
        assert result is not None
        assert result.name == "final"

    def test_find_checkpoint_not_found(self, tmp_path):
        """Test finding checkpoint that doesn't exist."""
        result = find_checkpoint(tmp_path / "nonexistent")
        assert result is None

    def test_find_checkpoint_invalid_dir(self, tmp_checkpoint_dir):
        """Test finding checkpoint from invalid directory."""
        invalid = tmp_checkpoint_dir / "not-a-checkpoint"
        invalid.mkdir()

        result = find_checkpoint(invalid)
        assert result is None
