"""Checkpoint management for training runs."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from finetune.utils.logging import get_logger

logger = get_logger("checkpoint")


class CheckpointManager:
    """Manages training checkpoints and run discovery."""

    def __init__(self, checkpoints_dir: Path):
        """Initialize checkpoint manager.

        Args:
            checkpoints_dir: Base directory containing training runs
        """
        self.checkpoints_dir = Path(checkpoints_dir)

    def list_runs(self) -> list[dict[str, Any]]:
        """List all training runs.

        Returns:
            List of run information dictionaries
        """
        if not self.checkpoints_dir.exists():
            return []

        runs = []
        for run_dir in sorted(self.checkpoints_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            # Skip hidden directories
            if run_dir.name.startswith("."):
                continue

            run_info = self._get_run_info(run_dir)
            if run_info:
                runs.append(run_info)

        # Sort by last modified (most recent first)
        runs.sort(key=lambda x: x.get("last_modified_ts", 0), reverse=True)
        return runs

    def list_checkpoints(self, run_name: str) -> Optional[list[dict[str, Any]]]:
        """List checkpoints for a specific run.

        Args:
            run_name: Name of the training run

        Returns:
            List of checkpoint information, or None if run not found
        """
        run_dir = self.checkpoints_dir / run_name
        if not run_dir.exists():
            return None

        checkpoints = []

        # Find checkpoint directories
        for item in sorted(run_dir.iterdir()):
            if not item.is_dir():
                continue

            # Match checkpoint-N or final
            if item.name.startswith("checkpoint-") or item.name == "final":
                checkpoint_info = self._get_checkpoint_info(item)
                if checkpoint_info:
                    checkpoints.append(checkpoint_info)

        # Sort by step number
        checkpoints.sort(key=lambda x: x.get("step", float("inf")))
        return checkpoints

    def get_latest_checkpoint(self, run_name: str) -> Optional[Path]:
        """Get the path to the latest checkpoint for a run.

        Args:
            run_name: Name of the training run

        Returns:
            Path to latest checkpoint, or None if not found
        """
        checkpoints = self.list_checkpoints(run_name)
        if not checkpoints:
            return None

        return Path(checkpoints[-1]["path"])

    def get_run_config(self, run_name: str) -> Optional[dict[str, Any]]:
        """Get the configuration for a training run.

        Args:
            run_name: Name of the training run

        Returns:
            Configuration dictionary, or None if not found
        """
        import yaml

        run_dir = self.checkpoints_dir / run_name
        config_path = run_dir / "run_config.yaml"

        if not config_path.exists():
            return None

        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def get_run_metadata(self, run_name: str) -> Optional[dict[str, Any]]:
        """Get metadata for a training run.

        Args:
            run_name: Name of the training run

        Returns:
            Metadata dictionary, or None if not found
        """
        run_dir = self.checkpoints_dir / run_name
        metadata_path = run_dir / "run_metadata.json"

        if not metadata_path.exists():
            return None

        with open(metadata_path, "r") as f:
            return json.load(f)

    def _get_run_info(self, run_dir: Path) -> Optional[dict[str, Any]]:
        """Extract information about a training run.

        Args:
            run_dir: Path to the run directory

        Returns:
            Run information dictionary
        """
        info: dict[str, Any] = {
            "name": run_dir.name,
            "path": str(run_dir),
        }

        # Get model from config if available
        config = self.get_run_config(run_dir.name)
        if config:
            info["model"] = config.get("model", {}).get("name", "Unknown")

        # Get status from metadata if available
        metadata = self.get_run_metadata(run_dir.name)
        if metadata:
            info["status"] = metadata.get("status", "Unknown")
        else:
            # Infer status from directory contents
            if (run_dir / "final").exists():
                info["status"] = "completed"
            elif any(run_dir.glob("checkpoint-*")):
                info["status"] = "in_progress"
            else:
                info["status"] = "unknown"

        # Count checkpoints
        checkpoints = list(run_dir.glob("checkpoint-*"))
        if (run_dir / "final").exists():
            checkpoints.append(run_dir / "final")
        info["checkpoint_count"] = len(checkpoints)

        # Get last modified time
        try:
            mtime = run_dir.stat().st_mtime
            info["last_modified_ts"] = mtime
            info["last_modified"] = datetime.fromtimestamp(mtime).strftime(
                "%Y-%m-%d %H:%M"
            )
        except OSError:
            info["last_modified"] = "Unknown"

        return info

    def _get_checkpoint_info(self, checkpoint_dir: Path) -> Optional[dict[str, Any]]:
        """Extract information about a checkpoint.

        Args:
            checkpoint_dir: Path to the checkpoint directory

        Returns:
            Checkpoint information dictionary
        """
        info: dict[str, Any] = {
            "name": checkpoint_dir.name,
            "path": str(checkpoint_dir),
        }

        # Extract step number from name
        if checkpoint_dir.name == "final":
            info["step"] = float("inf")  # Sort final last
            info["name"] = "final"
        else:
            match = re.match(r"checkpoint-(\d+)", checkpoint_dir.name)
            if match:
                info["step"] = int(match.group(1))

        # Calculate directory size
        info["size"] = self._get_directory_size(checkpoint_dir)

        # Get creation time
        try:
            ctime = checkpoint_dir.stat().st_ctime
            info["created"] = datetime.fromtimestamp(ctime).strftime("%Y-%m-%d %H:%M")
        except OSError:
            info["created"] = "Unknown"

        return info

    def _get_directory_size(self, path: Path) -> str:
        """Calculate the total size of a directory.

        Args:
            path: Directory path

        Returns:
            Human-readable size string
        """
        try:
            total_size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            return self._format_size(total_size)
        except OSError:
            return "Unknown"

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format bytes as human-readable string.

        Args:
            size_bytes: Size in bytes

        Returns:
            Human-readable size string
        """
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} PB"


def find_checkpoint(path: Path) -> Optional[Path]:
    """Find a valid checkpoint from a path.

    Handles various input formats:
    - Direct checkpoint directory
    - Run directory (returns latest checkpoint)
    - Checkpoint name within a run

    Args:
        path: Input path

    Returns:
        Path to checkpoint directory, or None if not found
    """
    if not path.exists():
        return None

    # Check if it's a direct checkpoint directory
    if _is_checkpoint_dir(path):
        return path

    # Check if it's a run directory
    if (path / "run_config.yaml").exists():
        # Find latest checkpoint
        checkpoints = list(path.glob("checkpoint-*"))
        if (path / "final").exists():
            return path / "final"
        if checkpoints:
            # Sort by step number and return latest
            checkpoints.sort(key=lambda p: int(p.name.split("-")[1]))
            return checkpoints[-1]

    return None


def _is_checkpoint_dir(path: Path) -> bool:
    """Check if a path is a valid checkpoint directory.

    Args:
        path: Path to check

    Returns:
        True if valid checkpoint directory
    """
    # Check for common checkpoint files
    checkpoint_markers = [
        "adapter_model.safetensors",
        "adapter_config.json",
        "pytorch_model.bin",
        "model.safetensors",
    ]

    return any((path / marker).exists() for marker in checkpoint_markers)
