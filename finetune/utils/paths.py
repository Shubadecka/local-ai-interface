"""Path management utilities."""

from pathlib import Path
from typing import Optional

# Default directories relative to project root
DEFAULT_CHECKPOINTS_DIR = "checkpoints"
DEFAULT_EXPORTS_DIR = "exports"
DEFAULT_CONFIGS_DIR = "configs"
DEFAULT_DATA_DIR = "data"


def get_project_root() -> Path:
    """Get the project root directory.

    Returns:
        Path to project root
    """
    # Start from current file and traverse up to find pyproject.toml
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback to current working directory
    return Path.cwd()


def get_checkpoints_dir(base_dir: Optional[Path] = None) -> Path:
    """Get the checkpoints directory.

    Args:
        base_dir: Optional base directory override

    Returns:
        Path to checkpoints directory
    """
    if base_dir:
        return base_dir / DEFAULT_CHECKPOINTS_DIR
    return get_project_root() / DEFAULT_CHECKPOINTS_DIR


def get_exports_dir(base_dir: Optional[Path] = None) -> Path:
    """Get the exports directory.

    Args:
        base_dir: Optional base directory override

    Returns:
        Path to exports directory
    """
    if base_dir:
        return base_dir / DEFAULT_EXPORTS_DIR
    return get_project_root() / DEFAULT_EXPORTS_DIR


def get_configs_dir(base_dir: Optional[Path] = None) -> Path:
    """Get the configs directory.

    Args:
        base_dir: Optional base directory override

    Returns:
        Path to configs directory
    """
    if base_dir:
        return base_dir / DEFAULT_CONFIGS_DIR
    return get_project_root() / DEFAULT_CONFIGS_DIR


def get_data_dir(base_dir: Optional[Path] = None) -> Path:
    """Get the data directory.

    Args:
        base_dir: Optional base directory override

    Returns:
        Path to data directory
    """
    if base_dir:
        return base_dir / DEFAULT_DATA_DIR
    return get_project_root() / DEFAULT_DATA_DIR


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        The same path (for chaining)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
