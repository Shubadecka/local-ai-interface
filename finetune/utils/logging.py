"""Logging setup for the finetuning CLI."""

import logging
import sys
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

console = Console()


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Configure logging with rich formatting.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional path to log file

    Returns:
        Configured logger instance
    """
    handlers: list[logging.Handler] = [
        RichHandler(
            console=console,
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
        )
    ]

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )

    # Reduce noise from transformers/datasets
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

    return logging.getLogger("finetune")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module.

    Args:
        name: Module name

    Returns:
        Logger instance
    """
    return logging.getLogger(f"finetune.{name}")
