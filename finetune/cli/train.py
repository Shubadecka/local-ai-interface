"""Train command for finetuning LLMs."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from finetune.config.loader import ConfigError, load_config
from finetune.utils.logging import setup_logging

console = Console()


def train(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to training configuration YAML file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    resume: Optional[Path] = typer.Option(
        None,
        "--resume",
        "-r",
        help="Path to checkpoint directory to resume from.",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate config and show what would be done without training.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging.",
    ),
) -> None:
    """Start or resume finetuning a language model.

    You must provide either --config to start a new training run,
    or --resume to continue from an existing checkpoint.

    [bold]Examples:[/bold]

        # Start new training
        finetune train --config configs/my-config.yaml

        # Resume from checkpoint
        finetune train --resume checkpoints/my-run/checkpoint-500

        # Dry run to validate config
        finetune train --config configs/my-config.yaml --dry-run
    """
    import logging

    setup_logging(level=logging.DEBUG if verbose else logging.INFO)

    if config is None and resume is None:
        console.print(
            "[red]Error:[/red] You must provide either --config or --resume",
            style="bold",
        )
        raise typer.Exit(1)

    if config is not None and resume is not None:
        console.print(
            "[red]Error:[/red] Cannot use both --config and --resume. "
            "Use --resume to continue training from a checkpoint.",
            style="bold",
        )
        raise typer.Exit(1)

    try:
        if resume:
            _resume_training(resume, dry_run=dry_run)
        else:
            assert config is not None
            _start_training(config, dry_run=dry_run)
    except ConfigError as e:
        console.print(f"[red]Configuration Error:[/red] {e}")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user.[/yellow]")
        raise typer.Exit(130)


def _start_training(config_path: Path, dry_run: bool = False) -> None:
    """Start a new training run from configuration."""
    console.print(f"\n[bold blue]Loading configuration from:[/bold blue] {config_path}")

    finetune_config = load_config(config_path)

    # Display configuration summary
    _display_config_summary(finetune_config)

    if dry_run:
        console.print(
            Panel(
                "[green]Dry run complete.[/green] Configuration is valid.",
                title="Dry Run",
            )
        )
        return

    console.print("\n[bold green]Starting training...[/bold green]\n")

    # Import trainer here to avoid loading heavy dependencies for dry-run
    from finetune.core.trainer import run_training

    run_training(finetune_config)


def _resume_training(checkpoint_path: Path, dry_run: bool = False) -> None:
    """Resume training from a checkpoint."""
    console.print(f"\n[bold blue]Resuming from checkpoint:[/bold blue] {checkpoint_path}")

    # Look for run_config.yaml in parent directory
    run_dir = checkpoint_path.parent
    if checkpoint_path.name.startswith("checkpoint-"):
        # We're in a checkpoint subdirectory
        run_dir = checkpoint_path.parent

    config_path = run_dir / "run_config.yaml"
    if not config_path.exists():
        console.print(
            f"[red]Error:[/red] Could not find run_config.yaml in {run_dir}",
            style="bold",
        )
        raise typer.Exit(1)

    finetune_config = load_config(config_path)

    # Display configuration summary
    _display_config_summary(finetune_config, resume_from=checkpoint_path)

    if dry_run:
        console.print(
            Panel(
                "[green]Dry run complete.[/green] Checkpoint and configuration are valid.",
                title="Dry Run",
            )
        )
        return

    console.print("\n[bold green]Resuming training...[/bold green]\n")

    # Import trainer here to avoid loading heavy dependencies for dry-run
    from finetune.core.trainer import run_training

    run_training(finetune_config, resume_from=checkpoint_path)


def _display_config_summary(config, resume_from: Optional[Path] = None) -> None:
    """Display a summary of the training configuration."""
    from rich.table import Table

    table = Table(title="Training Configuration", show_header=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Run Name", config.output.run_name)
    table.add_row("Model", config.model.name)
    table.add_row("Max Sequence Length", str(config.model.max_seq_length))
    table.add_row("LoRA Rank", str(config.lora.r))
    table.add_row("Dataset", config.data.dataset)
    table.add_row("Epochs", str(config.training.num_epochs))
    table.add_row("Batch Size", str(config.training.batch_size))
    table.add_row("Learning Rate", f"{config.training.learning_rate:.2e}")
    table.add_row("Output Directory", str(config.output.output_dir))

    if resume_from:
        table.add_row("Resume From", str(resume_from))

    console.print(table)
