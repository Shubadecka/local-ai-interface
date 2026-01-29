"""List command for viewing runs and checkpoints."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

console = Console()

app = typer.Typer(
    help="List training runs and checkpoints.",
    no_args_is_help=True,
)


@app.command()
def runs(
    checkpoints_dir: Optional[Path] = typer.Option(
        None,
        "--dir",
        "-d",
        help="Checkpoints directory. Defaults to ./checkpoints/",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed information.",
    ),
) -> None:
    """List all training runs.

    [bold]Example:[/bold]

        finetune list runs
        finetune list runs --verbose
    """
    from finetune.core.checkpoint import CheckpointManager
    from finetune.utils.paths import get_checkpoints_dir

    if checkpoints_dir is None:
        checkpoints_dir = get_checkpoints_dir()

    manager = CheckpointManager(checkpoints_dir)
    training_runs = manager.list_runs()

    if not training_runs:
        console.print("[yellow]No training runs found.[/yellow]")
        console.print(f"Looking in: {checkpoints_dir}")
        return

    table = Table(title="Training Runs")
    table.add_column("Run Name", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("Checkpoints", justify="right")
    table.add_column("Last Modified", style="dim")

    if verbose:
        table.add_column("Path", style="dim")

    for run in training_runs:
        row = [
            run["name"],
            run.get("model", "Unknown"),
            run.get("status", "Unknown"),
            str(run.get("checkpoint_count", 0)),
            run.get("last_modified", "Unknown"),
        ]
        if verbose:
            row.append(str(run.get("path", "")))
        table.add_row(*row)

    console.print(table)


@app.command()
def checkpoints(
    run_name: str = typer.Argument(
        ...,
        help="Name of the training run.",
    ),
    checkpoints_dir: Optional[Path] = typer.Option(
        None,
        "--dir",
        "-d",
        help="Checkpoints directory. Defaults to ./checkpoints/",
    ),
) -> None:
    """List checkpoints for a specific training run.

    [bold]Example:[/bold]

        finetune list checkpoints my-run-name
    """
    from finetune.core.checkpoint import CheckpointManager
    from finetune.utils.paths import get_checkpoints_dir

    if checkpoints_dir is None:
        checkpoints_dir = get_checkpoints_dir()

    manager = CheckpointManager(checkpoints_dir)
    run_checkpoints = manager.list_checkpoints(run_name)

    if run_checkpoints is None:
        console.print(f"[red]Run not found:[/red] {run_name}")
        raise typer.Exit(1)

    if not run_checkpoints:
        console.print(f"[yellow]No checkpoints found for run:[/yellow] {run_name}")
        return

    table = Table(title=f"Checkpoints for: {run_name}")
    table.add_column("Checkpoint", style="cyan")
    table.add_column("Step", justify="right", style="green")
    table.add_column("Size", justify="right")
    table.add_column("Created", style="dim")

    for checkpoint in run_checkpoints:
        table.add_row(
            checkpoint["name"],
            str(checkpoint.get("step", "N/A")),
            checkpoint.get("size", "Unknown"),
            checkpoint.get("created", "Unknown"),
        )

    console.print(table)

    # Show path to latest checkpoint
    if run_checkpoints:
        latest = run_checkpoints[-1]
        console.print(f"\n[bold]Latest checkpoint:[/bold] {latest.get('path', '')}")
        console.print(
            f"\n[dim]Resume training with:[/dim]\n"
            f"  finetune train --resume {latest.get('path', '')}"
        )
