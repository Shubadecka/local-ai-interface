"""Main CLI application."""

import typer
from rich.console import Console
from dotenv import load_dotenv

from finetune import __version__
from finetune.cli import config as config_cmd
from finetune.cli import convert as convert_cmd
from finetune.cli import list as list_cmd
from finetune.cli import train as train_cmd

load_dotenv()

console = Console()

app = typer.Typer(
    name="finetune",
    help="CLI tool for LLM finetuning with Ollama integration.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Add subcommands
app.add_typer(config_cmd.app, name="config", help="Configuration management")
app.add_typer(list_cmd.app, name="list", help="List runs and checkpoints")
app.command()(train_cmd.train)
app.command()(convert_cmd.convert)


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"[bold blue]finetune[/bold blue] version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """LLM Finetuning CLI with Ollama integration.

    Use this tool to finetune language models and deploy them to Ollama.

    [bold]Examples:[/bold]

        # Generate a config template
        finetune config generate

        # Start training
        finetune train --config configs/my-config.yaml

        # Resume from checkpoint
        finetune train --resume checkpoints/my-run/checkpoint-500

        # Convert to Ollama format
        finetune convert checkpoints/my-run/final --name my-model

        # List training runs
        finetune list runs
    """
    pass


if __name__ == "__main__":
    app()
