"""Config command group for configuration management."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()

app = typer.Typer(
    help="Configuration management commands.",
    no_args_is_help=True,
)


@app.command()
def generate(
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path. Defaults to ./configs/config.yaml",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing file.",
    ),
) -> None:
    """Generate a template configuration file.

    Creates a YAML configuration file with sensible defaults
    that you can customize for your training run.

    [bold]Example:[/bold]

        finetune config generate
        finetune config generate -o my-config.yaml
    """
    import yaml

    from finetune.config.loader import generate_default_config
    from finetune.utils.paths import get_configs_dir

    if output is None:
        output = get_configs_dir() / "config.yaml"

    if output.exists() and not force:
        console.print(
            f"[red]Error:[/red] File already exists: {output}\n"
            "Use --force to overwrite."
        )
        raise typer.Exit(1)

    # Generate default config
    config = generate_default_config()

    # Ensure directory exists
    output.parent.mkdir(parents=True, exist_ok=True)

    # Write YAML
    with open(output, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    console.print(
        Panel(
            f"[green]Configuration template created:[/green] {output}\n\n"
            "Edit this file to customize your training run, then start training with:\n"
            f"  finetune train --config {output}",
            title="Config Generated",
        )
    )


@app.command()
def validate(
    config_path: Path = typer.Argument(
        ...,
        help="Path to configuration file to validate.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Validate a configuration file.

    Checks that the configuration is valid YAML and
    conforms to the expected schema.

    [bold]Example:[/bold]

        finetune config validate configs/my-config.yaml
    """
    from finetune.config.loader import ConfigError, load_config

    try:
        config = load_config(config_path)
        console.print(
            Panel(
                f"[green]Configuration is valid![/green]\n\n"
                f"Run Name: {config.output.run_name}\n"
                f"Model: {config.model.name}\n"
                f"Dataset: {config.data.dataset}",
                title="Validation Passed",
            )
        )
    except ConfigError as e:
        console.print(f"[red]Validation failed:[/red]\n{e}")
        raise typer.Exit(1)


@app.command()
def show(
    config_path: Path = typer.Argument(
        ...,
        help="Path to configuration file to display.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Display a configuration file with syntax highlighting.

    [bold]Example:[/bold]

        finetune config show configs/my-config.yaml
    """
    content = config_path.read_text()
    syntax = Syntax(content, "yaml", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title=str(config_path)))
