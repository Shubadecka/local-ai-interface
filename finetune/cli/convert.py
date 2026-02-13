"""Convert command for exporting models to Ollama."""

from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.panel import Panel

console = Console()

# Available quantization methods
QUANTIZATION_METHODS = [
    "q4_k_m",  # Recommended: good balance of quality and size
    "q5_k_m",  # Higher quality, larger size
    "q8_0",  # Highest quality, largest size
    "q4_0",  # Fastest, lowest quality
    "q4_1",
    "q5_0",
    "q5_1",
    "f16",  # No quantization (half precision)
]


def convert(
    checkpoint: Path = typer.Argument(
        ...,
        help="Path to checkpoint directory to convert.",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    name: str = typer.Option(
        ...,
        "--name",
        "-n",
        help="Name for the Ollama model.",
    ),
    quantize: str = typer.Option(
        "q4_k_m",
        "--quantize",
        "-q",
        help=f"Quantization method. Options: {', '.join(QUANTIZATION_METHODS)}",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for GGUF file. Defaults to ./exports/<name>/",
    ),
    push_to_ollama: bool = typer.Option(
        False,
        "--push/--no-push",
        help="Register the model with Ollama after conversion.",
    ),
    ollama_host: str = typer.Option(
        "http://localhost:11434",
        "--ollama-host",
        envvar="OLLAMA_HOST",
        help="Ollama server URL.",
    ),
    merge_lora: bool = typer.Option(
        True,
        "--merge/--no-merge",
        help="Merge LoRA weights into base model before conversion.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging.",
    ),
) -> None:
    """Convert a finetuned checkpoint to Ollama format.

    This command:
    1. Loads the finetuned model checkpoint
    2. Optionally merges LoRA adapters into the base model
    3. Converts to GGUF format with specified quantization
    4. Generates an Ollama Modelfile
    5. Registers the model with Ollama (if --push)

    [bold]Examples:[/bold]

        # Basic conversion
        finetune convert checkpoints/my-run/final --name my-model

        # With specific quantization
        finetune convert checkpoints/my-run/final --name my-model -q q5_k_m

        # Without pushing to Ollama
        finetune convert checkpoints/my-run/final --name my-model --no-push

        # Custom output directory
        finetune convert checkpoints/my-run/final --name my-model -o ./my-exports/
    """
    import logging

    from finetune.utils.logging import setup_logging

    setup_logging(level=logging.DEBUG if verbose else logging.INFO)

    # Validate quantization method
    if quantize not in QUANTIZATION_METHODS:
        console.print(
            f"[red]Error:[/red] Invalid quantization method '{quantize}'. "
            f"Valid options: {', '.join(QUANTIZATION_METHODS)}"
        )
        raise typer.Exit(1)

    # Validate model name
    if not name.replace("-", "").replace("_", "").isalnum():
        console.print(
            "[red]Error:[/red] Model name must be alphanumeric "
            "(hyphens and underscores allowed)."
        )
        raise typer.Exit(1)

    # Set default output directory
    if output_dir is None:
        from finetune.utils.paths import get_exports_dir

        output_dir = get_exports_dir() / name

    # Display conversion plan
    _display_conversion_plan(
        checkpoint=checkpoint,
        name=name,
        quantize=quantize,
        output_dir=output_dir,
        push_to_ollama=push_to_ollama,
        ollama_host=ollama_host,
        merge_lora=merge_lora,
    )

    try:
        from finetune.core.converter import convert_to_ollama

        result = convert_to_ollama(
            checkpoint_path=checkpoint,
            model_name=name,
            quantization=quantize,
            output_dir=output_dir,
            merge_lora=merge_lora,
            push_to_ollama=push_to_ollama,
            ollama_host=ollama_host,
        )

        _display_conversion_result(result)

    except Exception as e:
        console.print(f"[red]Conversion failed:[/red] {e}")
        raise typer.Exit(1)


def _display_conversion_plan(
    checkpoint: Path,
    name: str,
    quantize: str,
    output_dir: Path,
    push_to_ollama: bool,
    ollama_host: str,
    merge_lora: bool,
) -> None:
    """Display the conversion plan."""
    from rich.table import Table

    table = Table(title="Conversion Plan", show_header=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Checkpoint", str(checkpoint))
    table.add_row("Model Name", name)
    table.add_row("Quantization", quantize)
    table.add_row("Output Directory", str(output_dir))
    table.add_row("Merge LoRA", "Yes" if merge_lora else "No")
    table.add_row("Push to Ollama", "Yes" if push_to_ollama else "No")
    if push_to_ollama:
        table.add_row("Ollama Host", ollama_host)

    console.print(table)
    console.print()


def _display_conversion_result(result: dict[str, Any]) -> None:
    """Display the conversion result."""
    console.print(
        Panel(
            f"[green]Conversion successful![/green]\n\n"
            f"GGUF file: {result.get('gguf_path', 'N/A')}\n"
            f"Modelfile: {result.get('modelfile_path', 'N/A')}\n"
            f"Ollama model: {result.get('ollama_model', 'N/A')}",
            title="Conversion Complete",
        )
    )

    if result.get("ollama_model"):
        console.print(
            f"\n[bold]Run your model:[/bold]\n" f"  ollama run {result['ollama_model']}"
        )
