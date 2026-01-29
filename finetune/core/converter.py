"""Model conversion to GGUF format for Ollama."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from finetune.utils.logging import get_logger
from finetune.utils.paths import ensure_dir

logger = get_logger("converter")


class ConversionError(Exception):
    """Error during model conversion."""

    pass


# Llama 3 chat template
LLAMA3_CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ if .System }}{{ .System }}{{ else }}You are a helpful assistant.{{ end }}<|eot_id|><|start_header_id|>user<|end_header_id|}

{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

# Default system prompt
DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant."


def convert_to_ollama(
    checkpoint_path: Path,
    model_name: str,
    quantization: str = "q4_k_m",
    output_dir: Optional[Path] = None,
    merge_lora: bool = True,
    push_to_ollama: bool = True,
    ollama_host: str = "http://localhost:11434",
) -> dict[str, Any]:
    """Convert a finetuned checkpoint to Ollama format.

    Args:
        checkpoint_path: Path to the checkpoint directory
        model_name: Name for the Ollama model
        quantization: Quantization method (e.g., q4_k_m, q5_k_m, q8_0)
        output_dir: Output directory for GGUF and Modelfile
        merge_lora: Whether to merge LoRA weights into base model
        push_to_ollama: Whether to register with Ollama after conversion
        ollama_host: Ollama server URL

    Returns:
        Dictionary with conversion results

    Raises:
        ConversionError: If conversion fails
    """
    checkpoint_path = Path(checkpoint_path)

    if output_dir is None:
        from finetune.utils.paths import get_exports_dir

        output_dir = get_exports_dir() / model_name

    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    logger.info(f"Converting checkpoint: {checkpoint_path}")
    logger.info(f"Output directory: {output_dir}")

    # Load the model and convert
    gguf_path = _convert_to_gguf(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        model_name=model_name,
        quantization=quantization,
        merge_lora=merge_lora,
    )

    # Generate Modelfile
    modelfile_path = _generate_modelfile(
        output_dir=output_dir,
        gguf_filename=gguf_path.name,
        model_name=model_name,
    )

    # Save conversion metadata
    _save_conversion_metadata(
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        quantization=quantization,
        gguf_path=gguf_path,
        modelfile_path=modelfile_path,
    )

    result: dict[str, Any] = {
        "gguf_path": str(gguf_path),
        "modelfile_path": str(modelfile_path),
        "metadata_path": str(output_dir / "metadata.json"),
    }

    # Register with Ollama if requested
    if push_to_ollama:
        from finetune.core.ollama import OllamaClient

        client = OllamaClient(ollama_host)

        if client.check_connection():
            logger.info(f"Registering model with Ollama: {model_name}")
            try:
                client.create_model(model_name, modelfile_path)
                result["ollama_model"] = model_name
                logger.info(f"Model registered successfully: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to register with Ollama: {e}")
                result["ollama_error"] = str(e)
        else:
            logger.warning(f"Could not connect to Ollama at {ollama_host}")
            result["ollama_error"] = f"Could not connect to {ollama_host}"

    return result


def _convert_to_gguf(
    checkpoint_path: Path,
    output_dir: Path,
    model_name: str,
    quantization: str,
    merge_lora: bool,
) -> Path:
    """Convert model to GGUF format using Unsloth.

    Args:
        checkpoint_path: Path to checkpoint
        output_dir: Output directory
        model_name: Model name
        quantization: Quantization method
        merge_lora: Whether to merge LoRA

    Returns:
        Path to GGUF file
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError as e:
        raise ConversionError(
            "Unsloth not installed. Install with: uv pip install llm-finetune[train]"
        ) from e

    # Find the run config to get base model info
    run_dir = checkpoint_path.parent
    if checkpoint_path.name in ("final",) or checkpoint_path.name.startswith(
        "checkpoint-"
    ):
        run_dir = checkpoint_path.parent

    config_path = run_dir / "run_config.yaml"
    if config_path.exists():
        import yaml

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        base_model = config.get("model", {}).get("name")
        max_seq_length = config.get("model", {}).get("max_seq_length", 2048)
    else:
        # Try to infer from adapter config
        adapter_config_path = checkpoint_path / "adapter_config.json"
        if adapter_config_path.exists():
            with open(adapter_config_path, "r") as f:
                adapter_config = json.load(f)
            base_model = adapter_config.get("base_model_name_or_path")
            max_seq_length = 2048
        else:
            raise ConversionError(
                "Could not determine base model. "
                "No run_config.yaml or adapter_config.json found."
            )

    logger.info(f"Base model: {base_model}")
    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    # Load the finetuned model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(checkpoint_path),
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    # Determine output filename
    gguf_filename = f"{model_name}-{quantization}.gguf"
    gguf_path = output_dir / gguf_filename

    logger.info(f"Converting to GGUF with quantization: {quantization}")

    # Use Unsloth's save_pretrained_gguf
    # This handles merging and conversion in one step
    model.save_pretrained_gguf(
        str(output_dir),
        tokenizer,
        quantization_method=quantization,
    )

    # Unsloth saves with a specific naming convention, rename if needed
    # Find the generated GGUF file
    generated_files = list(output_dir.glob("*.gguf"))
    if generated_files:
        source_gguf = generated_files[0]
        if source_gguf != gguf_path:
            shutil.move(str(source_gguf), str(gguf_path))

    if not gguf_path.exists():
        raise ConversionError(f"GGUF conversion failed - no output file at {gguf_path}")

    logger.info(f"GGUF file created: {gguf_path}")
    return gguf_path


def _generate_modelfile(
    output_dir: Path,
    gguf_filename: str,
    model_name: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    context_length: int = 4096,
) -> Path:
    """Generate an Ollama Modelfile.

    Args:
        output_dir: Output directory
        gguf_filename: Name of the GGUF file
        model_name: Model name
        system_prompt: Optional system prompt
        temperature: Default temperature
        context_length: Context length

    Returns:
        Path to Modelfile
    """
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    modelfile_content = f"""# Modelfile for {model_name}
# Generated by llm-finetune

FROM ./{gguf_filename}

# Llama 3 chat template
TEMPLATE \"\"\"{LLAMA3_CHAT_TEMPLATE}\"\"\"

# Default parameters
PARAMETER temperature {temperature}
PARAMETER num_ctx {context_length}
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"

# System prompt
SYSTEM \"\"\"{system_prompt}\"\"\"
"""

    modelfile_path = output_dir / "Modelfile"
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    logger.info(f"Modelfile created: {modelfile_path}")
    return modelfile_path


def _save_conversion_metadata(
    output_dir: Path,
    checkpoint_path: Path,
    model_name: str,
    quantization: str,
    gguf_path: Path,
    modelfile_path: Path,
) -> dict[str, Any]:
    """Save metadata about the conversion.

    Args:
        output_dir: Output directory
        checkpoint_path: Source checkpoint path
        model_name: Model name
        quantization: Quantization method
        gguf_path: Path to GGUF file
        modelfile_path: Path to Modelfile

    Returns:
        Metadata dictionary
    """
    # Get GGUF file size
    gguf_size = gguf_path.stat().st_size if gguf_path.exists() else 0

    metadata = {
        "model_name": model_name,
        "source_checkpoint": str(checkpoint_path),
        "quantization": quantization,
        "gguf_file": gguf_path.name,
        "gguf_size_bytes": gguf_size,
        "gguf_size_human": _format_size(gguf_size),
        "modelfile": modelfile_path.name,
        "converted_at": datetime.now().isoformat(),
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def _format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    size: float = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"
