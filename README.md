# local-ai-interface

Docker tools to build a local AI interface using Open Web UI, ComfyUI, and Ollama, with integrated LLM finetuning capabilities.

## Features

- **Open WebUI**: Chat interface for interacting with LLMs
- **Ollama**: Local LLM inference with ROCm (AMD GPU) support
- **ComfyUI**: Image generation with Stable Diffusion
- **LLM Finetuning**: CLI tool for finetuning models with Unsloth + TRL

## Quick Start

### Running the AI Interface

```bash
# Start all services
docker compose up -d

# Access Open WebUI at http://localhost:3000
```

### LLM Finetuning

The finetuning CLI allows you to train custom models and deploy them to Ollama.

#### Installation (Host Development)

**Important:** This project requires AMD GPUs with ROCm. Follow these steps in order.

```bash
# 1. Create a fresh virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install PyTorch ROCm stack (from official Unsloth guide)
pip install -r requirements-rocm.txt

# 3. Install Unsloth (handles transformers/trl/accelerate versions automatically)
pip install unsloth

# 4. Install this package
pip install -e ".[train]"

# 5. (Optional) Install dev dependencies for contributing
pip install -e ".[dev]"
pre-commit install
```

**Note for consumer AMD GPUs (RX 7000 series):** If you get xformers errors, edit `requirements-rocm.txt` and comment out the `xformers` line, then reinstall.

Verify GPU detection:
```bash
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not found')"
```

#### Usage

```bash
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

# List checkpoints
finetune list checkpoints my-run
```

#### Example Workflow

1. **Create a configuration file**:
   ```bash
   finetune config generate -o configs/my-llama-finetune.yaml
   ```

2. **Edit the configuration** to specify your dataset and hyperparameters

3. **Start training**:
   ```bash
   finetune train --config configs/my-llama-finetune.yaml
   ```

4. **Convert and deploy to Ollama**:
   ```bash
   finetune convert checkpoints/my-run/final --name my-custom-llama
   ollama run my-custom-llama
   ```

## Project Structure

```
local-ai-interface/
├── docker-compose.yaml      # Main services (Ollama, ComfyUI, Open WebUI)
├── pyproject.toml           # Python package configuration
├── finetune/                # Finetuning CLI package
│   ├── cli/                 # CLI commands
│   ├── core/                # Training, conversion, checkpoint logic
│   ├── config/              # Configuration schema and loading
│   └── utils/               # Utilities
├── configs/                 # Training configuration templates
│   ├── base.yaml            # Base config with defaults
│   └── examples/            # Example configurations
├── checkpoints/             # Training checkpoints (gitignored)
├── exports/                 # Exported models for Ollama (gitignored)
├── data/                    # Training data (gitignored)
├── docker/                  # Docker files for finetuning
└── tests/                   # Test suite
```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
# Service ports
OLLAMA_PORT=11434
COMFYUI_PORT=8188
OPENWEBUI_PORT=3000

# For gated models (optional)
HF_TOKEN=your_token_here
```

### Training Configuration

See `configs/base.yaml` for all available options. Key sections:

- **model**: Base model, sequence length, quantization
- **lora**: LoRA rank, alpha, target modules
- **training**: Epochs, batch size, learning rate
- **data**: Dataset, prompt template
- **output**: Run name, output directory

## Docker Services

| Service | Description | Port |
|---------|-------------|------|
| ollama | LLM inference (ROCm) | 11434 |
| open-webui | Chat interface | 3000 |
| comfyui | Image generation | 8188 |
| finetune | Finetuning (optional) | - |

To run finetuning in Docker:

```bash
docker compose --profile finetune run finetune train --config configs/my-config.yaml
```

## Development

### Running Tests

```bash
pytest --cov=finetune --cov-report=term-missing
```

### Pre-commit Hooks

```bash
# Run all checks
pre-commit run --all-files

# Hooks run automatically on git commit
```

### Code Quality Tools

- **black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest-cov**: Test coverage (70% minimum)

## Requirements

- Python 3.11+
- Docker & Docker Compose
- AMD GPU with ROCm support (for training)

## License

MIT
