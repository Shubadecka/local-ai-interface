#!/bin/bash
# Entrypoint script for the finetuning container

set -e

# Print environment info
echo "=== LLM Finetune Container ==="
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "ROCm available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Check for AMD GPU
if python -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    echo "VRAM: $(python -c 'import torch; print(f\"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\")')"
else
    echo "WARNING: No GPU detected. Training will be slow!"
fi

echo "================================"

# Execute the command
exec "$@"
