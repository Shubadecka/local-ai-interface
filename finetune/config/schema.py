"""Pydantic models for finetuning configuration."""

from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    """Configuration for the base model."""

    name: str = Field(
        description="HuggingFace model name or path",
        examples=["unsloth/Meta-Llama-3.1-8B-bnb-4bit"],
    )
    max_seq_length: int = Field(
        default=2048,
        ge=128,
        le=131072,
        description="Maximum sequence length for training",
    )
    load_in_4bit: bool = Field(
        default=True,
        description="Whether to load the model in 4-bit quantization",
    )
    load_in_8bit: bool = Field(
        default=False,
        description="Whether to load the model in 8-bit quantization",
    )
    dtype: Optional[Literal["float16", "bfloat16", "float32"]] = Field(
        default=None,
        description="Data type for model weights (None for auto-detection)",
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Whether to trust remote code from HuggingFace",
    )

    @field_validator("name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate that model name is not empty."""
        if not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()


class LoRAConfig(BaseModel):
    """Configuration for LoRA adapters."""

    r: int = Field(
        default=16,
        ge=1,
        le=256,
        description="LoRA rank (higher = more parameters, more capacity)",
    )
    lora_alpha: int = Field(
        default=16,
        ge=1,
        description="LoRA alpha scaling factor",
    )
    lora_dropout: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Dropout probability for LoRA layers",
    )
    target_modules: list[str] = Field(
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        description="Model modules to apply LoRA to",
    )
    bias: Literal["none", "all", "lora_only"] = Field(
        default="none",
        description="Which biases to train",
    )
    use_gradient_checkpointing: bool = Field(
        default=True,
        description="Use gradient checkpointing to save memory",
    )
    random_state: int = Field(
        default=3407,
        description="Random seed for LoRA initialization",
    )
    use_rslora: bool = Field(
        default=False,
        description="Use Rank-Stabilized LoRA",
    )
    loftq_config: Optional[dict[str, Any]] = Field(
        default=None,
        description="LoftQ quantization config",
    )


class TrainingConfig(BaseModel):
    """Configuration for training parameters."""

    num_epochs: int = Field(
        default=1,
        ge=1,
        description="Number of training epochs",
    )
    batch_size: int = Field(
        default=2,
        ge=1,
        description="Per-device training batch size",
    )
    gradient_accumulation_steps: int = Field(
        default=4,
        ge=1,
        description="Number of gradient accumulation steps",
    )
    learning_rate: float = Field(
        default=2e-4,
        gt=0,
        description="Learning rate",
    )
    weight_decay: float = Field(
        default=0.01,
        ge=0.0,
        description="Weight decay for regularization",
    )
    warmup_steps: int = Field(
        default=5,
        ge=0,
        description="Number of warmup steps",
    )
    warmup_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Warmup ratio (alternative to warmup_steps)",
    )
    max_steps: int = Field(
        default=-1,
        description="Max training steps (-1 for epoch-based)",
    )
    logging_steps: int = Field(
        default=1,
        ge=1,
        description="Log every N steps",
    )
    save_steps: int = Field(
        default=100,
        ge=1,
        description="Save checkpoint every N steps",
    )
    save_total_limit: Optional[int] = Field(
        default=3,
        ge=1,
        description="Maximum number of checkpoints to keep",
    )
    lr_scheduler_type: str = Field(
        default="linear",
        description="Learning rate scheduler type",
    )
    optim: str = Field(
        default="adamw_8bit",
        description="Optimizer to use",
    )
    fp16: bool = Field(
        default=False,
        description="Use FP16 mixed precision",
    )
    bf16: bool = Field(
        default=False,
        description="Use BF16 mixed precision",
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )
    gradient_checkpointing: bool = Field(
        default=True,
        description="Enable gradient checkpointing",
    )
    max_grad_norm: float = Field(
        default=0.3,
        ge=0.0,
        description="Maximum gradient norm for clipping",
    )


class DataConfig(BaseModel):
    """Configuration for training data."""

    dataset: str = Field(
        description="HuggingFace dataset name or local path",
        examples=["yahma/alpaca-cleaned", "./data/my_dataset.json"],
    )
    split: str = Field(
        default="train",
        description="Dataset split to use",
    )
    max_samples: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of samples to use (None for all)",
    )
    text_field: str = Field(
        default="text",
        description="Field name containing the text data",
    )
    prompt_template: Optional[str] = Field(
        default=None,
        description="Prompt template with {instruction}, {input}, {output} placeholders",
    )
    packing: bool = Field(
        default=False,
        description="Pack multiple sequences into one for efficiency",
    )
    test_size: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="Fraction of data to use for validation",
    )


class OutputConfig(BaseModel):
    """Configuration for output and checkpoints."""

    run_name: str = Field(
        description="Name for this training run",
        examples=["llama3-alpaca-lora"],
    )
    output_dir: Path = Field(
        default=Path("./checkpoints"),
        description="Base directory for saving checkpoints",
    )
    hub_model_id: Optional[str] = Field(
        default=None,
        description="HuggingFace Hub model ID for pushing",
    )
    push_to_hub: bool = Field(
        default=False,
        description="Whether to push model to HuggingFace Hub",
    )

    @field_validator("run_name")
    @classmethod
    def validate_run_name(cls, v: str) -> str:
        """Validate run name format."""
        if not v.strip():
            raise ValueError("Run name cannot be empty")
        # Replace spaces with hyphens
        return v.strip().replace(" ", "-")


class FinetuneConfig(BaseModel):
    """Complete finetuning configuration."""

    model: ModelConfig
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig
    output: OutputConfig

    def get_output_path(self) -> Path:
        """Get the full output path for this run."""
        return self.output.output_dir / self.output.run_name

    model_config = {"extra": "forbid"}
