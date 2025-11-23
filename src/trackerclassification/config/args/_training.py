from __future__ import annotations
from pydantic import Field

from ...utils.argparsing import AdditionalArgsBase
    

class TrainingArgs(AdditionalArgsBase):
    output_dir: str = Field(
        description="Directory to save training results and checkpoints",
        default="./results"
    )
    per_device_train_batch_size: int = Field(
        description="Batch size per device during training",
        default=32
    )
    per_device_eval_batch_size: int = Field(
        description="Batch size per device during evaluation",
        default=64
    )
    learning_rate: float = Field(
        description="Learning rate for the optimizer",
        default=0.001
    )
    weight_decay: float = Field(
        description="Weight decay for the optimizer",
        default=0.0
    )
    num_train_epochs: int = Field(
        description="Total number of training epochs to perform",
        default=100
    )
    logging_steps: int = Field(
        description="Number of steps between logging training metrics",
        default=100
    )
    eval_strategy: str = Field(
        description="Evaluation strategy to use",
        default="steps"
    )
    save_strategy: str = Field(
        description="Model saving strategy to use",
        default="steps"
    )
    save_steps: int = Field(
        description="Number of steps between model saves",
        default=5000
    )
    eval_steps: int = Field(
        description="Number of steps between evaluations",
        default=5000
    )
    save_total_limit: int = Field(
        description="Maximum number of saved checkpoints to keep",
        default=2
    )
    dataloader_num_workers: int = Field(
        description="Number of subprocesses to use for data loading",
        default=4
    )
    fp16: bool = Field(
        description="Whether to use mixed precision training (FP16)",
        default=False
    )
    seed: int = Field(
        description="Random seed for initialization",
        default=42
    )
    report_to: list[str] = Field(
        description="List of integrations to report training metrics to",
        default=["wandb"]
    )
    metric_for_best_model: str = Field(
        description="Metric to use to evaluate the best model",
        default="loss"
    )
    load_best_model_at_end: bool = Field(
        description="Whether to load the best model found during training at the end",
        default=True
    )