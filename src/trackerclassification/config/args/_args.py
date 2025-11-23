from __future__ import annotations
from pydantic import Field

from ...utils.argparsing import PydanticArgsBase
from ..components import ModelOptions, TrackerOptions
from ._huggingface import HuggingfaceArgs
from ._model import ModelArgs
from ._training import TrainingArgs


class Args(PydanticArgsBase):
    train_size: int = Field(
        description="Number of samples in the training dataset", 
        default=50000)
    eval_size: int = Field(
        description="Number of samples in the evaluation dataset", 
        default=2000)
    num_trackers: int = Field(
        description="Number of trackers in each sample", 
        default=3)
    tracker: TrackerOptions = Field(
        description="Tracker type to use", 
        default=TrackerOptions.V4)
    num_leds: int = Field(
        description="Number of LEDs per tracker", 
        default=7)
    model: ModelOptions = Field(
        description="Model architecture to use", 
        default=ModelOptions.DGCNN)
    model_args: ModelArgs = Field(
        description="Additional arguments for the model", 
        default_factory=ModelArgs)
    training_args: TrainingArgs = Field(
        description="Additional arguments for training", 
        default_factory=TrainingArgs)
    huggingface_args: HuggingfaceArgs = Field(
        description="Additional arguments for HuggingFace integration", 
        default_factory=HuggingfaceArgs)
