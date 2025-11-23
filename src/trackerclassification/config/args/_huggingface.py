from __future__ import annotations
from pydantic import Field

from ...utils.argparsing import AdditionalArgsBase


class HuggingfaceArgs(AdditionalArgsBase):
    push_to_hub: bool = Field(
        description="Whether to push the model and checkpoints to HuggingFace Hub", 
        default=True)
    repo_id: str = Field(
        description="Repository ID for HuggingFace Hub", 
        default="yannikheizmann/trackerclassification")