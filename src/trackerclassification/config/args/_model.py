from __future__ import annotations
from pydantic import Field

from ...utils.argparsing import AdditionalArgsBase


class ModelArgs(AdditionalArgsBase):
    in_dim: int = Field(
        description="Input feature dimension per node", 
        default=2)
    dropout: float = Field(
        description="Dropout rate for GNN layers", 
        default=0.2)
    hidden_dims: list[int] = Field(
        description="List of hidden dimensions for each DGCNN layer", 
        default=[64, 128, 256, 512, 1024, 1024, 512])