from __future__ import annotations
from pydantic import Field

from ...utils.argparsing import AdditionalArgsBase


class ModelArgs(AdditionalArgsBase):
    in_dim: int = Field(
        description="Input feature dimension per node", 
        default=2)
    out_channels: int = Field(
        description="Number of output channels/features per node", 
        default=64)
    k: int = Field(
        description="Number of nearest neighbours for graph construction", 
        default=3)