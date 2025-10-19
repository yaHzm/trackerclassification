from pydantic import Field

from ..utils.argparsing import PydanticArgsBase
from ._config import (
    TRAIN_SIZE,
    VALIDATION_SIZE,
    TEST_SIZE,
    NUM_TRACKERS
)


class Args(PydanticArgsBase):
    train_size: int = Field(
        description="Number of samples in the training dataset", 
        default=TRAIN_SIZE)
    validation_size: int = Field(
        description="Number of samples in the validation dataset", 
        default=VALIDATION_SIZE)
    test_size: int = Field(
        description="Number of samples in the test dataset", 
        default=TEST_SIZE)
    num_trackers: int = Field(
        description="Number of trackers in each sample", 
        default=NUM_TRACKERS)