from __future__ import annotations

from .dataset import TrackingDataset
from .dataset.sample import SampleVisualizer
from .utils.argparsing import ArgsParser
from .config import Args


class Main:
    def __init__(self) -> None:
        self._args: Args = ArgsParser(Args).parse()

    @classmethod
    def visualize_sample(cls) -> None:
        SampleVisualizer.main()

    @classmethod
    def train(cls) -> None:
        args: Args = ArgsParser(Args).parse()
        train_dataset = TrackingDataset(
            size=args.train_size,
            num_trackers=args.num_trackers,          
        )
        print(train_dataset[0])