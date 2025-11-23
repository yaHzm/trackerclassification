from __future__ import annotations
from typing import Type, Literal
import wandb
import time
from huggingface_hub import HfApi, create_repo

import logging
LOGGER = logging.getLogger(__name__)

from .dataset import TrackingDataset
from .dataset.tracker import TrackerBase
from .model import ModelBase
from .training import Trainer
from .utils.argparsing import ArgsParser
from .utils.registry import Registry
from .utils.logging import setup_logging
from .config.args import Args
from .config.secrets import HF_TOKEN


class Main:
    @classmethod
    def run(cls, task: Literal["train", "visualize"]) -> None:
        args: Args = ArgsParser(Args).parse()
        cls._startup(args)
        try:
            match task:
                case "train":
                    cls._train(args)
                case "visualize":
                    cls._visualize()
        except Exception as e:
            LOGGER.error("An error occurred: %s", e)
            raise
        finally:
            cls._shutdown(args)

    @classmethod
    def _startup(cls, args: Args) -> None:
        setup_logging()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_name = (
            f"{args.model.value}_k{args.model_args.k}"
            f"_{timestamp}"
        )
        wandb.init(
            project="trackerclassification",
            group=f"{args.model.value}_k{args.model_args.k}",
            name=run_name,
        )
        if args.huggingface_args.push_to_hub:
            api = HfApi(token=HF_TOKEN)
            create_repo(args.huggingface_args.repo_id, exist_ok=True, token=HF_TOKEN)
        LOGGER.info("Logging is set up.")

    @classmethod
    def _shutdown(cls, args: Args) -> None:
        wandb.config.update({"args": args.model_dump()}, allow_val_change=True)
        wandb.finish()
        LOGGER.info("Shutdown complete.")

    @classmethod
    def _visualize(cls) -> None:
        TrackingDataset.visualize()

    @classmethod
    def _train(cls, args: Args) -> None:
        LOGGER.info("Started training script with args: %s", args)

        TrackerClass: Type[TrackerBase] = Registry.get("TrackerBase", str(args.tracker))
        LOGGER.info("Using tracker type: %s", TrackerClass)

        train_dataset = TrackingDataset(
            size=args.train_size,
            num_trackers=args.num_trackers,
            TrackerClass=TrackerClass,
            seed=0)
        eval_dataset = TrackingDataset(
            size=args.eval_size,
            num_trackers=args.num_trackers,
            TrackerClass=TrackerClass,
            seed=1)
        LOGGER.info("Initialized training and evaluation datasets.")

        run = wandb.run
        group = run.group or "no-group"
        run_name = run.name or run.id
        experiment_id = f"{group}/{run_name}"

        LOGGER.info("Experiment ID for HF Hub: %s", experiment_id)

        num_unique_ids = TrackerClass.num_unique_ids()

        ModelClass: Type[ModelBase] = Registry.get("ModelBase", str(args.model))
        model: ModelBase = args.call(ModelClass, num_unique_ids=num_unique_ids)
        LOGGER.info("Initialized model: %s", model)

        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            experiment_id=experiment_id,
            training_args=args.training_args,
            hf_args=args.huggingface_args
        )
    
        trainer.train()
        LOGGER.info("Training completed.")

    @classmethod
    def train(cls) -> None:
        # tmux new-session -d -s training 'uv run train'
        # tmux ls
        # tmux attach -t training
        # ctrl + b -> d 
        # tmux kill-session -t training
        cls.run("train")

    @classmethod
    def visualize(cls) -> None:
        cls.run("visualize")