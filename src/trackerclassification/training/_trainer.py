from __future__ import annotations
from typing import Any, Dict, List
from torch.utils.data import Dataset
from transformers import Trainer as HFTrainer, TrainingArguments, TrainerCallback
import json
import wandb

import logging
LOGGER = logging.getLogger(__name__)

from ._huggingfacehub import PushCheckpointsToHubCallback 
from ..model import ModelBase
from ..dataset import TrackingDataset, PyGTrackingDataCollator
from ..config.args import HuggingfaceArgs, TrainingArgs
from ._metrics import TrackingMetrics


class Trainer:
    """
    Wrapper Class of a huggingface transformers Trainer, initializing the datasets and providing the logic to train a model. 

    Attributes:
        model (ModelBase): The model to be trained.
        train_dataset (TrackingDataset): The training dataset.
        eval_dataset (TrackingDataset): The evaluation dataset.
        experiment_id (str): The experiment ID for HuggingFace Hub.
        repo_id (str): The repository ID for HuggingFace Hub.
    """
    def __init__(
        self,
        model: ModelBase,
        train_dataset: TrackingDataset,
        eval_dataset: TrackingDataset,
        experiment_id: str,
        training_args: TrainingArgs,
        hf_args: HuggingfaceArgs,

        ) -> None:
        self._experiment_id = experiment_id
        self._train_dataset = train_dataset
        self._eval_dataset = eval_dataset
        self._hf_args = hf_args
        self._initialize_training_args(training_args)
        self._initialize_metrics()
        self._initialize_callbacks()
        self._initialize_trainer(model)

    def _initialize_training_args(self, training_args: TrainingArgs) -> None:
        self._args = TrainingArguments(
            output_dir=training_args.output_dir,
            per_device_train_batch_size=training_args.per_device_train_batch_size,
            per_device_eval_batch_size=training_args.per_device_eval_batch_size,
            learning_rate=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
            num_train_epochs=training_args.num_train_epochs,
            logging_steps=training_args.logging_steps,
            eval_strategy=training_args.eval_strategy,
            save_strategy=training_args.save_strategy,
            save_steps=training_args.save_steps,
            eval_steps=training_args.eval_steps,
            save_total_limit=training_args.save_total_limit,
            dataloader_num_workers=training_args.dataloader_num_workers,
            fp16=training_args.fp16,
            seed=training_args.seed,
            report_to=training_args.report_to,
            metric_for_best_model=training_args.metric_for_best_model,
            load_best_model_at_end=training_args.load_best_model_at_end,
            remove_unused_columns=False
        )

    def _initialize_metrics(self) -> None:
        metrics = TrackingMetrics(num_trackers=self._train_dataset._num_trackers, num_leds=7)
        self._compute_metrics = metrics.compute_metrics
        LOGGER.info("Tracking the following metrics: tracker_accuracy, led_accuracy, joint_accuracy")

    def _initialize_callbacks(self) -> None:
        self._callbacks: List[TrainerCallback] = []
        if self._hf_args.push_to_hub:
            push_callback = PushCheckpointsToHubCallback(
                repo_id=self._hf_args.repo_id,
                experiment=self._experiment_id,
                save_dirname="checkpoints",
            )
            self._callbacks.append(push_callback)
            LOGGER.info("Added PushCheckpointsToHubCallback callback for experiment: %s", self._experiment_id)

    def _initialize_trainer(self, model: ModelBase) -> None:
        self._trainer = HFTrainer(
            model=model,
            args=self._args,
            train_dataset=self._train_dataset,
            eval_dataset=self._eval_dataset,
            data_collator=PyGTrackingDataCollator(),
            compute_metrics=self._compute_metrics,
            callbacks=self._callbacks,
        )
        LOGGER.info("Initialized trainer: %s", self._trainer)

    def train(self) -> Dict[str, float]:
        """
        Run training.

        Returns:
            Dict[str, float]: Training metrics (as reported by HF Trainer).
        """
        try:
            LOGGER.info("Starting training for %s epochs", self._args.num_train_epochs)
            train_result = self._trainer.train()
            self._trainer.save_model()

            metrics = train_result.metrics or {}
            self._trainer.log_metrics("train", metrics)
            self._trainer.save_metrics("train", metrics)
            self._trainer.save_state()
        except Exception as e:
            LOGGER.error("An error occurred during training: %s", e)
            raise
        finally:
            LOGGER.info("Training complete.")
            if self._hf_args.push_to_hub:
                model: ModelBase = self._trainer.model 

                card_text = (
                    f"# Experiment: {self._experiment_id}\n\n"
                    f"## Summary\n\n"
                    f"- Model: `{model.__class__.__name__}`\n"
                    f"## TrainingArguments\n\n"
                    "```json\n"
                    f"{json.dumps(self._args.to_dict(), indent=2)}\n"
                    f"## WandB Run: https://wandb.ai/yannikheizmann-hochschule-offenburg/trackerclassification \n\n"
                    f"- Run ID: `{wandb.run.id}`\n"
                    f"- Run Name: `{wandb.run.name}`\n"
                    f"- Group: `{wandb.run.group}`\n"
                    "```\n"
                )

                files = {
                    "training_args.json": json.dumps(self._args.to_dict(), indent=2),
                }

                ckpt_name = "best" if self._args.load_best_model_at_end else "last"

                model.push_to_hub(
                    repo_id=self._hf_args.repo_id,
                    experiment=self._experiment_id,
                    card_text=card_text,
                    save_dirname=".",
                    files=files,
                    ckpt_name=ckpt_name,
                )

                LOGGER.info("Pushed experiment to HF Hub: %s", self._experiment_id)
        

    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation on the provided ``eval_dataset`` (if any).

        Returns:
            Dict[str, float]: Evaluation metrics (possibly empty if no eval set).
        """
        metrics = self._trainer.evaluate()
        self._trainer.log_metrics("eval", metrics)
        self._trainer.save_metrics("eval", metrics)
        return metrics

    def predict(self, test_dataset: Dataset) -> Any:
        """Run prediction on a test dataset (delegates to HF Trainer)."""
        return self._trainer.predict(test_dataset)

    @property
    def output_dir(self) -> str:
        """Directory where checkpoints and artifacts are written."""
        return self._args.output_dir