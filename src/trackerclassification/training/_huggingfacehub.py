from __future__ import annotations
from typing import Dict, Any
from transformers import TrainerCallback, TrainingArguments

import logging
LOGGER = logging.getLogger(__name__)

from ..model import ModelBase


class PushCheckpointsToHubCallback(TrainerCallback):
    def __init__(
        self,
        repo_id: str,
        experiment: str,
        save_dirname: str = "checkpoints",
    ) -> None:
        self.repo_id = repo_id
        self.experiment = experiment
        self.save_dirname = save_dirname

    def on_save(
        self,
        args: TrainingArguments,
        state,
        control,
        **kwargs: Dict[str, Any],
    ):
        try: 
            model: ModelBase = kwargs["model"]
            step = state.global_step

            card_text = (
                f"# Experiment: {self.experiment}\n\n"
                f"## Checkpoint\n\n"
                f"- Step: `{step}`\n"
            )

            ckpt_name = f"step-{step}"

            LOGGER.info("Pushing checkpoint to HF Hub: %s (step %s)", self.experiment, step)
            model.push_to_hub(
                repo_id=self.repo_id,
                experiment=self.experiment,
                card_text=card_text,
                save_dirname=self.save_dirname,
                ckpt_name=ckpt_name,
            )
            return control
        except Exception as e:
            LOGGER.warning("An error occurred while pushing checkpoint to Hub: %s", e)
            return control