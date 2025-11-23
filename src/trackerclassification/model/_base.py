from __future__ import annotations
from abc import ABC
from typing import Tuple, Iterable, Optional
import json, tempfile, os

import torch
import torch.nn as nn
from huggingface_hub import upload_folder

from ..utils.registry import RegistryMeta
from ..utils import repo_exists
from ..config.secrets import HF_TOKEN


class ModelBase(nn.Module, ABC, metaclass=RegistryMeta["ModelBase"]):
    """Common I/O utilities shared by all models (incl. PyG models)."""

    def __init__(self) -> None:
        super().__init__()

    def save_local(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_local(self, path: str, map_location: str | torch.device = "cpu") -> None:
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state)

    @torch.no_grad()
    def export_onnx(
        self,
        path: str,
        example_inputs: Tuple[torch.Tensor, ...],
        dynamic_axes: Optional[dict] = None,
        opset_version: int = 17,
        output_names: Iterable[str] = ("tracker_logits", "led_logits"),
    ) -> None:
        self.eval()
        torch.onnx.export(
            self, example_inputs, path,
            opset_version=opset_version,
            input_names=[f"input_{i}" for i in range(len(example_inputs))],
            output_names=list(output_names),
            dynamic_axes=dynamic_axes or {},
        )

    def push_to_hub(
        self,
        repo_id: str,
        experiment: str,
        card_text: str,
        save_dirname: str = ".",
        files: dict | None = None,
        ckpt_name: str = "last",
    ) -> None:
        repo_exists(repo_id)
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "experiments", experiment)

            if save_dirname == ".":
                ckpt_path = os.path.join(root, f"{ckpt_name}.pt")
            else:
                ckpt_dir = os.path.join(root, save_dirname)
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f"{ckpt_name}.pt")

            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(self.state_dict(), ckpt_path)

            config = {
                "architecture": self.__class__.__name__,
                "num_parameters": sum(p.numel() for p in self.parameters()),
            }
            with open(os.path.join(root, "config.json"), "w") as f:
                json.dump(config, f, indent=2)

            with open(os.path.join(root, "README.md"), "w") as f:
                f.write(card_text)

            if files:
                for rel, content in files.items():
                    abspath = os.path.join(root, rel)
                    os.makedirs(os.path.dirname(abspath), exist_ok=True)
                    mode = "wb" if isinstance(content, (bytes, bytearray)) else "w"
                    with open(abspath, mode) as f:
                        f.write(content)

            upload_folder(
                repo_id=repo_id,
                folder_path=tmp,
                path_in_repo=".",
                token=HF_TOKEN,
            )