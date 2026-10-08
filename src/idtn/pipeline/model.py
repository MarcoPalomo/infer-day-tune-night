"""Chargement du modèle de base (+ adapters) pour l'entraînement, le jugement et l'évaluation."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from idtn.pipeline.common import torch_device


def base_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["models"]) / "base-hf"


def current_adapter(cfg: dict) -> Path | None:
    peft = Path(cfg["paths"]["adapters"]) / "current" / "peft"
    return peft if (peft / "adapter_config.json").exists() else None


def load_base(cfg: dict, for_training: bool = False):
    torch.set_num_threads(cfg["training"]["max_threads"])
    device = torch_device(cfg)
    kwargs: dict = {"torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32}
    if for_training and cfg["training"].get("quantization") == "4bit":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
        kwargs["device_map"] = {"": 0}
    tok = AutoTokenizer.from_pretrained(base_path(cfg))
    model = AutoModelForCausalLM.from_pretrained(base_path(cfg), **kwargs)
    if "device_map" not in kwargs:
        model.to(device)
    return model, tok
