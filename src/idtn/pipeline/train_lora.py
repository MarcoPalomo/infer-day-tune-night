"""Entraînement LoRA interruptible : checkpoints réguliers, arrêt propre sur garde-fou."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from idtn import guards
from idtn.pipeline.common import torch_device
from idtn.pipeline.model import load_base

log = logging.getLogger(__name__)


class GuardCallback(TrainerCallback):
    def __init__(self, cfg: dict):
        self.cfg = cfg["guards"]
        self.last = 0.0
        self.tripped: list[str] = []

    def on_step_end(self, args, state, control, **kwargs):
        if time.monotonic() - self.last < self.cfg["poll_seconds"]:
            return
        self.last = time.monotonic()
        if problems := guards.violations(self.cfg):
            log.warning("garde-fou déclenché : %s, checkpoint et arrêt", problems)
            self.tripped = problems
            control.should_save = True
            control.should_training_stop = True


def run(cfg: dict, run_dir: Path) -> Path | None:
    """Entraîne et sauvegarde l'adapter dans run_dir/peft. None si interrompu."""
    t = cfg["training"]
    model, tok = load_base(cfg, for_training=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if t.get("quantization") == "4bit":
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=t["lora"]["r"],
            lora_alpha=t["lora"]["alpha"],
            lora_dropout=t["lora"]["dropout"],
            target_modules=t["lora"]["target_modules"],
            task_type="CAUSAL_LM",
        ),
    )

    data = load_dataset("json", data_files=str(Path(cfg["paths"]["datasets"]) / "train.jsonl"))
    data = data["train"].map(
        lambda ex: tok(
            tok.apply_chat_template(ex["messages"], tokenize=False),
            truncation=True,
            max_length=t["max_seq_len"],
        ),
        remove_columns=["messages"],
    )

    ckpt_dir = run_dir / "ckpt"
    on_gpu = torch_device(cfg) == "cuda"
    args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=t["epochs"],
        learning_rate=t["lr"],
        per_device_train_batch_size=4 if on_gpu else 1,
        gradient_accumulation_steps=4 if on_gpu else 16,
        save_steps=t["checkpoint_every_steps"],
        save_total_limit=2,
        logging_steps=10,
        bf16=on_gpu,
        use_cpu=not on_gpu,
        dataloader_num_workers=0,
        report_to=[],
    )
    guard = GuardCallback(cfg)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=data,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
        callbacks=[guard],
    )
    resume = get_last_checkpoint(str(ckpt_dir)) if ckpt_dir.exists() else None
    trainer.train(resume_from_checkpoint=resume)
    if guard.tripped:
        return None
    out = run_dir / "peft"
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return out
