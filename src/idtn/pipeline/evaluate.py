"""Porte d'évaluation : candidat vs adapter courant (ou base) sur holdout et golden set."""

from __future__ import annotations

import math
from contextlib import nullcontext
from pathlib import Path

import torch
from peft import PeftModel

from idtn.pipeline.common import read_jsonl
from idtn.pipeline.model import current_adapter, load_base


@torch.no_grad()
def perplexity(model, tok, rows: list[dict], max_len: int) -> float:
    total, count = 0.0, 0
    for row in rows:
        text = tok.apply_chat_template(row["messages"], tokenize=False)
        ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids
        ids = ids.to(model.device)
        loss = model(ids, labels=ids).loss
        total += loss.item() * ids.shape[1]
        count += ids.shape[1]
    return math.exp(total / max(count, 1))


def run(cfg: dict, candidate: Path) -> dict:
    max_len = cfg["training"]["max_seq_len"]
    holdout = list(read_jsonl(Path(cfg["paths"]["datasets"]) / "holdout.jsonl"))
    golden_path = Path(cfg["paths"]["golden"])
    golden = list(read_jsonl(golden_path)) if golden_path.exists() else []

    base, tok = load_base(cfg)
    model = PeftModel.from_pretrained(base, candidate, adapter_name="candidate")
    current = current_adapter(cfg)
    if current:
        model.load_adapter(current, adapter_name="current")
    model.eval()

    def score(rows: list[dict], adapter: str | None) -> float:
        if not rows:
            return float("nan")
        if adapter:
            model.set_adapter(adapter)
        ctx = model.disable_adapter() if adapter is None else nullcontext()
        with ctx:
            return perplexity(model, tok, rows, max_len)

    ref = "current" if current else None
    report = {
        "holdout": {"ref": score(holdout, ref), "candidate": score(holdout, "candidate")},
        "golden": {"ref": score(golden, ref), "candidate": score(golden, "candidate")},
    }
    e = cfg["eval"]
    h, g = report["holdout"], report["golden"]
    gain = (h["ref"] - h["candidate"]) / h["ref"]
    regression = 0.0 if not golden else (g["candidate"] - g["ref"]) / g["ref"]
    report.update(
        holdout_gain=gain,
        golden_regression=regression,
        golden_checked=bool(golden),
        promote=gain >= e["min_holdout_gain"] and regression <= e["max_golden_regression"],
    )
    return report
