"""Outils partagés par les étapes du pipeline de nuit."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_messages(record: dict) -> list[dict]:
    """Échange journalisé -> conversation complète (assistant en dernier)."""
    return [*record["messages"], {"role": "assistant", "content": record["answer"]}]


def torch_device(cfg: dict) -> str:
    # PyTorch ROCm expose les GPU AMD sous "cuda".
    return "cuda" if cfg["training"]["device"] in ("cuda", "rocm") else "cpu"
