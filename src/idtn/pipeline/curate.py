"""Curation : nouveaux logs -> dédoublonnage -> scrub PII -> auto-jugement -> train/holdout."""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

from idtn.pipeline.common import read_jsonl, to_messages, write_jsonl

PII_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "<EMAIL>"),
    (re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b"), "<IBAN>"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "<CARD>"),
    (re.compile(r"(?:\+\d{1,3}[ .-]?)?(?:\(?\d{1,4}\)?[ .-]?){2,5}\d{2,4}\b"), "<PHONE>"),
]


def scrub(text: str) -> str:
    for pattern, token in PII_PATTERNS:
        text = pattern.sub(token, text)
    return text


def _key(messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()


def load_cursor(cursor_file: Path) -> dict[str, int]:
    """{fichier journalier: nb de lignes déjà consommées}."""
    return json.loads(cursor_file.read_text()) if cursor_file.exists() else {}


def run(cfg: dict, judge=None) -> dict:
    """Retourne {"train": n, "holdout": n, "cursor": {...}}. `judge(messages) -> int 1..5`."""
    paths = cfg["paths"]
    cur = cfg["curation"]
    datasets = Path(paths["datasets"])
    cursor = load_cursor(datasets / "cursor.json")
    new_cursor = dict(cursor)

    seen: set[str] = set()
    rows: list[dict] = []
    for f in sorted(Path(paths["logs"]).glob("*.jsonl")):
        records = list(read_jsonl(f))
        new_cursor[f.name] = len(records)
        for record in records[cursor.get(f.name, 0):]:
            if not record.get("answer"):
                continue
            msgs = to_messages(record)
            if cur["scrub_pii"]:
                msgs = [{**m, "content": scrub(m["content"])} for m in msgs]
            k = _key(msgs)
            if cur["dedupe"] and k in seen:
                continue
            seen.add(k)
            if judge and cur["self_judge"] and judge(msgs) < cur["self_judge_min_score"]:
                continue
            rows.append({"messages": msgs})

    random.Random(0).shuffle(rows)
    n_hold = max(1, int(len(rows) * cur["holdout_ratio"])) if rows else 0
    write_jsonl(datasets / "holdout.jsonl", rows[:n_hold])
    write_jsonl(datasets / "train.jsonl", rows[n_hold:])
    return {"train": len(rows) - n_hold, "holdout": n_hold, "cursor": new_cursor}


def commit_cursor(cfg: dict, cursor: dict[str, int]) -> None:
    """Appelé seulement si la nuit s'est terminée : sinon les logs seront repris demain."""
    path = Path(cfg["paths"]["datasets"]) / "cursor.json"
    path.write_text(json.dumps(cursor, indent=2))
