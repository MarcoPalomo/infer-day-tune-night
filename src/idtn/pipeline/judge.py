"""Auto-jugement : le SLM (avec son adapter courant) note ses propres réponses de 1 à 5."""

from __future__ import annotations

import re

import torch
from peft import PeftModel

from idtn.pipeline.model import current_adapter, load_base

PROMPT = (
    "Évalue la réponse de l'assistant à la dernière demande de l'utilisateur. "
    "Critères : exactitude, utilité, clarté. Réponds uniquement par un chiffre de 1 à 5."
)


def make_judge(cfg: dict):
    model, tok = load_base(cfg)
    if adapter := current_adapter(cfg):
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    @torch.no_grad()
    def judge(messages: list[dict]) -> int:
        convo = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        chat = [{"role": "system", "content": PROMPT}, {"role": "user", "content": convo}]
        ids = tok.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt")
        ids = ids[:, -2048:].to(model.device)
        out = model.generate(ids, max_new_tokens=3, do_sample=False)
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        match = re.search(r"[1-5]", text)
        return int(match.group()) if match else 3

    return judge
