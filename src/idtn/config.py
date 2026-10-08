"""Chargement de la configuration : platform.yaml + surcharge du profil matériel."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Dans le cluster : ConfigMap montée à plat dans /etc/idtn.
# En local : arborescence du dépôt (config/ et config/profiles/).
_REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"


def _config_dir() -> Path:
    if env := os.environ.get("IDTN_CONFIG_DIR"):
        return Path(env)
    return Path("/etc/idtn") if Path("/etc/idtn/platform.yaml").exists() else _REPO_CONFIG


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load() -> dict[str, Any]:
    root = _config_dir()
    cfg = yaml.safe_load((root / "platform.yaml").read_text())
    profile = os.environ.get("IDTN_PROFILE", cfg.get("profile", "cpu"))
    candidates = [root / f"{profile}.yaml", root / "profiles" / f"{profile}.yaml"]
    profile_file = next((p for p in candidates if p.exists()), None)
    if profile_file is None:
        raise FileNotFoundError(f"profil inconnu : {profile}")
    cfg = deep_merge(cfg, yaml.safe_load(profile_file.read_text()) or {})
    cfg["profile"] = profile
    return cfg
