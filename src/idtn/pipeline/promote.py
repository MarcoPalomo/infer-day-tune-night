"""Promotion atomique d'un adapter + redémarrage de l'inférence si elle tourne."""

from __future__ import annotations

import logging
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from idtn.pipeline.model import base_path

log = logging.getLogger(__name__)
LLAMACPP_CONVERT = os.environ.get("IDTN_LORA_CONVERT", "/opt/llama.cpp/convert_lora_to_gguf.py")


def to_gguf(cfg: dict, run_dir: Path) -> None:
    subprocess.run(
        [
            "python3", LLAMACPP_CONVERT,
            "--base", str(base_path(cfg)),
            "--outfile", str(run_dir / "adapter.gguf"),
            str(run_dir / "peft"),
        ],
        check=True,
    )


def switch_current(cfg: dict, run_dir: Path) -> None:
    adapters = Path(cfg["paths"]["adapters"])
    tmp = adapters / ".current.tmp"
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(run_dir.name)  # relatif : survit à un autre point de montage
    os.replace(tmp, adapters / "current")


def restart_inference(namespace: str = "idtn") -> None:
    """Si l'inférence tourne (réveil nocturne), on la redémarre ; sinon le prochain
    démarrage par KEDA (cron du matin) chargera le nouvel adapter de lui-même."""
    from kubernetes import client
    from kubernetes import config as kconfig

    try:
        kconfig.load_incluster_config()
    except kconfig.ConfigException:  # hors cluster (dev local)
        log.info("hors cluster : pas de redémarrage")
        return
    apps = client.AppsV1Api()
    dep = apps.read_namespaced_deployment("inference", namespace)
    if not dep.spec.replicas:
        return
    now = datetime.now(UTC).isoformat()
    patch = {"spec": {"template": {"metadata": {"annotations": {"idtn/restartedAt": now}}}}}
    apps.patch_namespaced_deployment("inference", namespace, patch)


def run(cfg: dict, run_dir: Path) -> None:
    if cfg["inference"]["backend"] == "llamacpp":
        to_gguf(cfg, run_dir)
    switch_current(cfg, run_dir)
    restart_inference()
