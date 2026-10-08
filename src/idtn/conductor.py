"""Point d'entrée des Jobs : `idtn bootstrap | night | guards`.

La bascule jour/nuit de l'inférence est faite par KEDA ; l'arbitrage des ressources
par KAI. Le conductor ne fait qu'enchaîner les étapes du pipeline de nuit.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from idtn import config, guards

log = logging.getLogger("idtn")


def bootstrap(cfg: dict) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    models = Path(cfg["paths"]["models"])
    m = cfg["model"]
    snapshot_download(m["hf_id"], local_dir=models / "base-hf")
    if cfg["inference"]["backend"] == "llamacpp":
        repo = m.get("gguf_repo", f"{m['hf_id']}-GGUF")
        path = hf_hub_download(repo, m["gguf_file"], local_dir=models)
        link = models / "base.gguf"
        link.unlink(missing_ok=True)
        link.symlink_to(Path(path).name)
    for key in ("logs", "datasets", "adapters"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
    log.info("bootstrap terminé (%s)", m["hf_id"])


def _run_dir(cfg: dict) -> Path:
    """Reprend le dernier run inachevé (préemption, garde-fou, budget dépassé), sinon en crée un."""
    adapters = Path(cfg["paths"]["adapters"])
    for d in sorted((p for p in adapters.glob("20*") if p.is_dir()), reverse=True):
        report = d / "report.json"
        status = json.loads(report.read_text()).get("status") if report.exists() else None
        if status in (None, "interrupted"):
            return d
        break
    d = adapters / datetime.now(ZoneInfo(cfg["timezone"])).date().isoformat()
    d.mkdir(parents=True, exist_ok=True)
    return d


def night(cfg: dict) -> int:
    from idtn.pipeline import curate, evaluate, promote, train_lora

    if problems := guards.violations(cfg["guards"]):
        log.warning("nuit sautée : %s", problems)
        return 0

    run_dir = _run_dir(cfg)
    report_file = run_dir / "report.json"
    report: dict = json.loads(report_file.read_text()) if report_file.exists() else {}

    def save(**kw):
        report.update(kw)
        report_file.write_text(json.dumps(report, indent=2, default=str))

    # Après une préemption (KAI) le Job est relancé : on ne refait pas la curation.
    if "curation" not in report:
        judge = None
        if cfg["curation"]["self_judge"]:
            from idtn.pipeline.judge import make_judge

            judge = make_judge(cfg)
        save(curation=curate.run(cfg, judge=judge))
    stats = report["curation"]
    if stats["train"] < cfg["training"]["min_new_samples"]:
        save(status="skipped", reason=f"{stats['train']} exemples < min_new_samples")
        return 0

    adapter = train_lora.run(cfg, run_dir)
    if adapter is None:
        save(status="interrupted")  # reprise au prochain lancement depuis le checkpoint
        return 0

    verdict = evaluate.run(cfg, adapter)
    save(eval=verdict)
    if verdict["promote"]:
        promote.run(cfg, run_dir)
        save(status="promoted")
    else:
        save(status="rejected")
    curate.commit_cursor(cfg, stats["cursor"])
    log.info("nuit terminée : %s", report["status"])
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="idtn")
    parser.add_argument("command", choices=["bootstrap", "night", "guards"])
    args = parser.parse_args(argv)
    cfg = config.load()
    log.info("profil %s", cfg["profile"])

    if args.command == "bootstrap":
        bootstrap(cfg)
    elif args.command == "night":
        return night(cfg)
    elif args.command == "guards":
        problems = guards.violations(cfg["guards"])
        print(json.dumps({"ok": not problems, "violations": problems}))
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
