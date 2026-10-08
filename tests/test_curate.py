import json

from idtn.pipeline import curate


def _cfg(tmp_path):
    return {
        "paths": {"logs": str(tmp_path / "logs"), "datasets": str(tmp_path / "ds")},
        "curation": {"dedupe": True, "scrub_pii": True, "holdout_ratio": 0.5,
                     "self_judge": False, "self_judge_min_score": 3},
    }


def test_scrub():
    out = curate.scrub("écris à jean.dupont@acme.fr ou au +33 6 12 34 56 78")
    assert "<EMAIL>" in out and "<PHONE>" in out and "acme" not in out


def test_dedupe_and_cursor(tmp_path):
    cfg = _cfg(tmp_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    rec = {"messages": [{"role": "user", "content": "salut"}], "answer": "bonjour"}
    other = {"messages": [{"role": "user", "content": "2+2"}], "answer": "4"}
    (logs / "2026-10-08.jsonl").write_text("\n".join(json.dumps(r) for r in (rec, rec, other)))

    stats = curate.run(cfg)
    assert stats["train"] + stats["holdout"] == 2
    curate.commit_cursor(cfg, stats["cursor"])

    # Rien de neuf : rien à curer.
    assert curate.run(cfg)["train"] == 0
