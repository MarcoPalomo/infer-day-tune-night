"""Gateway OpenAI-compatible.

- Journalise chaque échange (JSONL quotidien) : matière première de l'entraînement de nuit.
- Expose /keda/metrics {"pending": n} : KEDA réveille l'inférence depuis 0 réplique.
- Garde les requêtes en attente pendant le démarrage de l'inférence.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from idtn import config

CFG = config.load()
UPSTREAM = os.environ.get("IDTN_UPSTREAM", "http://inference:8080")
LOG_DIR = Path(CFG["paths"]["logs"])
TZ = ZoneInfo(CFG["timezone"])  # fichiers journaliers à l'heure locale du client
WAKE_TIMEOUT_S = float(os.environ.get("IDTN_WAKE_TIMEOUT_S", "300"))
MODEL_NAME = "idtn"  # nom exposé par llama.cpp (--alias) et vLLM (--served-model-name/lora)

app = FastAPI(title="idtn-gateway")
client = httpx.AsyncClient(base_url=UPSTREAM, timeout=httpx.Timeout(600, connect=5))
pending = 0


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/keda/metrics")
async def keda_metrics():
    return {"pending": pending}


async def _wait_upstream() -> bool:
    deadline = time.monotonic() + WAKE_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            if (await client.get("/health", timeout=2)).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        await asyncio.sleep(2)
    return False


def _log(request_body: dict, answer: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": time.time(),
        "messages": request_body.get("messages", []),
        "answer": answer,
    }
    with (LOG_DIR / f"{datetime.now(TZ).date().isoformat()}.jsonl").open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@app.post("/v1/chat/completions")
async def chat(request: Request):
    global pending
    body = await request.json()
    body["model"] = MODEL_NAME
    pending += 1
    try:
        if not await _wait_upstream():
            return JSONResponse({"error": "inference indisponible"}, status_code=503)
        if body.get("stream"):
            return StreamingResponse(_stream(body), media_type="text/event-stream")
        resp = await client.post("/v1/chat/completions", json=body)
        data = resp.json()
        if resp.status_code == 200:
            _log(body, data["choices"][0]["message"]["content"])
        return JSONResponse(data, status_code=resp.status_code)
    finally:
        if not body.get("stream"):
            pending -= 1


async def _stream(body: dict):
    global pending
    parts: list[str] = []
    try:
        async with client.stream("POST", "/v1/chat/completions", json=body) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        delta = json.loads(line[6:])["choices"][0]["delta"]
                        parts.append(delta.get("content") or "")
                    except (KeyError, IndexError, json.JSONDecodeError):
                        pass
                yield line + "\n"
        _log(body, "".join(parts))
    finally:
        pending -= 1


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}
