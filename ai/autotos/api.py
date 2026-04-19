"""FastAPI application — thin HTTP layer over the generator package.

Connectivity check + warm-up moved off the import path to avoid blocking
container startup (Ollama may come up after the AI service starts polling).
"""
from __future__ import annotations

import json
import logging
import threading
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .config import (
    CHUNK_SIZE,
    GENERATION_WORKERS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
)
from .generator import (
    Progress,
    generate_from_records,
    generate_quiz_for_topics,
    get_model_cache_stats,
)
from .io_utils import lesson_from_upload
from .llm import is_ready, warmup

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

logger.info(
    "Ollama config: base=%s model=%s timeout=%ds workers=%d",
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, GENERATION_WORKERS,
)

app = FastAPI(title="AutoTOS AI Service", version="5.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"],
)


# ── Lifecycle: lazy Ollama check + background warm-up ─────────────────
@app.on_event("startup")
async def _on_startup() -> None:
    # Don't block — fire-and-forget so /health works while Ollama is loading.
    threading.Thread(target=_startup_check, name="autotos-warmup", daemon=True).start()


def _startup_check() -> None:
    if is_ready(force_recheck=True):
        warmup()


# ── Schemas ───────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    records:     List[dict]
    max_items:   Optional[int] = Field(default=None, ge=1)
    test_labels: Optional[List[str]] = None


class ExtractRequest(BaseModel):
    data: str


# ── Routes ────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    ready = is_ready()
    return {
        "status":             "ok" if ready else "degraded",
        "ollama_ready":       ready,
        "ollama_model":       OLLAMA_MODEL,
        "chunk_size":         CHUNK_SIZE,
        "generation_workers": GENERATION_WORKERS,
        "version":            "5.0",
    }


@app.get("/cache_stats")
async def cache_stats():
    return get_model_cache_stats()


@app.get("/progress")
async def generation_progress():
    return Progress.snapshot()


@app.post("/extract")
async def extract(req: ExtractRequest):
    try:
        text = await run_in_threadpool(lesson_from_upload, req.data)
        return {"text": text or ""}
    except Exception as exc:
        logger.exception("extract error: %s", exc)
        raise HTTPException(status_code=500, detail="Extraction failed")


@app.post("/generate")
async def generate(req: GenerateRequest):
    if not req.records:
        raise HTTPException(status_code=400, detail="records must be non-empty")
    try:
        return await run_in_threadpool(
            generate_quiz_for_topics, req.records, req.max_items, req.test_labels,
        )
    except Exception as exc:
        logger.exception("generate error: %s", exc)
        raise HTTPException(status_code=500, detail="Generation failed")


@app.post("/generate_from_records")
async def generate_from_records_endpoint(payload: GenerateRequest):
    if not payload.records:
        raise HTTPException(status_code=400, detail="records must be non-empty")
    try:
        out = await run_in_threadpool(
            generate_from_records, payload.records, payload.max_items,
        )
        return {"quizzes": out}
    except Exception as exc:
        logger.exception("generate_from_records error: %s", exc)
        raise HTTPException(status_code=500, detail="Generation failed")


# ── CLI for local one-off testing ─────────────────────────────────────
def _cli() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", "-j")
    parser.add_argument("--sample", "-n", type=int, default=None)
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()

    if args.jsonl and not args.serve:
        recs = []
        with open(args.jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        recs.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        logger.warning("Skipping bad line: %s", exc)
        n = args.sample or min(5, len(recs))
        for r in generate_from_records(recs[:n], max_items=n):
            print(json.dumps(r, indent=2, ensure_ascii=False))
    elif args.serve:
        uvicorn.run("autotos.api:app", host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    _cli()