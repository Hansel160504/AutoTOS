from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from typing import List, Dict, Any, Optional
import logging

# Import everything (including Progress) from the shim so all symbols
# resolve to the same autotos.generator module object. Importing Progress
# via `from ai.autotos.generator import Progress` creates a second, separate
# class instance — Progress.tick() in generation would never affect its counter.
from ai.ai_model import (
    generate_quiz_for_topics,
    lesson_from_upload,
    get_model_cache_stats,
    Progress,
)

logger = logging.getLogger(__name__)
app = FastAPI(title="AutoTOS AI Service", version="1.4")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    records:     List[Dict[str, Any]]
    max_items:   Optional[int]       = None
    test_labels: Optional[List[str]] = None


class ExtractRequest(BaseModel):
    data: str


# ── Lightweight endpoints: async so they run on the event loop ────────
# and are never queued behind the blocking /generate thread pool call.

@app.get("/health")
async def health():
    return {"status": "ok", "service": "autotos-ai"}


@app.get("/cache_stats")
async def cache_stats():
    try:
        return get_model_cache_stats()
    except Exception as e:
        logger.exception("cache_stats error: %s", e)
        raise HTTPException(status_code=500, detail="cache_stats failed")


@app.get("/progress")
async def generation_progress():
    # Progress.snapshot() is microseconds of work (one lock + dict copy).
    # Safe on the event loop, and now reads the SAME Progress instance
    # that generate_from_records calls .tick() on.
    return Progress.snapshot()


# ── Blocking endpoints: offloaded to thread pool ──────────────────────

@app.post("/extract")
async def extract_text(req: ExtractRequest):
    try:
        text = await run_in_threadpool(lambda: lesson_from_upload(req.data))
        return {"text": text or ""}
    except Exception as e:
        logger.exception("extract_text error: %s", e)
        raise HTTPException(status_code=500, detail="Extraction failed")


@app.post("/generate")
async def generate(req: GenerateRequest):
    if not isinstance(req.records, list) or len(req.records) == 0:
        raise HTTPException(status_code=400, detail="records must be a non-empty list")
    try:
        result = await run_in_threadpool(
            lambda: generate_quiz_for_topics(
                req.records,
                max_items=req.max_items,
                test_labels=req.test_labels,
            )
        )
        return result
    except Exception as e:
        logger.exception("generate endpoint error: %s", e)
        raise HTTPException(status_code=500, detail="Generation failed")