# ai_api.py  — lightweight FastAPI wrapper around ai_model.py
# Run with:  uvicorn ai_api:app --host 0.0.0.0 --port 8000
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from typing import List, Dict, Any, Optional
import logging

# Import everything needed from ai_model (model loads once here, at startup)
from ai.ai_model import (
    generate_quiz_for_topics,
    lesson_from_upload,
    get_model_cache_stats,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="AutoTOS AI Service", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Request schemas
# ─────────────────────────────────────────────

class GenerateRequest(BaseModel):
    records:     List[Dict[str, Any]]
    max_items:   Optional[int]       = None
    test_labels: Optional[List[str]] = None


class ExtractRequest(BaseModel):
    data: str   # plain text, file path, or data-URL


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "autotos-ai"}


@app.get("/cache_stats")
def cache_stats():
    """
    Returns model cache hit/miss counters.
    Called by dashboard.py → get_cache_stats() after every generation.
    Without this endpoint, Flask falls back to importlib, re-loading the model.
    """
    try:
        return get_model_cache_stats()
    except Exception as e:
        logger.exception("cache_stats error: %s", e)
        raise HTTPException(status_code=500, detail="cache_stats failed")


@app.post("/extract")
async def extract_text(req: ExtractRequest):
    """
    Extracts plain text from a data-URL (PDF/DOCX/PPTX), file path, or raw text.
    Called by dashboard.py → extract_lesson() during topic processing.
    Without this endpoint, Flask falls back to importlib, re-loading the model.
    """
    try:
        text = await run_in_threadpool(lambda: lesson_from_upload(req.data))
        return {"text": text or ""}
    except Exception as e:
        logger.exception("extract_text error: %s", e)
        raise HTTPException(status_code=500, detail="Extraction failed")


@app.post("/generate")
async def generate(req: GenerateRequest):
    """
    Generates quiz questions for the given records.
    Returns {"quizzes": [...]}
    """
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