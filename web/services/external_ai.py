"""HTTP client for the external AI FastAPI service, with local fallback.

All ai_model interaction the dashboard needs lives here so the routes
contain no requests/importlib gymnastics.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import requests
from flask import current_app

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Progress tracker (one global; same shape as the AI service emits)
# ──────────────────────────────────────────────────────────────────────
class ProgressTracker:
    """Thread-safe progress state shared between requests and pollers."""

    __slots__ = ("_lock", "_state")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: Dict[str, Any] = {"current": 0, "total": 0, "active": False}

    def reset(self) -> None:
        with self._lock:
            self._state.update(current=0, total=0, active=False)

    def update(self, current: int, total: int, active: bool = True) -> None:
        with self._lock:
            self._state.update(current=current, total=total, active=active)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._state)


progress = ProgressTracker()


# ──────────────────────────────────────────────────────────────────────
# URL resolution
# ──────────────────────────────────────────────────────────────────────
def _model_url() -> Optional[str]:
    return os.getenv("AUTO_TOS_MODEL_URL") or current_app.config.get("AUTO_TOS_MODEL_URL")


def _model_timeout() -> int:
    return int(os.getenv("AUTO_TOS_MODEL_TIMEOUT", "5"))


# ──────────────────────────────────────────────────────────────────────
# Local-import fallback (for single-container deploys)
# ──────────────────────────────────────────────────────────────────────
def _load_local_ai():
    """Lazy import — keeps web container lean if running with remote AI only."""
    import importlib
    return importlib.import_module("ai_model")


# ──────────────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────────────
def _mirror_remote_progress(model_url: str, total: int, stop: threading.Event) -> None:
    """Poll the AI service /progress and reflect it locally so the UI updates."""
    prog_endpoint = model_url.rstrip("/") + "/progress"
    while not stop.is_set():
        try:
            r = requests.get(prog_endpoint, timeout=2)
            if r.ok:
                d = r.json()
                progress.update(d.get("current", 0), d.get("total", total))
        except requests.RequestException:
            pass
        time.sleep(0.8)


def call_model_service(
    expanded_records: List[dict],
    test_labels: Optional[List[str]] = None,
) -> Optional[dict]:
    """Generate quizzes via the external AI service, falling back to local import."""
    total = len(expanded_records)
    progress.update(0, total)

    url = _model_url()
    if url:
        result = _call_remote_generate(url, expanded_records, test_labels, total)
        if result is not None:
            return result

    # Fallback: in-process generation.
    try:
        ai = _load_local_ai()
        result = ai.generate_quiz_for_topics(
            expanded_records, max_items=None, test_labels=test_labels
        )
        progress.update(total, total)
        return result
    except Exception as exc:
        logger.exception("Local generation failed: %s", exc)
        return None


def _call_remote_generate(
    url: str,
    records: List[dict],
    test_labels: Optional[List[str]],
    total: int,
) -> Optional[dict]:
    endpoint = url.rstrip("/") + "/generate"
    stop = threading.Event()
    poller = threading.Thread(
        target=_mirror_remote_progress, args=(url, total, stop), daemon=True,
    )
    poller.start()

    try:
        resp = requests.post(
            endpoint,
            json={"records": records, "test_labels": test_labels},
            timeout=None,
        )
    except requests.RequestException as exc:
        logger.exception("External /generate failed: %s", exc)
        stop.set(); poller.join(timeout=2)
        return None

    stop.set(); poller.join(timeout=2)

    if not resp.ok:
        logger.warning("Model service returned %d: %s", resp.status_code, resp.text[:500])
        return None

    try:
        data = resp.json()
    except ValueError as exc:
        logger.warning("Failed to parse /generate JSON: %s", exc)
        return None

    progress.update(total, total)
    return data


# ──────────────────────────────────────────────────────────────────────
# Lesson extraction
# ──────────────────────────────────────────────────────────────────────
def extract_lesson(data_or_text: str) -> str:
    """Extract text from a file path / data URL / raw text. Remote first, local fallback."""
    url = _model_url()
    if url:
        try:
            resp = requests.post(
                url.rstrip("/") + "/extract",
                json={"data": data_or_text},
                timeout=None,
            )
            if resp.ok:
                payload = resp.json()
                if isinstance(payload, dict) and "text" in payload:
                    return payload.get("text") or ""
                if isinstance(payload, str):
                    return payload
        except requests.RequestException as exc:
            logger.debug("Remote /extract failed (will fall back): %s", exc)

    try:
        ai = _load_local_ai()
        return ai.lesson_from_upload(data_or_text)
    except Exception as exc:
        logger.exception("Local lesson_from_upload failed: %s", exc)
        return ""


# ──────────────────────────────────────────────────────────────────────
# Cache stats
# ──────────────────────────────────────────────────────────────────────
def get_cache_stats() -> Dict[str, Any]:
    url = _model_url()
    if url:
        try:
            resp = requests.get(url.rstrip("/") + "/cache_stats", timeout=_model_timeout())
            if resp.ok:
                return resp.json()
        except requests.RequestException:
            pass
    try:
        ai = _load_local_ai()
        if hasattr(ai, "get_model_cache_stats"):
            return ai.get_model_cache_stats()
    except Exception:
        pass
    return {}


# ──────────────────────────────────────────────────────────────────────
# Remote progress (for the dashboard's /generation_progress endpoint)
# ──────────────────────────────────────────────────────────────────────
def fetch_remote_progress() -> Optional[Dict[str, Any]]:
    """Return the AI service's progress snapshot, or None if no remote configured."""
    url = _model_url()
    if not url:
        return None
    try:
        resp = requests.get(url.rstrip("/") + "/progress", timeout=2)
        if resp.ok:
            return resp.json()
    except requests.RequestException:
        pass
    return None