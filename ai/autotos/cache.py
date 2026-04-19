"""Two-tier cache: in-memory LRU + on-disk JSON.

Used by both the model-call cache (deterministic re-use of identical prompts)
and the chunk cache (avoid re-tokenising the same document).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from collections import OrderedDict
from typing import Any, Optional

from .config import (
    CACHE_CLEANUP_EVERY,
    DISK_CACHE_MAX,
    MEM_CACHE_MAX,
    MODEL_CACHE_DIR,
    OLLAMA_MODEL,
)

logger = logging.getLogger(__name__)


class LRUCache:
    """Thread-safe size-bounded LRU."""

    __slots__ = ("_data", "_max", "_lock")

    def __init__(self, maxsize: int = MEM_CACHE_MAX) -> None:
        self._data: OrderedDict[str, Any] = OrderedDict()
        self._max = maxsize
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


# ── Module-level caches ───────────────────────────────────────────────
_mem_cache = LRUCache(MEM_CACHE_MAX)


# ── Disk cache helpers ────────────────────────────────────────────────
def prompt_hash_key(prompt: str, max_tokens: int, temperature: float, num_ctx: int) -> str:
    """Deterministic key for a prompt + generation params."""
    raw = f"{OLLAMA_MODEL}|{max_tokens}|{temperature:.4f}|{num_ctx}|{prompt}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _disk_path(key: str) -> str:
    return os.path.join(MODEL_CACHE_DIR, f"{key}.json")


def read_disk(key: str) -> Optional[Any]:
    path = _disk_path(key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("disk cache read failed (%s) — purging entry", exc)
        try:
            os.remove(path)
        except OSError:
            pass
        return None


_write_count = 0
_write_lock = threading.Lock()


def write_disk(key: str, obj: Any) -> None:
    """Atomic write via tempfile.replace; periodic LRU-style cleanup."""
    target = _disk_path(key)
    try:
        fd, tmp = tempfile.mkstemp(dir=MODEL_CACHE_DIR, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False)
            os.replace(tmp, target)
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
    except OSError as exc:
        logger.debug("disk cache write failed: %s", exc)
        return

    global _write_count
    with _write_lock:
        _write_count += 1
        should_cleanup = _write_count % CACHE_CLEANUP_EVERY == 0
    if should_cleanup:
        cleanup_disk()


def cleanup_disk(max_files: int = DISK_CACHE_MAX) -> None:
    try:
        files = [
            os.path.join(MODEL_CACHE_DIR, f)
            for f in os.listdir(MODEL_CACHE_DIR)
            if f.endswith(".json")
        ]
        if len(files) <= max_files:
            return
        files.sort(key=os.path.getmtime)
        for path in files[: len(files) - max_files]:
            try:
                os.remove(path)
            except OSError:
                pass
        logger.info("Disk cache cleanup: kept %d entries", max_files)
    except OSError as exc:
        logger.warning("Cache cleanup failed (non-fatal): %s", exc)


# ── Two-tier read/write API ───────────────────────────────────────────
class CacheStats:
    """Counters live on the class to make access lock-free in the hot path
    (single-writer, reader-tolerant — exact values are not critical)."""
    hits   = 0
    misses = 0


def get(key: str) -> Optional[Any]:
    """Mem-first, then disk. Promotes disk hits into mem."""
    val = _mem_cache.get(key)
    if val is not None:
        CacheStats.hits += 1
        return val
    val = read_disk(key)
    if val is not None:
        CacheStats.hits += 1
        _mem_cache.set(key, val)
        return val
    CacheStats.misses += 1
    return None


def put(key: str, value: Any) -> None:
    _mem_cache.set(key, value)
    write_disk(key, value)


def stats() -> dict:
    cache_files = sum(1 for f in os.listdir(MODEL_CACHE_DIR) if f.endswith(".json"))
    return {
        "cache_hits":       CacheStats.hits,
        "cache_misses":     CacheStats.misses,
        "mem_cache_size":   len(_mem_cache),
        "disk_cache_files": cache_files,
    }


# Run one cleanup pass at import to bound startup state.
cleanup_disk()