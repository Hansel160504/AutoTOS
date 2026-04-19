"""AutoTOS AI service — modular package.

Public API (re-exported by ai_model.py shim for backwards compatibility):
    generate_quiz_for_topics(records, max_items, test_labels)
    generate_from_records(records, max_items)
    lesson_from_upload(data)
    get_model_cache_stats()
    app  (FastAPI instance)
"""
from .generator import (
    generate_from_records,
    generate_quiz_for_topics,
    get_model_cache_stats,
)
from .io_utils import lesson_from_upload
from .api import app

__all__ = [
    "generate_from_records",
    "generate_quiz_for_topics",
    "get_model_cache_stats",
    "lesson_from_upload",
    "app",
]