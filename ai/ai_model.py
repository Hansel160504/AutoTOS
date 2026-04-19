import sys
import os

_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from autotos import (  # noqa: F401, E402
    app,
    generate_from_records,
    generate_quiz_for_topics,
    get_model_cache_stats,
    lesson_from_upload,
)
# Export Progress through the SAME import path as generation uses,
# so both the /progress endpoint and generate_from_records share
# one class object (not two separate copies).
from autotos.generator import Progress  # noqa: F401

__all__ = [
    "app",
    "generate_from_records",
    "generate_quiz_for_topics",
    "get_model_cache_stats",
    "lesson_from_upload",
    "Progress",
]