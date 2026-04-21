from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR in sys.path:
    sys.path.remove(PROJECT_ROOT_STR)
sys.path.insert(0, PROJECT_ROOT_STR)


from scripts.riskqa_support import (  # noqa: E402
    append_record_block,
    build_question_query,
    build_task_id,
    extract_choice_from_text,
    extract_last_assistant_message,
    format_record_block,
    load_dataset,
    load_record_blocks,
    make_json_serializable,
    normalize_choices,
    normalize_match_text,
    parse_record_blocks,
    render_choices,
    select_entries,
)


__all__ = [
    "append_record_block",
    "build_question_query",
    "build_task_id",
    "extract_choice_from_text",
    "extract_last_assistant_message",
    "format_record_block",
    "load_dataset",
    "load_record_blocks",
    "make_json_serializable",
    "normalize_choices",
    "normalize_match_text",
    "parse_record_blocks",
    "render_choices",
    "select_entries",
]
