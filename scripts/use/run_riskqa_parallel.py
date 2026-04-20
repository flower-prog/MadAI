from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR in sys.path:
    sys.path.remove(PROJECT_ROOT_STR)
sys.path.insert(0, PROJECT_ROOT_STR)


from scripts.run_riskqa_parallel import (  # noqa: E402
    build_evaluation_summary,
    build_failure_record,
    build_parser,
    build_result_record,
    default_evaluation_report_path,
    extract_answer_text,
    main,
    resolve_batch_corpus_paths,
)


__all__ = [
    "build_evaluation_summary",
    "build_failure_record",
    "build_parser",
    "build_result_record",
    "default_evaluation_report_path",
    "extract_answer_text",
    "main",
    "resolve_batch_corpus_paths",
]


if __name__ == "__main__":
    raise SystemExit(main())
