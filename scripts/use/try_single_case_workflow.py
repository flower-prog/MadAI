from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR in sys.path:
    sys.path.remove(PROJECT_ROOT_STR)
sys.path.insert(0, PROJECT_ROOT_STR)


from scripts.try_single_case_workflow import (  # noqa: E402
    build_parser,
    build_summary,
    default_corpus_paths,
    main,
    normalize_case_text,
    parse_args,
    read_case_text,
    resolve_case_text,
)


__all__ = [
    "build_parser",
    "build_summary",
    "default_corpus_paths",
    "main",
    "normalize_case_text",
    "parse_args",
    "read_case_text",
    "resolve_case_text",
]


if __name__ == "__main__":
    raise SystemExit(main())
