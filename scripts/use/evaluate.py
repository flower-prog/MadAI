from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR in sys.path:
    sys.path.remove(PROJECT_ROOT_STR)
sys.path.insert(0, PROJECT_ROOT_STR)


from scripts.evaluate import (  # noqa: E402
    build_parser,
    build_report,
    evaluate_records,
    main,
    resolve_dataset_index,
)


__all__ = [
    "build_parser",
    "build_report",
    "evaluate_records",
    "main",
    "resolve_dataset_index",
]


if __name__ == "__main__":
    raise SystemExit(main())
