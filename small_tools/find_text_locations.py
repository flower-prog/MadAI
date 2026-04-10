from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERY = "department_tags"
SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    "node_modules",
    ".venv",
    "venv",
}


def _should_skip(path: Path) -> bool:
    return any(part in SKIP_DIR_NAMES for part in path.parts)


def iter_text_matches(root: Path, query: str) -> list[tuple[Path, int, int, str]]:
    matches: list[tuple[Path, int, int, str]] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if _should_skip(path):
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        for line_number, line in enumerate(text.splitlines(), start=1):
            column_number = line.find(query)
            if column_number < 0:
                continue
            matches.append((path.resolve(), line_number, column_number + 1, line.rstrip()))

    return matches


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search the MedAI project for a target string and print clickable "
            "absolute paths in the form path:line:column."
        )
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help=f"Root path to search. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help=f"Target string to search. Default: {DEFAULT_QUERY!r}",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=0,
        help="Maximum number of results to print. Default: 0 (print all matches).",
    )
    return parser


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = build_parser()
    args = parser.parse_args()

    root = Path(args.root).resolve()
    query = str(args.query or "")

    if not root.exists():
        parser.error(f"Root path does not exist: {root}")
    if not root.is_dir():
        parser.error(f"Root path is not a directory: {root}")
    if not query.strip():
        parser.error("Query string cannot be empty.")

    matches = iter_text_matches(root, query)
    if int(args.max_results) > 0:
        matches = matches[: int(args.max_results)]

    print(f"root={root}")
    print(f"query={query}")
    print(f"matches={len(matches)}")
    print()

    for path, line_number, column_number, line in matches:
        print(f"{path}:{line_number}:{column_number}: {line}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
