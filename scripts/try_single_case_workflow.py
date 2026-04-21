from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR in sys.path:
    sys.path.remove(PROJECT_ROOT_STR)
sys.path.insert(0, PROJECT_ROOT_STR)


from agent.config.env import load_dotenv_if_present
from agent.corpus_paths import discover_complete_corpus_pair, resolve_default_corpus_paths
from agent.workflow import run_workflow


DEFAULT_CASE_FILE = PROJECT_ROOT / "数据" / "病例.txt"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "outputs" / "try_single_case_workflow_result.json"
DEFAULT_SNAPSHOT_DIR = PROJECT_ROOT / "outputs" / "snapshots"
SECTION_PATTERN = re.compile(
    r'\{"label":"(?P<label>(?:\\.|[^"])*)","text":"(?P<text>(?:\\.|[^"])*)"\}',
    re.DOTALL,
)


def default_corpus_paths(root: Path) -> tuple[str | None, str | None]:
    try:
        riskcalcs_path, pmid_metadata_path = discover_complete_corpus_pair(root)
    except FileNotFoundError:
        riskcalcs_path, pmid_metadata_path = resolve_default_corpus_paths(root)
    return (
        str(riskcalcs_path) if riskcalcs_path is not None else None,
        str(pmid_metadata_path) if pmid_metadata_path is not None else None,
    )


def _resolve_optional_file_path(raw_path: str | None, *, label: str) -> str | None:
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    return str(path.resolve())


def _resolve_directory_path(raw_path: str | None) -> str | None:
    if raw_path is None:
        return None
    text = str(raw_path).strip()
    if not text:
        return None
    return str(Path(text).expanduser().resolve())


def _normalize_whitespace(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _decode_json_string(value: str) -> str:
    try:
        return json.loads(f'"{value}"')
    except json.JSONDecodeError:
        return value.replace('\\"', '"').replace("\\n", "\n")


def _extract_section_lines(raw_text: str) -> list[str]:
    section_lines: list[str] = []
    for match in SECTION_PATTERN.finditer(raw_text):
        label = _decode_json_string(match.group("label")).strip()
        text = _normalize_whitespace(_decode_json_string(match.group("text")))
        if not label or not text:
            continue
        section_lines.append(f"{label}：{text}")
    return section_lines


def normalize_case_text(raw_text: str) -> str:
    text = str(raw_text or "").strip()
    if not text:
        return ""

    if '"resolved_detail_abstract_zh_sections"' not in text:
        return _normalize_whitespace(text)

    section_lines = _extract_section_lines(text)
    if section_lines:
        return "\n".join(section_lines)

    if ',"resolved_detail_abstract_en":' in text:
        text = text.split(',"resolved_detail_abstract_en":', 1)[0]
    return _normalize_whitespace(text.rstrip('",'))


def read_case_text(case_file: Path) -> str:
    if not case_file.exists():
        raise FileNotFoundError(f"Case file not found: {case_file}")

    normalized = normalize_case_text(case_file.read_text(encoding="utf-8"))
    if not normalized:
        raise ValueError(f"Case file is empty after normalization: {case_file}")

    return normalized


def resolve_case_text(args: argparse.Namespace) -> tuple[str, str]:
    case_text_arg = str(args.case_text or "").strip()
    if case_text_arg:
        normalized = normalize_case_text(case_text_arg)
        if not normalized:
            raise ValueError("The provided --case-text is empty after normalization.")
        return normalized, "command line --case-text"

    case_file = Path(args.case_file).expanduser()
    return read_case_text(case_file), str(case_file)


def build_summary(result: dict[str, Any]) -> dict[str, Any]:
    final_output = dict(result.get("final_output") or {})
    clinical_tool_agent = dict(final_output.get("clinical_tool_agent") or {})
    execution = dict(clinical_tool_agent.get("execution") or {})
    raw_executions = list(clinical_tool_agent.get("executions") or [])
    if not raw_executions and execution:
        raw_executions = [execution]
    executions = [dict(item) for item in raw_executions if isinstance(item, dict)]

    return {
        "status": result.get("status"),
        "review_passed": result.get("review_passed"),
        "clinical_answer": list(result.get("clinical_answer") or final_output.get("clinical_answer") or []),
        "selected_tool": dict(clinical_tool_agent.get("selected_tool") or {}),
        "execution": execution,
        "executions": executions,
        "errors": list(result.get("errors") or []),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MedAI workflow against a single demo clinical case.")
    parser.add_argument(
        "--case-text",
        help="Clinical case text passed directly on the command line.",
    )
    parser.add_argument(
        "--case-file",
        default=str(DEFAULT_CASE_FILE),
        help="Path to the clinical case text file. Used when --case-text is not provided.",
    )
    parser.add_argument(
        "--mode",
        choices=("patient_note", "question"),
        default="patient_note",
        help="Workflow mode. Use patient_note for this rehabilitation case by default.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Retrieval recall top-k. Defaults to MEDAI_RETRIEVAL_TOP_K or MEDAI_TOP_K.",
    )
    parser.add_argument(
        "--riskcalcs-path",
        help="Optional override for riskcalcs.json. Defaults to local 数据/data first, then discovered corpora.",
    )
    parser.add_argument(
        "--pmid-metadata-path",
        help="Optional override for pmid2info.json. Defaults to local 数据/data first, then discovered corpora.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_FILE),
        help="Path to save the full workflow result JSON.",
    )
    parser.add_argument(
        "--show-json",
        action="store_true",
        help="Print the full workflow JSON instead of only the summary.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for the workflow run.",
    )
    parser.add_argument(
        "--snapshot-dir",
        default=str(DEFAULT_SNAPSHOT_DIR),
        help="Directory for workflow snapshots. Defaults to outputs/snapshots.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main() -> int:
    args = parse_args()
    load_dotenv_if_present(PROJECT_ROOT / ".env")
    default_riskcalcs_path, default_pmid_metadata_path = default_corpus_paths(PROJECT_ROOT)
    riskcalcs_path = _resolve_optional_file_path(args.riskcalcs_path, label="RiskCalcs") or default_riskcalcs_path
    pmid_metadata_path = (
        _resolve_optional_file_path(args.pmid_metadata_path, label="PMID metadata") or default_pmid_metadata_path
    )
    snapshot_dir = _resolve_directory_path(args.snapshot_dir)

    case_text, case_source = resolve_case_text(args)
    result = run_workflow(
        case_text=case_text,
        mode=args.mode,
        riskcalcs_path=riskcalcs_path,
        pmid_metadata_path=pmid_metadata_path,
        top_k=args.top_k,
        debug=args.debug,
        snapshot_dir=snapshot_dir,
    )

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Case source:", case_source)
    print("Result JSON:", output_path)
    if snapshot_dir:
        print("Snapshot dir:", snapshot_dir)
    if args.show_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(build_summary(result), ensure_ascii=False, indent=2))

    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
