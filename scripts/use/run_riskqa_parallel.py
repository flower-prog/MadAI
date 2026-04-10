from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import threading
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR in sys.path:
    sys.path.remove(PROJECT_ROOT_STR)
sys.path.insert(0, PROJECT_ROOT_STR)


from agent.config.env import load_dotenv_if_present
from agent.workflow import run_workflow

if __package__ in {None, ""}:
    from scripts.use.riskqa_support import (
        append_record_block,
        build_question_query,
        build_task_id,
        extract_choice_from_text,
        extract_last_assistant_message,
        load_dataset,
        normalize_choices,
        select_entries,
    )
    from scripts.use.try_single_case_workflow import build_summary, default_corpus_paths
else:
    from .riskqa_support import (
        append_record_block,
        build_question_query,
        build_task_id,
        extract_choice_from_text,
        extract_last_assistant_message,
        load_dataset,
        normalize_choices,
        select_entries,
    )
    from .try_single_case_workflow import build_summary, default_corpus_paths


DEFAULT_DATASET_PATH = PROJECT_ROOT / "数据" / "riskqa.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "riskqa" / "riskqa_answers.txt"


def _resolve_existing_path(raw_path: str | None, *, label: str) -> str | None:
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    return str(path.resolve())


def _first_existing(paths: list[Path | None]) -> str | None:
    for path in paths:
        if path is not None and path.exists():
            return str(path.resolve())
    return None


def resolve_batch_corpus_paths(
    root: Path,
    *,
    riskcalcs_path: str | None,
    pmid_metadata_path: str | None,
) -> tuple[str | None, str | None]:
    explicit_riskcalcs = _resolve_existing_path(riskcalcs_path, label="RiskCalcs")
    explicit_pmid = _resolve_existing_path(pmid_metadata_path, label="PMID metadata")
    if explicit_riskcalcs or explicit_pmid:
        return explicit_riskcalcs, explicit_pmid

    default_riskcalcs_path, default_pmid_metadata_path = default_corpus_paths(root)
    resolved_riskcalcs = _first_existing(
        [
            root / "数据" / "riskcalcs.json",
            root / "data" / "data" / "riskcalcs.json",
            root / "data" / "riskcalcs.json",
            Path(default_riskcalcs_path) if default_riskcalcs_path else None,
        ]
    )
    resolved_pmid = _first_existing(
        [
            root / "数据" / "pmid2info.json",
            root / "data" / "data" / "pmid2info.json",
            root / "data" / "pmid2info.json",
            Path(default_pmid_metadata_path) if default_pmid_metadata_path else None,
        ]
    )
    return resolved_riskcalcs, resolved_pmid


def extract_answer_text(summary: dict[str, Any]) -> str:
    execution = dict(summary.get("execution") or {})
    assistant_message = extract_last_assistant_message(list(execution.get("messages") or []))
    if assistant_message:
        return assistant_message
    return str(execution.get("final_text") or "").strip()


def build_result_record(
    *,
    task_id: str,
    dataset_index: int,
    entry: dict[str, Any],
    workflow_result: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    choices = normalize_choices(entry.get("choices"))
    gold_answer = str(entry.get("answer") or "").strip().upper() or None
    summary = build_summary(workflow_result)
    answer_text = extract_answer_text(summary)
    predicted_choice = extract_choice_from_text(answer_text, choices)

    return {
        "task_id": task_id,
        "dataset_index": dataset_index,
        "question": str(entry.get("question") or "").strip(),
        "query_text": build_question_query(entry),
        "choices": choices,
        "gold_answer": gold_answer,
        "gold_pmid": str(entry.get("pmid") or "").strip() or None,
        "predicted_choice": predicted_choice,
        "correct": (predicted_choice == gold_answer) if predicted_choice and gold_answer else None,
        "status": str(summary.get("status") or workflow_result.get("status") or "").strip() or "unknown",
        "review_passed": summary.get("review_passed"),
        "selected_tool": dict(summary.get("selected_tool") or {}),
        "answer_text": answer_text or None,
        "clinical_answer": list(summary.get("clinical_answer") or []),
        "errors": list(summary.get("errors") or []),
        "elapsed_seconds": round(float(elapsed_seconds), 3),
    }


def build_failure_record(
    *,
    task_id: str,
    dataset_index: int,
    entry: dict[str, Any],
    error_message: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    choices = normalize_choices(entry.get("choices"))
    gold_answer = str(entry.get("answer") or "").strip().upper() or None
    return {
        "task_id": task_id,
        "dataset_index": dataset_index,
        "question": str(entry.get("question") or "").strip(),
        "query_text": build_question_query(entry),
        "choices": choices,
        "gold_answer": gold_answer,
        "gold_pmid": str(entry.get("pmid") or "").strip() or None,
        "predicted_choice": None,
        "correct": False if gold_answer else None,
        "status": "failed",
        "review_passed": False,
        "selected_tool": {},
        "answer_text": None,
        "clinical_answer": [],
        "errors": [error_message],
        "elapsed_seconds": round(float(elapsed_seconds), 3),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MedAI concurrently over every RiskQA question and save answer records to a text file."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH), help="Path to riskqa.json.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Path to the output txt file.")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent worker threads.")
    parser.add_argument("--start-index", type=int, default=0, help="Start processing from this dataset index.")
    parser.add_argument("--end-index", type=int, default=None, help="Stop before this dataset index.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N rows after slicing.")
    parser.add_argument("--top-k", type=int, default=None, help="Retrieval top-k override.")
    parser.add_argument("--llm-model", default=None, help="Optional model override.")
    parser.add_argument("--retriever-backend", choices=("keyword", "vector", "hybrid"), default="hybrid")
    parser.add_argument("--riskcalcs-path", default=None, help="Optional override for riskcalcs.json.")
    parser.add_argument("--pmid-metadata-path", default=None, help="Optional override for pmid2info.json.")
    parser.add_argument("--debug", action="store_true", help="Enable workflow debug logging.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    load_dotenv_if_present(PROJECT_ROOT / ".env")

    dataset_path = Path(args.dataset).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")

    entries = load_dataset(dataset_path)
    selected_entries = select_entries(
        entries,
        start_index=args.start_index,
        end_index=args.end_index,
        limit=args.limit,
    )
    if not selected_entries:
        raise SystemExit("No RiskQA entries were selected. Adjust --start-index/--end-index/--limit.")

    resolved_riskcalcs_path, resolved_pmid_metadata_path = resolve_batch_corpus_paths(
        PROJECT_ROOT,
        riskcalcs_path=args.riskcalcs_path,
        pmid_metadata_path=args.pmid_metadata_path,
    )
    if not resolved_riskcalcs_path or not resolved_pmid_metadata_path:
        raise SystemExit(
            "Unable to resolve riskcalcs / pmid metadata paths. Pass --riskcalcs-path and --pmid-metadata-path explicitly."
        )

    total = len(selected_entries)
    worker_count = max(1, min(int(args.workers), total))
    write_lock = threading.Lock()

    def run_one(dataset_index: int, entry: dict[str, Any]) -> dict[str, Any]:
        task_id = build_task_id(dataset_index)
        started_at = time.perf_counter()
        try:
            workflow_result = run_workflow(
                case_text=build_question_query(entry),
                mode="question",
                riskcalcs_path=resolved_riskcalcs_path,
                pmid_metadata_path=resolved_pmid_metadata_path,
                llm_model=args.llm_model,
                retriever_backend=args.retriever_backend,
                top_k=args.top_k,
                max_selected_tools=1,
                debug=args.debug,
            )
            elapsed_seconds = time.perf_counter() - started_at
            return build_result_record(
                task_id=task_id,
                dataset_index=dataset_index,
                entry=entry,
                workflow_result=workflow_result,
                elapsed_seconds=elapsed_seconds,
            )
        except Exception as exc:
            elapsed_seconds = time.perf_counter() - started_at
            return build_failure_record(
                task_id=task_id,
                dataset_index=dataset_index,
                entry=entry,
                error_message=f"{exc.__class__.__name__}: {exc}",
                elapsed_seconds=elapsed_seconds,
            )

    futures: dict[Any, tuple[int, int]] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for position, (dataset_index, entry) in enumerate(selected_entries, start=1):
            future = executor.submit(run_one, dataset_index, entry)
            futures[future] = (position, dataset_index)

        for future in as_completed(futures):
            position, _dataset_index = futures[future]
            record = future.result()
            with write_lock:
                append_record_block(output_path, record)
            correctness = (
                "correct"
                if record.get("correct") is True
                else ("wrong" if record.get("correct") is False else "unscored")
            )
            print(
                f"[{position}/{total}] {record['task_id']} | status={record['status']} | "
                f"pred={record.get('predicted_choice') or '?'} | gold={record.get('gold_answer') or '?'} | {correctness}"
            )

    print("")
    print(f"Dataset: {dataset_path}")
    print(f"Output txt: {output_path}")
    print(f"Workers: {worker_count}")
    print(f"Resolved riskcalcs: {resolved_riskcalcs_path}")
    print(f"Resolved pmid metadata: {resolved_pmid_metadata_path}")
    print(f"Completed tasks: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
