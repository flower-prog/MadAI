from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR in sys.path:
    sys.path.remove(PROJECT_ROOT_STR)
sys.path.insert(0, PROJECT_ROOT_STR)


if __package__ in {None, ""}:
    from scripts.use.riskqa_support import build_task_id, load_dataset, load_record_blocks
else:
    from .riskqa_support import build_task_id, load_dataset, load_record_blocks


DEFAULT_DATASET_PATH = PROJECT_ROOT / "数据" / "riskqa.json"
DEFAULT_ANSWERS_PATH = PROJECT_ROOT / "outputs" / "riskqa" / "riskqa_answers.txt"


def resolve_dataset_index(record: dict[str, Any]) -> int | None:
    raw_index = record.get("dataset_index")
    if isinstance(raw_index, int):
        return raw_index
    task_id = str(record.get("task_id") or "").strip()
    if task_id.startswith("riskqa-"):
        suffix = task_id.split("-", 1)[1]
        if suffix.isdigit():
            return int(suffix)
    return None


def evaluate_records(records: list[dict[str, Any]], dataset_entries: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    correct_count = 0
    comparable_count = 0

    for record in records:
        dataset_index = resolve_dataset_index(record)
        if dataset_index is None or dataset_index < 0 or dataset_index >= len(dataset_entries):
            continue

        entry = dict(dataset_entries[dataset_index])
        gold_answer = str(entry.get("answer") or "").strip().upper() or None
        predicted_choice = str(record.get("predicted_choice") or "").strip().upper() or None
        correct = bool(gold_answer and predicted_choice and gold_answer == predicted_choice)
        if gold_answer:
            comparable_count += 1
            if correct:
                correct_count += 1

        rows.append(
            {
                "task_id": str(record.get("task_id") or build_task_id(dataset_index)),
                "dataset_index": dataset_index,
                "status": record.get("status"),
                "predicted_choice": predicted_choice,
                "gold_answer": gold_answer,
                "correct": correct,
                "question": str(entry.get("question") or "").strip(),
            }
        )

    wrong_rows = [row for row in rows if row.get("correct") is False]
    accuracy = (correct_count / comparable_count) if comparable_count else None
    return {
        "total_records": len(rows),
        "comparable_count": comparable_count,
        "correct_count": correct_count,
        "wrong_count": len(wrong_rows),
        "accuracy": accuracy,
        "rows": rows,
        "wrong_rows": wrong_rows,
    }


def build_report(summary: dict[str, Any], *, wrong_limit: int) -> str:
    accuracy = summary.get("accuracy")
    accuracy_text = f"{float(accuracy) * 100:.2f}%" if accuracy is not None else "N/A"
    lines = [
        "RiskQA answer evaluation",
        "",
        f"Total parsed records: {summary.get('total_records', 0)}",
        f"Comparable rows: {summary.get('comparable_count', 0)}",
        f"Correct rows: {summary.get('correct_count', 0)}",
        f"Wrong rows: {summary.get('wrong_count', 0)}",
        f"Accuracy: {accuracy_text}",
    ]

    wrong_rows = list(summary.get("wrong_rows") or [])[: max(wrong_limit, 0)]
    if wrong_rows:
        lines.extend(["", "Wrong examples:"])
        for row in wrong_rows:
            lines.append(
                f"- {row.get('task_id')} | pred={row.get('predicted_choice') or '?'} | "
                f"gold={row.get('gold_answer') or '?'} | question={row.get('question')}"
            )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a RiskQA txt result file by comparing predicted option letters against gold answers."
    )
    parser.add_argument("--answers", default=str(DEFAULT_ANSWERS_PATH), help="Path to the txt answers file.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH), help="Path to riskqa.json.")
    parser.add_argument("--report-path", default=None, help="Optional output report path. Defaults to <answers>.evaluation.txt.")
    parser.add_argument("--wrong-limit", type=int, default=20, help="How many wrong examples to include in the report.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    answers_path = Path(args.answers).expanduser().resolve()
    dataset_path = Path(args.dataset).expanduser().resolve()
    report_path = (
        Path(args.report_path).expanduser().resolve()
        if args.report_path
        else answers_path.with_name(f"{answers_path.stem}.evaluation.txt")
    )

    records = load_record_blocks(answers_path)
    dataset_entries = load_dataset(dataset_path)
    summary = evaluate_records(records, dataset_entries)
    report_text = build_report(summary, wrong_limit=args.wrong_limit)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")

    print(report_text, end="")
    print(f"Report saved to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
