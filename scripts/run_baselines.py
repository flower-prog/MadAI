from __future__ import annotations

__author__ = "qiao"

"""
Run a direct LLM baseline on the RiskQA dataset.

This script asks a model to answer each multiple-choice question directly,
stores a short rationale for each prediction, and writes both a detailed
comparison file and a compatibility file that can be scored with
`get_overall_performance.py`.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from openai import AzureOpenAI, OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR in sys.path:
    sys.path.remove(PROJECT_ROOT_STR)
sys.path.insert(0, PROJECT_ROOT_STR)

from agent.config.env import load_dotenv_if_present


DEFAULT_AZURE_API_VERSION = "2023-09-01-preview"
ENV_FILENAMES = (".env", ".env.local")
DEFAULT_DATASET_PATH = PROJECT_ROOT / "\u6570\u636e" / "riskqa.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "riskqa"
CHOICE_PATTERN = re.compile(r"^\s*Choice\s*:\s*([A-F])\b", re.IGNORECASE | re.MULTILINE)
ANSWER_PATTERN = re.compile(r"^\s*Answer\s*:\s*([A-F])\b", re.IGNORECASE | re.MULTILINE)
LEADING_CHOICE_PATTERN = re.compile(r"^\s*([A-F])\s*[\.\):]\s*", re.IGNORECASE)
BRIEF_REASON_PATTERN = re.compile(
    r"^\s*Brief reason\s*:\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)

SYSTEM_PROMPT = """You are answering one medical multiple-choice question at a time.
Pick exactly one option from the provided choices.
Do not show long chain-of-thought.
Return exactly two lines in this format:
Choice: <single letter>
Brief reason: <one short sentence>"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run direct LLM baseline on RiskQA.")
    parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help="Optional model name or Azure deployment name. Defaults to values loaded from .env.",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="Path to the RiskQA dataset JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for cached outputs and summaries.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Optional number of questions to evaluate.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Dataset start index for partial runs.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between API calls.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent worker threads.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore any cached results and start over.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return base_dir / path


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "model"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_env_value(raw_value: str) -> str:
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


def load_env_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        if not key:
            continue
        os.environ.setdefault(key, parse_env_value(raw_value))


def bootstrap_environment(base_dir: Path) -> list[str]:
    loaded_paths: list[str] = []
    candidate_dirs = [base_dir, base_dir.parent]
    for candidate_dir in candidate_dirs:
        for filename in ENV_FILENAMES:
            env_path = candidate_dir / filename
            if env_path.exists():
                load_env_file(env_path)
                loaded_paths.append(str(env_path))
    return loaded_paths


def dump_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def resolve_default_model(explicit_model: str | None) -> str:
    if explicit_model and explicit_model.strip():
        return explicit_model.strip()

    for env_name in (
        "OPENAI_MODEL",
        "BASIC_MODEL",
        "CLINICAL_TOOL_AGENT_MODEL",
        "CODING_MODEL",
    ):
        env_value = os.getenv(env_name)
        if env_value and env_value.strip():
            return env_value.strip()

    raise RuntimeError(
        "No model provided. Pass a model argument or set one of "
        "OPENAI_MODEL, BASIC_MODEL, CLINICAL_TOOL_AGENT_MODEL, CODING_MODEL in .env."
    )


def build_client() -> tuple[Any, str]:
    endpoint = os.getenv("OPENAI_ENDPOINT")
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. For Azure, also set OPENAI_ENDPOINT. "
            "For direct OpenAI usage, leave OPENAI_ENDPOINT unset."
        )

    if endpoint:
        client = AzureOpenAI(
            api_version=os.getenv("OPENAI_API_VERSION", DEFAULT_AZURE_API_VERSION),
            azure_endpoint=endpoint,
            api_key=api_key,
        )
        return client, "azure"

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url

    return OpenAI(**client_kwargs), "openai"


def format_question(entry: dict[str, Any]) -> str:
    lines = [f"Question: {entry['question']}", "Choices:"]
    for label, choice in entry["choices"].items():
        lines.append(f"{label}. {choice}")
    lines.append("")
    lines.append("Return the most likely answer.")
    return "\n".join(lines)


def _normalize_valid_choices(valid_choices: set[str] | None) -> set[str] | None:
    if valid_choices is None:
        return None
    return {choice.upper() for choice in valid_choices}


def extract_choice_token(answer: str, valid_choices: set[str] | None = None) -> str:
    normalized_choices = _normalize_valid_choices(valid_choices)

    for pattern in (CHOICE_PATTERN, ANSWER_PATTERN):
        match = pattern.search(answer)
        if not match:
            continue
        choice = match.group(1).upper()
        if normalized_choices is None or choice in normalized_choices:
            return choice

    first_nonempty_line = next((line.strip() for line in answer.splitlines() if line.strip()), "")
    leading_match = LEADING_CHOICE_PATTERN.match(first_nonempty_line)
    if leading_match:
        choice = leading_match.group(1).upper()
        if normalized_choices is None or choice in normalized_choices:
            return choice

    return "X"


def extract_answer(answer: str) -> str:
    return extract_choice_token(answer, {"A", "B", "C", "D", "E", "F"})


def extract_brief_reason(text: str) -> str:
    brief_match = BRIEF_REASON_PATTERN.search(text)
    if brief_match:
        return brief_match.group(1).strip()

    nonempty_lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not nonempty_lines:
        return "No brief reason returned."

    if CHOICE_PATTERN.match(nonempty_lines[0]) or ANSWER_PATTERN.match(nonempty_lines[0]):
        trailing_lines = nonempty_lines[1:]
        if trailing_lines:
            return " ".join(trailing_lines)
        return "No brief reason returned."

    leading_match = LEADING_CHOICE_PATTERN.match(nonempty_lines[0])
    if leading_match:
        stripped = nonempty_lines[0][leading_match.end() :].strip()
        if stripped:
            return stripped
        trailing_lines = nonempty_lines[1:]
        if trailing_lines:
            return " ".join(trailing_lines)
        return "No brief reason returned."

    return " ".join(nonempty_lines)


def parse_response(text: str, valid_choices: set[str]) -> tuple[str, str]:
    choice = extract_choice_token(text, valid_choices)
    brief_reason = extract_brief_reason(text)
    return choice, brief_reason


def query_model(
    client: Any,
    model: str,
    prompt: str,
    valid_choices: set[str],
) -> tuple[str, str, str]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(3):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        choice, brief_reason = parse_response(content, valid_choices)
        if choice != "X":
            return content, choice, brief_reason

        if attempt == 2:
            return content, choice, brief_reason

        messages.extend(
            [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        "Your previous answer did not follow the required format. "
                        "Return exactly two lines:\n"
                        "Choice: <single letter>\n"
                        "Brief reason: <one short sentence>"
                    ),
                },
            ]
        )

    return "", "X", "No response returned."


def build_target_indices(dataset_size: int, start_index: int, max_questions: int | None) -> list[int]:
    if start_index < 0 or start_index >= dataset_size:
        raise ValueError(f"start-index must be between 0 and {dataset_size - 1}.")

    end_index = dataset_size
    if max_questions is not None:
        if max_questions <= 0:
            raise ValueError("max-questions must be positive.")
        end_index = min(dataset_size, start_index + max_questions)

    return list(range(start_index, end_index))


def compute_accuracy(comparison_rows: list[dict[str, Any]]) -> float:
    if not comparison_rows:
        return 0.0
    correct = sum(1 for row in comparison_rows if row["correct"])
    return correct / len(comparison_rows)


def compute_worker_count(*, requested_workers: int, task_count: int) -> int:
    if requested_workers <= 0:
        raise ValueError("workers must be positive.")
    if task_count <= 0:
        return 0
    return min(int(requested_workers), task_count)


def build_cache_record(
    *,
    idx: int,
    entry: dict[str, Any],
    raw_response: str,
    pred_choice: str,
    brief_reason: str,
) -> dict[str, Any]:
    gold_choice = entry["answer"]
    return {
        "index": idx,
        "question": entry["question"],
        "choices": entry["choices"],
        "pmid": entry["pmid"],
        "gold_answer": gold_choice,
        "gold_choice_text": entry["choices"].get(gold_choice),
        "pred_answer": pred_choice,
        "pred_choice_text": entry["choices"].get(pred_choice),
        "brief_reason": brief_reason,
        "raw_response": raw_response,
        "correct": pred_choice == gold_choice,
    }


def run_baseline_question(
    *,
    idx: int,
    entry: dict[str, Any],
    model: str,
    sleep_seconds: float,
) -> tuple[int, dict[str, Any]]:
    client, _provider = build_client()
    prompt = format_question(entry)
    valid_choices = set(entry["choices"].keys())
    raw_response, pred_choice, brief_reason = query_model(
        client=client,
        model=model,
        prompt=prompt,
        valid_choices=valid_choices,
    )

    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    return (
        idx,
        build_cache_record(
            idx=idx,
            entry=entry,
            raw_response=raw_response,
            pred_choice=pred_choice,
            brief_reason=brief_reason,
        ),
    )


def main() -> int:
    args = parse_args()
    load_dotenv_if_present(PROJECT_ROOT / ".env")
    base_dir = PROJECT_ROOT
    loaded_env_files = bootstrap_environment(base_dir)

    dataset_path = resolve_path(args.dataset, base_dir)
    output_dir = resolve_path(args.output_dir, base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_json(dataset_path)
    target_indices = build_target_indices(len(dataset), args.start_index, args.max_questions)

    _, provider = build_client()
    resolved_model = resolve_default_model(args.model)
    model_slug = sanitize_filename(resolved_model)

    cache_path = output_dir / f"{model_slug}_riskqa_baseline_results.json"
    answers_path = output_dir / f"{model_slug}_riskqa_baseline_answers.json"
    comparison_path = output_dir / f"{model_slug}_riskqa_baseline_comparison.json"
    summary_path = output_dir / f"{model_slug}_riskqa_baseline_summary.json"

    if cache_path.exists() and not args.overwrite:
        cache: dict[str, Any] = load_json(cache_path)
    else:
        cache = {}

    total = len(target_indices)
    pending_indices = [idx for idx in target_indices if str(idx) not in cache]
    worker_count = compute_worker_count(requested_workers=args.workers, task_count=len(pending_indices))
    position_lookup = {idx: position for position, idx in enumerate(target_indices, start=1)}

    if pending_indices:
        futures: dict[Any, int] = {}
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for idx in pending_indices:
                futures[executor.submit(
                    run_baseline_question,
                    idx=idx,
                    entry=dataset[idx],
                    model=resolved_model,
                    sleep_seconds=args.sleep_seconds,
                )] = idx

            for future in as_completed(futures):
                idx, record = future.result()
                idx_key = str(idx)
                cache[idx_key] = record
                dump_json(cache_path, cache)
                print(
                    f"[{position_lookup[idx]}/{total}] question {idx}: "
                    f"pred={record['pred_answer']} gold={record['gold_answer']}"
                )

    compatibility_answers: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []

    for idx in target_indices:
        idx_key = str(idx)
        entry = cache.get(idx_key)
        if entry is None:
            continue

        pred_answer = entry["pred_answer"]
        brief_reason = entry["brief_reason"]
        compatibility_answers[idx_key] = [
            f"{pred_answer}. {brief_reason}",
            entry["raw_response"],
        ]

        comparison_rows.append(
            {
                "index": idx,
                "pmid": entry["pmid"],
                "question": entry["question"],
                "choices": entry["choices"],
                "gold_answer": entry["gold_answer"],
                "gold_choice_text": entry["gold_choice_text"],
                "pred_answer": entry["pred_answer"],
                "pred_choice_text": entry["pred_choice_text"],
                "brief_reason": entry["brief_reason"],
                "correct": entry["correct"],
            }
        )

    accuracy = compute_accuracy(comparison_rows)
    compatibility_preds = [extract_answer(compatibility_answers[str(idx)][0]) for idx in target_indices if str(idx) in compatibility_answers]
    compatibility_golds = [cache[str(idx)]["gold_answer"] for idx in target_indices if str(idx) in cache]

    summary = {
        "provider": provider,
        "model": resolved_model,
        "loaded_env_files": loaded_env_files,
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "requested_workers": args.workers,
        "effective_workers": worker_count,
        "num_questions": len(comparison_rows),
        "num_correct": sum(1 for row in comparison_rows if row["correct"]),
        "num_incorrect": sum(1 for row in comparison_rows if not row["correct"]),
        "accuracy": accuracy,
        "compatibility_accuracy": (
            sum(1 for pred, gold in zip(compatibility_preds, compatibility_golds) if pred == gold) / len(compatibility_golds)
            if compatibility_golds
            else 0.0
        ),
        "answers_path": str(answers_path),
        "comparison_path": str(comparison_path),
        "cache_path": str(cache_path),
        "scoring_hint": f"python get_overall_performance.py {answers_path.name}",
    }

    dump_json(answers_path, compatibility_answers)
    dump_json(comparison_path, comparison_rows)
    dump_json(summary_path, summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
