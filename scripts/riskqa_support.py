from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Mapping, Sequence


_EXPLICIT_CHOICE_PATTERNS = (
    re.compile(r"\bAnswer\s*:\s*([A-Z])\b", re.IGNORECASE),
    re.compile(r"\bFinal\s+choice\s*:\s*([A-Z])\b", re.IGNORECASE),
    re.compile(r"\bSelected\s+option\s*:\s*([A-Z])\b", re.IGNORECASE),
    re.compile(r"\b(?:option|choice)\s*([A-Z])\b", re.IGNORECASE),
    re.compile(r"\b(?:option|choice)\s*[:=-]?\s*([A-Z])\b", re.IGNORECASE),
)

_BLOCK_PATTERN = re.compile(
    r"^=== RISKQA RESULT (?P<task_id>\S+) BEGIN ===\n"
    r"(?P<payload>.*?)\n"
    r"=== RISKQA RESULT (?P=task_id) END ===$",
    re.MULTILINE | re.DOTALL,
)


def make_json_serializable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_json_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_serializable(item) for item in value]
    return repr(value)


def normalize_match_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z]+", " ", str(text or "").lower())).strip()


def normalize_choices(raw_choices: Mapping[str, Any] | None) -> dict[str, str]:
    return {str(key).strip(): str(value).strip() for key, value in dict(raw_choices or {}).items()}


def render_choices(choices: Mapping[str, str]) -> str:
    normalized_choices = normalize_choices(choices)
    ordered_keys = sorted(normalized_choices.keys())
    return "\n".join(f"{key}. {normalized_choices[key]}" for key in ordered_keys)


def build_question_query(entry: Mapping[str, Any]) -> str:
    question = str(entry.get("question") or "").strip()
    choices = normalize_choices(entry.get("choices"))
    if not question:
        raise ValueError("RiskQA entry is missing a question.")
    if not choices:
        raise ValueError("RiskQA entry is missing choices.")

    return "\n".join(
        [
            "Solve the following RiskQA clinical calculation question.",
            "",
            f"Question: {question}",
            "",
            "Choices:",
            render_choices(choices),
            "",
            "Instructions:",
            "1. Use the appropriate medical calculator when needed.",
            "2. Compute the relevant value or interpretation before selecting an option.",
            "3. Explain the computed answer briefly.",
            "4. End your final response with a single line in the exact format: Answer: <LETTER>.",
        ]
    )


def build_task_id(index: int) -> str:
    return f"riskqa-{index:05d}"


def select_entries(
    entries: list[dict[str, Any]],
    *,
    start_index: int = 0,
    end_index: int | None = None,
    limit: int | None = None,
) -> list[tuple[int, dict[str, Any]]]:
    if start_index < 0:
        raise ValueError("--start-index must be >= 0.")
    if end_index is not None and end_index < start_index:
        raise ValueError("--end-index must be >= --start-index.")
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be > 0.")

    indexed_entries = list(enumerate(entries))
    sliced = indexed_entries[start_index:end_index]
    if limit is not None:
        sliced = sliced[:limit]
    return sliced


def load_dataset(path_value: str | Path) -> list[dict[str, Any]]:
    dataset_path = Path(path_value).expanduser().resolve()
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"RiskQA dataset must be a JSON array: {dataset_path}")
    return [dict(item) for item in payload]


def extract_last_assistant_message(messages: Sequence[Mapping[str, Any]] | None) -> str:
    for message in reversed(list(messages or [])):
        if str(message.get("role") or "").strip().lower() != "assistant":
            continue
        content = str(message.get("content") or "").strip()
        if content:
            return content
    return ""


def extract_choice_from_text(text: str, choices: Mapping[str, str]) -> str | None:
    if not text:
        return None

    normalized_choices = normalize_choices(choices)
    available_letters = {key.upper() for key in normalized_choices.keys()}
    stripped_text = str(text).strip()
    if len(stripped_text) == 1:
        single_letter = stripped_text.upper()
        if single_letter in available_letters:
            return single_letter

    for pattern in _EXPLICIT_CHOICE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        letter = str(match.group(1) or "").strip().upper()
        if letter in available_letters:
            return letter

    normalized_text = normalize_match_text(text)
    if not normalized_text:
        return None

    for key, choice_text in normalized_choices.items():
        normalized_choice = normalize_match_text(choice_text)
        if normalized_choice and normalized_choice in normalized_text:
            return key.upper()

    best_letter = None
    best_score = 0.0
    text_tokens = set(normalized_text.split())
    for key, choice_text in normalized_choices.items():
        normalized_choice = normalize_match_text(choice_text)
        if not normalized_choice:
            continue
        choice_tokens = set(normalized_choice.split())
        overlap = 0.0
        if text_tokens and choice_tokens:
            overlap = len(text_tokens.intersection(choice_tokens)) / len(choice_tokens)
        score = max(
            overlap,
            SequenceMatcher(None, normalized_choice, normalized_text).ratio(),
        )
        if score > best_score:
            best_score = score
            best_letter = key.upper()

    if best_letter is not None and best_score >= 0.85:
        return best_letter
    return None


def format_record_block(record: Mapping[str, Any]) -> str:
    task_id = str(record.get("task_id") or "").strip()
    if not task_id:
        raise ValueError("Record is missing task_id.")

    payload = json.dumps(
        make_json_serializable(dict(record)),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return (
        f"=== RISKQA RESULT {task_id} BEGIN ===\n"
        f"{payload}\n"
        f"=== RISKQA RESULT {task_id} END ===\n"
    )


def append_record_block(path: str | Path, record: Mapping[str, Any]) -> None:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(format_record_block(record))


def parse_record_blocks(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for match in _BLOCK_PATTERN.finditer(str(text or "")):
        payload = json.loads(match.group("payload"))
        if isinstance(payload, dict):
            records.append(dict(payload))
    return records


def load_record_blocks(path: str | Path) -> list[dict[str, Any]]:
    record_path = Path(path).expanduser().resolve()
    if not record_path.exists():
        raise FileNotFoundError(f"Results text file not found: {record_path}")
    return parse_record_blocks(record_path.read_text(encoding="utf-8"))
