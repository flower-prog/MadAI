from __future__ import annotations

import ast
import contextlib
import inspect
import io
import json
import os
import re
import traceback
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Protocol


_MISSING = object()


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    required: bool = False
    state_fields: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(slots=True)
class BoundTool:
    definition: ToolDefinition
    handler: Callable[..., Any]

    @property
    def provider(self) -> Any:
        return getattr(self.handler, "__self__", None)

    @property
    def name(self) -> str:
        return self.definition.name

    @property
    def description(self) -> str:
        return self.definition.description

    @property
    def input_schema(self) -> dict[str, Any]:
        return dict(self.definition.input_schema)

    def invoke(self, state: Any = None, /, **kwargs: Any) -> Any:
        resolved_kwargs = dict(kwargs)
        if state is not None:
            state_kwargs = self._extract_state_kwargs(state)
            for key, value in state_kwargs.items():
                resolved_kwargs.setdefault(key, value)
        return self.handler(**resolved_kwargs)

    __call__ = invoke

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.definition.name,
            "description": self.definition.description,
            "input_schema": dict(self.definition.input_schema),
            "required": bool(self.definition.required),
        }

    def __getattr__(self, name: str) -> Any:
        provider = self.provider
        if provider is not None and hasattr(provider, name):
            return getattr(provider, name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def _extract_state_kwargs(self, state: Any) -> dict[str, Any]:
        if not self.definition.state_fields:
            if isinstance(state, dict):
                return dict(state)
            return {}

        resolved: dict[str, Any] = {}
        for parameter_name, candidate_paths in self.definition.state_fields.items():
            for candidate_path in candidate_paths:
                value = _resolve_path(state, candidate_path)
                if value is _MISSING:
                    continue
                resolved[parameter_name] = value
                break
        return resolved


def tool(
    *,
    name: str,
    description: str,
    input_schema: dict[str, Any] | None = None,
    required: bool = False,
    state_fields: dict[str, tuple[str, ...] | list[str] | str] | None = None,
):
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        normalized_state_fields = {
            str(param_name): _normalize_paths(paths)
            for param_name, paths in dict(state_fields or {}).items()
        }
        definition = ToolDefinition(
            name=str(name),
            description=str(description),
            input_schema=dict(input_schema or {}),
            required=bool(required),
            state_fields=normalized_state_fields,
        )
        setattr(func, "__tool_definition__", definition)
        return func

    return decorator


def collect_tools(*providers: Any) -> dict[str, BoundTool]:
    collected: dict[str, BoundTool] = {}
    for provider in providers:
        if provider is None:
            continue
        for attribute_name, member in _iter_tool_members(provider):
            definition = get_tool_definition(member)
            if definition is None:
                continue
            bound_handler = getattr(provider, attribute_name, None)
            if not callable(bound_handler):
                continue
            collected[definition.name] = BoundTool(definition=definition, handler=bound_handler)
    return collected


def export_tool_specs(*providers: Any) -> list[Any]:
    from agent.graph.types import ToolSpec

    exported: dict[str, ToolSpec] = {}
    for provider in providers:
        if provider is None:
            continue
        for _attribute_name, member in _iter_tool_members(provider):
            definition = get_tool_definition(member)
            if definition is None:
                continue
            exported[definition.name] = ToolSpec(
                name=definition.name,
                description=definition.description,
                required=definition.required,
                input_schema=dict(definition.input_schema),
            )
    return list(exported.values())


def get_tool_definition(candidate: Any) -> ToolDefinition | None:
    target = getattr(candidate, "__func__", candidate)
    definition = getattr(target, "__tool_definition__", None)
    if isinstance(definition, ToolDefinition):
        return definition
    return None


def _normalize_paths(paths: tuple[str, ...] | list[str] | str) -> tuple[str, ...]:
    if isinstance(paths, str):
        return (paths,)
    return tuple(str(path) for path in list(paths))


def _iter_tool_members(provider: Any):
    owner = provider if inspect.isclass(provider) else type(provider)
    return inspect.getmembers_static(owner)


def _resolve_path(payload: Any, path: str) -> Any:
    current = payload
    for part in str(path).split("."):
        if isinstance(current, dict):
            if part not in current:
                return _MISSING
            current = current[part]
            continue
        if not hasattr(current, part):
            return _MISSING
        current = getattr(current, part)
    return current


class ChatClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> str: ...


def _first_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip():
            return str(value)
    return default


class OpenAIChatClient:
    def __init__(
        self,
        *,
        model: str | None = None,
        azure_endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.default_model = model or _first_env(
            "CLINICAL_TOOL_AGENT_MODEL",
            "CODING_MODEL",
            "BASIC_MODEL",
            "OPENAI_MODEL",
            default="gpt-4o-mini",
        )
        self.azure_endpoint = azure_endpoint or _first_env("OPENAI_ENDPOINT", "AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key or _first_env("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY")
        self.api_version = api_version or _first_env(
            "AZURE_OPENAI_API_VERSION",
            default="2024-12-01-preview",
        )
        self.base_url = base_url or _first_env("OPENAI_BASE_URL")
        self.default_headers = {"User-Agent": "OpenAI/Python 2.16.0"}
        self._client = self._build_client()

    def _build_client(self):
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY or AZURE_OPENAI_API_KEY is required for the clinical tool agent.")

        try:
            if self.azure_endpoint:
                from openai import AzureOpenAI

                return AzureOpenAI(
                    azure_endpoint=self.azure_endpoint,
                    api_key=self.api_key,
                    api_version=self.api_version,
                    default_headers=self.default_headers,
                )

            from openai import OpenAI

            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            client_kwargs["default_headers"] = self.default_headers
            return OpenAI(**client_kwargs)
        except Exception as exc:
            detail = str(exc).strip()
            if detail:
                detail = f"{type(exc).__name__}: {detail}"
            else:
                detail = type(exc).__name__
            raise RuntimeError(
                "Failed to initialize OpenAI client for the clinical tool agent. "
                f"Root cause: {detail}"
            ) from exc

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        response = self._client.chat.completions.create(
            model=model or self.default_model,
            messages=messages,
            temperature=temperature,
        )
        return _extract_chat_completion_text(response)


def _extract_chat_completion_text(response: Any) -> str:
    """Coerce OpenAI-compatible chat responses into plain text.

    Some OpenAI-compatible gateways return a native SDK object, while others
    return a plain string, a JSON-like dict, or even an SSE event-stream blob.
    The workflow only needs the assistant text, so this helper accepts all
    common shapes.
    """

    def _try_extract_sse_text(raw_text: str) -> str | None:
        if not isinstance(raw_text, str):
            return None

        data_lines: list[str] = []
        for raw_line in raw_text.splitlines():
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            data_lines.append(line[5:].strip())

        if not data_lines:
            return None

        extracted_parts: list[str] = []
        fallback_parts: list[str] = []
        saw_payload = False

        for payload_text in data_lines:
            if not payload_text:
                continue
            if payload_text == "[DONE]":
                saw_payload = True
                continue

            saw_payload = True
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                fallback_parts.append(payload_text)
                continue

            piece = _coerce(payload).strip()
            if piece:
                extracted_parts.append(piece)

        if extracted_parts:
            return "".join(extracted_parts).strip()
        if fallback_parts:
            return "".join(fallback_parts).strip()
        if saw_payload:
            return ""
        return None

    def _coerce(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            sse_text = _try_extract_sse_text(value)
            return value if sse_text is None else sse_text
        if isinstance(value, bytes):
            decoded = value.decode("utf-8", errors="ignore")
            sse_text = _try_extract_sse_text(decoded)
            return decoded if sse_text is None else sse_text
        if isinstance(value, list):
            parts = [_coerce(item).strip() for item in value]
            return "\n".join(part for part in parts if part).strip()
        if isinstance(value, dict):
            if "choices" in value:
                return _coerce(value.get("choices"))
            if "message" in value:
                return _coerce(value.get("message"))
            if "delta" in value:
                return _coerce(value.get("delta"))
            if "content" in value:
                return _coerce(value.get("content"))
            if "text" in value:
                return _coerce(value.get("text"))
            if "output_text" in value:
                return _coerce(value.get("output_text"))
            if value.get("type") == "text" and "text" in value:
                return _coerce(value.get("text"))
            return ""

        choices = getattr(value, "choices", None)
        if choices is not None:
            return _coerce(choices)

        output_text = getattr(value, "output_text", None)
        if output_text is not None:
            return _coerce(output_text)

        message = getattr(value, "message", None)
        if message is not None:
            return _coerce(message)

        delta = getattr(value, "delta", None)
        if delta is not None:
            return _coerce(delta)

        content = getattr(value, "content", None)
        if content is not None:
            return _coerce(content)

        text = getattr(value, "text", None)
        if text is not None:
            return _coerce(text)

        return str(value)

    extracted = _coerce(response).strip()
    return extracted


def build_default_chat_client(model: str | None = None) -> ChatClient:
    return OpenAIChatClient(model=model)


_JSON_CODE_BLOCK_RE = re.compile(r"```json\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"(\{[\s\S]*\}|\[[\s\S]*\])")


def maybe_load_json(text: str) -> Any:
    if not text:
        return None

    for match in _JSON_CODE_BLOCK_RE.findall(text):
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    direct = text.strip()
    if direct:
        try:
            return json.loads(direct)
        except json.JSONDecodeError:
            pass

    match = _JSON_OBJECT_RE.search(text)
    if not match:
        return None

    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


_PYTHON_BLOCK_PATTERN = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_PROSE_RULE_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*•]|\d+\.)\s*(.+?)\s*$")
_PROSE_UNIFORM_POINTS_RE = re.compile(
    r"Each\b.*?\bassigned\b.*?(?P<points>-?\d+(?:\.\d+)?)\s*(?:points?)?\s*(?:if|when)\b",
    re.IGNORECASE | re.DOTALL,
)
_PROSE_EXPLICIT_POINTS_PATTERNS = (
    re.compile(r"^(?P<criterion>.+?)\s*:\s*(?P<points>-?\d+(?:\.\d+)?)\s*points?\s*$", re.IGNORECASE),
    re.compile(r"^(?P<criterion>.+?)\s*\(\s*(?P<points>-?\d+(?:\.\d+)?)\s*points?\s*\)\s*$", re.IGNORECASE),
)
_PROSE_SYMBOL_COMPARISON_RE = re.compile(
    r"^(?P<label>.+?)\s*(?P<op>>=|<=|>|<|=|==|≥|≤)\s*(?P<value>-?\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)
_PROSE_TRAILING_COMPARISON_RE = re.compile(
    r"^(?P<label>.+?)\s+(?P<value>-?\d+(?:\.\d+)?)"
    r"(?:\s*(?:years?|yrs?|yr|mg/?dl|g/?dl|mg/?l|mmol/?l|cm|mmhg|%)\b)?"
    r"\s+(?P<phrase>or older|or more|or greater)\s*$",
    re.IGNORECASE,
)
_PROSE_PHRASE_COMPARISON_PATTERNS = (
    (re.compile(r"^(?P<label>.+?)\s+at least\s+(?P<value>-?\d+(?:\.\d+)?)\b", re.IGNORECASE), ">="),
    (
        re.compile(r"^(?P<label>.+?)\s+(?:greater than or equal to|more than or equal to)\s+(?P<value>-?\d+(?:\.\d+)?)\b", re.IGNORECASE),
        ">=",
    ),
    (re.compile(r"^(?P<label>.+?)\s+(?:less than or equal to)\s+(?P<value>-?\d+(?:\.\d+)?)\b", re.IGNORECASE), "<="),
    (re.compile(r"^(?P<label>.+?)\s+(?:less than|below|under)\s+(?P<value>-?\d+(?:\.\d+)?)\b", re.IGNORECASE), "<"),
    (re.compile(r"^(?P<label>.+?)\s+(?:more than|greater than|over)\s+(?P<value>-?\d+(?:\.\d+)?)\b", re.IGNORECASE), ">"),
)


def extract_python_code_blocks(text: str) -> str:
    """Return a single executable script from all fenced Python blocks in text."""
    matches = _PYTHON_BLOCK_PATTERN.findall(text or "")
    cleaned = [match.strip() for match in matches if match.strip()]
    return "\n\n".join(cleaned)


def execute_python_code(code: str, globals_dict: dict[str, Any] | None = None) -> str:
    """Execute Python code and capture stdout, stderr, and tracebacks."""
    namespace: dict[str, Any] = {}
    if globals_dict:
        namespace.update(globals_dict)

    with io.StringIO() as buffer:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            try:
                exec(code, namespace)
            except Exception:
                traceback.print_exc()
        return buffer.getvalue()


def json_ready(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)  # type: ignore[arg-type]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, set):
        return [json_ready(item) for item in sorted(value, key=repr)]
    return value


def summarize_text(text: str, *, max_chars: int = 280) -> dict[str, Any]:
    normalized = str(text or "").strip()
    excerpt = normalized
    if len(excerpt) > max_chars:
        excerpt = excerpt[: max_chars - 3].rstrip() + "..."
    return {
        "char_count": len(normalized),
        "excerpt": excerpt,
    }


def _normalize_bool_like(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value or "").strip().lower()
    if not text:
        return False
    if text in {"yes", "y", "true", "1", "present", "positive", "pos", "male", "m"}:
        return True
    if text in {"no", "n", "false", "0", "absent", "negative", "neg", "female", "f"}:
        return False
    return True


def _normalize_numeric_like(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _compare_numeric(value: Any, operator: str, threshold: float) -> bool:
    numeric_value = _normalize_numeric_like(value)
    if numeric_value is None:
        return False

    normalized_operator = str(operator or "").strip()
    if normalized_operator == ">":
        return numeric_value > threshold
    if normalized_operator == ">=":
        return numeric_value >= threshold
    if normalized_operator == "<":
        return numeric_value < threshold
    if normalized_operator == "<=":
        return numeric_value <= threshold
    if normalized_operator in {"=", "=="}:
        return numeric_value == threshold
    return False


def _matches_male_like(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"male", "m", "man", "boy"}:
            return True
        if lowered in {"female", "f", "woman", "girl"}:
            return False
    return _normalize_bool_like(value)


def _matches_non_white_ethnicity_like(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"white", "caucasian"}:
            return False
        if lowered in {"unknown", "declined", "not reported", "unreported"}:
            return False
        return bool(lowered)
    return _normalize_bool_like(value)


def _canonicalize_prose_parameter_name(raw_text: str) -> str:
    try:
        from .retrieval_tools import _canonicalize_parameter_candidate

        normalized = str(_canonicalize_parameter_candidate(raw_text) or "").strip()
        if normalized:
            return normalized
    except Exception:
        pass

    fallback = re.sub(r"[^a-z0-9]+", "_", str(raw_text or "").strip().lower()).strip("_")
    return re.sub(r"_+", "_", fallback)


def _normalize_python_identifier(value: str, *, default_prefix: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value or "").strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        normalized = default_prefix
    if normalized[0].isdigit():
        normalized = f"{default_prefix}_{normalized}"
    return normalized.lower()


@dataclass(frozen=True, slots=True)
class _CompiledProseCriterion:
    parameter_name: str
    condition_expr: str
    points: float


def _extract_prose_rule_items(computation: str) -> list[str]:
    items: list[str] = []
    for raw_line in str(computation or "").splitlines():
        match = _PROSE_RULE_LIST_ITEM_RE.match(raw_line)
        if not match:
            continue
        item = str(match.group(1) or "").strip()
        if item:
            items.append(item)
    return items


def _extract_uniform_prose_points(computation: str) -> float | None:
    match = _PROSE_UNIFORM_POINTS_RE.search(str(computation or ""))
    if not match:
        return None
    try:
        return float(match.group("points"))
    except (TypeError, ValueError):
        return None


def _split_prose_rule_and_points(raw_item: str, *, default_points: float | None) -> tuple[str, float] | None:
    text = str(raw_item or "").strip().rstrip(".")
    if not text:
        return None

    for pattern in _PROSE_EXPLICIT_POINTS_PATTERNS:
        match = pattern.match(text)
        if not match:
            continue
        try:
            return str(match.group("criterion") or "").strip(), float(match.group("points"))
        except (TypeError, ValueError):
            return None

    if default_points is None:
        return None
    return text, float(default_points)


def _build_prose_condition_expr(parameter_name: str, raw_criterion: str) -> str | None:
    text = str(raw_criterion or "").strip().replace("≥", ">=").replace("≤", "<=").replace("µ", "u").replace("μ", "u")
    if not text:
        return None

    symbol_match = _PROSE_SYMBOL_COMPARISON_RE.match(text)
    if symbol_match:
        operator = symbol_match.group("op").replace("≥", ">=").replace("≤", "<=")
        return f"_compare_numeric({parameter_name}, {operator!r}, {float(symbol_match.group('value'))!r})"

    trailing_match = _PROSE_TRAILING_COMPARISON_RE.match(text)
    if trailing_match:
        return f"_compare_numeric({parameter_name}, '>=' , {float(trailing_match.group('value'))!r})".replace("'>=' ,", "'>=',")

    for pattern, operator in _PROSE_PHRASE_COMPARISON_PATTERNS:
        match = pattern.match(text)
        if not match:
            continue
        return f"_compare_numeric({parameter_name}, {operator!r}, {float(match.group('value'))!r})"

    lowered = text.lower()
    if "male" in lowered and parameter_name == "sex":
        return f"_matches_male_like({parameter_name})"
    if "non-white" in lowered or "non white" in lowered:
        return f"_matches_non_white_ethnicity_like({parameter_name})"
    return f"_normalize_bool_like({parameter_name})"


def _compile_prose_only_registration(calc_id: str, payload: Any) -> RiskCalcRegistration | None:
    computation = str(getattr(payload, "computation", "") or "")
    if not computation.strip():
        return None

    rule_items = _extract_prose_rule_items(computation)
    if not rule_items:
        return None

    uniform_points = _extract_uniform_prose_points(computation)
    compiled_rules: list[_CompiledProseCriterion] = []
    parameter_names: list[str] = []
    seen_parameters: set[str] = set()

    for raw_item in rule_items:
        split_rule = _split_prose_rule_and_points(raw_item, default_points=uniform_points)
        if split_rule is None:
            return None

        criterion_text, points = split_rule
        parameter_name = _normalize_python_identifier(
            _canonicalize_prose_parameter_name(criterion_text),
            default_prefix="feature",
        )
        condition_expr = _build_prose_condition_expr(parameter_name, criterion_text)
        if not condition_expr:
            return None

        compiled_rules.append(
            _CompiledProseCriterion(
                parameter_name=parameter_name,
                condition_expr=condition_expr,
                points=float(points),
            )
        )
        if parameter_name not in seen_parameters:
            seen_parameters.add(parameter_name)
            parameter_names.append(parameter_name)

    if not compiled_rules or not parameter_names:
        return None

    function_name = _normalize_python_identifier(
        f"compute_{getattr(payload, 'title', '') or calc_id}",
        default_prefix="compute_score",
    )
    parameter_signature = ", ".join(parameter_names)
    body_lines = ["score = 0"]
    for rule in compiled_rules:
        body_lines.append(f"if {rule.condition_expr}:")
        body_lines.append(f"    score += {rule.points!r}")
    body_lines.append("return int(score) if float(score).is_integer() else score")
    code = "\n".join(
        [f"def {function_name}({parameter_signature}):"]
        + [f"    {line}" for line in body_lines]
    )

    return RiskCalcRegistration(
        calc_id=str(calc_id),
        title=str(getattr(payload, "title", "") or "").strip(),
        function_name=function_name,
        parameter_names=parameter_names,
        code=code.strip(),
        purpose=str(getattr(payload, "purpose", "") or "").strip(),
        eligibility=str(getattr(payload, "eligibility", "") or "").strip(),
        interpretation=str(getattr(payload, "interpretation", "") or "").strip(),
        utility=str(getattr(payload, "utility", "") or "").strip(),
    )


def build_tool_call(
    tool_name: str,
    *,
    status: str = "completed",
    input_payload: Any = None,
    output_payload: Any = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry = {
        "tool_name": tool_name,
        "status": status,
        "input": json_ready(input_payload),
        "output": json_ready(output_payload),
    }
    if metadata:
        entry["metadata"] = json_ready(metadata)
    return entry


def build_agent_trace(
    agent_name: str,
    *,
    status: str,
    summary: str,
    output_payload: Any = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "agent_name": agent_name,
        "status": status,
        "summary": summary,
        "output": json_ready(output_payload),
        "tool_calls": [json_ready(item) for item in list(tool_calls or [])],
    }


def append_agent_trace(final_output: dict[str, Any], entry: dict[str, Any]) -> None:
    trace = dict(final_output.get("execution_trace") or {})
    agents = list(trace.get("agents") or [])
    tool_calls = list(trace.get("tool_calls") or [])

    normalized_entry = json_ready(entry)
    agents.append(normalized_entry)
    for item in normalized_entry.get("tool_calls") or []:
        flat_item = dict(item)
        flat_item.setdefault("agent_name", normalized_entry.get("agent_name"))
        tool_calls.append(flat_item)

    trace["agents"] = agents
    trace["tool_calls"] = tool_calls
    final_output["execution_trace"] = trace


# Agent-level healthy defaults are intentionally limited to parameters whose
# "healthy" or non-pathologic value is stable enough to use as a fallback.
AGENT_HEALTHY_DEFAULTS: dict[str, Any] = {
    "albumin": 4.0,
    "altered_mental_status": False,
    "anaemia": False,
    "bilirubin": 0.7,
    "bun": 12.0,
    "confusion": False,
    "creatinine": 0.9,
    "diastolic_bp": 80,
    "ejection_fraction": 60.0,
    "elevated_bun": False,
    "fever": False,
    "gcs": 15,
    "heart_rate": 72,
    "high_pulse": False,
    "hypoalbuminemia": False,
    "hypotension": False,
    "inr": 1.0,
    "lactate": 1.0,
    "ldh": 180.0,
    "low_blood_pressure": False,
    "pain": False,
    "platelet_count": 250,
    "platelets": 250,
    "pulse": 72,
    "respiratory_rate": 16,
    "sbp": 120,
    "systolic_bp": 120,
    "temp": 36.8,
    "temperature": 36.8,
    "triglycerides": 1.0,
    "urea": 5.0,
}


def discover_healthy_defaults_path(search_root: str | Path | None = None) -> Path | None:
    root = Path(search_root) if search_root else Path(__file__).resolve().parents[2]
    candidates = [
        root / "data" / "riskcalc_healthy_defaults.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_calculator_healthy_defaults(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    normalized: dict[str, dict[str, Any]] = {}
    for calc_id, defaults in dict(payload or {}).items():
        if not isinstance(defaults, dict):
            continue
        normalized[str(calc_id)] = {str(name): value for name, value in defaults.items()}
    return normalized


def resolve_healthy_default(
    parameter_name: str,
    *,
    calculator_defaults: dict[str, Any] | None = None,
) -> Any | None:
    normalized_name = str(parameter_name or "").strip()
    if not normalized_name:
        return None

    if calculator_defaults:
        if normalized_name in calculator_defaults:
            return calculator_defaults[normalized_name]
        lowered_defaults = {str(name).lower(): value for name, value in calculator_defaults.items()}
        if normalized_name.lower() in lowered_defaults:
            return lowered_defaults[normalized_name.lower()]

    if normalized_name in AGENT_HEALTHY_DEFAULTS:
        return AGENT_HEALTHY_DEFAULTS[normalized_name]
    return AGENT_HEALTHY_DEFAULTS.get(normalized_name.lower())


@dataclass(slots=True)
class RiskCalcRegistration:
    calc_id: str
    title: str
    function_name: str
    parameter_names: list[str]
    code: str
    purpose: str = ""
    eligibility: str = ""
    interpretation: str = ""
    utility: str = ""
    healthy_defaults: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_primary_function(code: str) -> tuple[str, list[str]]:
    tree = ast.parse(code)
    function_nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if not function_nodes:
        raise ValueError("No top-level function found in calculator code.")

    primary = function_nodes[0]
    parameter_names = [arg.arg for arg in primary.args.args]
    return primary.name, parameter_names


def _build_registration(calc_id: str, payload: Any) -> RiskCalcRegistration:
    code = extract_python_code_blocks(str(getattr(payload, "computation", "") or ""))
    if not code.strip():
        compiled_registration = _compile_prose_only_registration(calc_id, payload)
        if compiled_registration is not None:
            return compiled_registration
        raise ValueError(f"Calculator {calc_id} does not contain executable Python code.")

    function_name, parameter_names = _extract_primary_function(code)
    return RiskCalcRegistration(
        calc_id=str(calc_id),
        title=str(getattr(payload, "title", "") or "").strip(),
        function_name=function_name,
        parameter_names=parameter_names,
        code=code.strip(),
        purpose=str(getattr(payload, "purpose", "") or "").strip(),
        eligibility=str(getattr(payload, "eligibility", "") or "").strip(),
        interpretation=str(getattr(payload, "interpretation", "") or "").strip(),
        utility=str(getattr(payload, "utility", "") or "").strip(),
    )


class RiskCalcRegistry:
    def __init__(self, registrations: dict[str, RiskCalcRegistration]) -> None:
        self._registrations = dict(registrations)

    @classmethod
    def from_catalog(cls, catalog: "RiskCalcCatalog") -> "RiskCalcRegistry":
        return cls.from_catalog_with_defaults(catalog)

    @classmethod
    def from_catalog_with_defaults(
        cls,
        catalog: "RiskCalcCatalog",
        *,
        healthy_defaults_by_calc: dict[str, dict[str, Any]] | None = None,
        healthy_defaults_path: str | Path | None = None,
    ) -> "RiskCalcRegistry":
        if healthy_defaults_by_calc is None and healthy_defaults_path is not None:
            healthy_defaults_by_calc = load_calculator_healthy_defaults(healthy_defaults_path)

        registrations: dict[str, RiskCalcRegistration] = {}
        for document in catalog.documents():
            try:
                registration = _build_registration(document.pmid, document)
            except Exception:
                continue
            calculator_defaults = dict((healthy_defaults_by_calc or {}).get(registration.calc_id) or {})
            if calculator_defaults:
                registration.healthy_defaults = calculator_defaults
            registrations[registration.calc_id] = registration
        return cls(registrations)

    @classmethod
    def from_paths(
        cls,
        riskcalcs_path: str | Path,
        pmid_metadata_path: str | Path,
    ) -> "RiskCalcRegistry":
        from .retrieval_tools import RiskCalcCatalog

        catalog = RiskCalcCatalog.from_paths(riskcalcs_path, pmid_metadata_path)
        return cls.from_catalog(catalog)

    def __len__(self) -> int:
        return len(self._registrations)

    def get(self, calc_id: str) -> RiskCalcRegistration:
        return self._registrations[str(calc_id)]

    def has(self, calc_id: str) -> bool:
        return str(calc_id) in self._registrations

    def registrations(self) -> list[RiskCalcRegistration]:
        return list(self._registrations.values())

    def to_manifest(self) -> dict[str, dict[str, Any]]:
        return {
            calc_id: registration.to_dict()
            for calc_id, registration in sorted(self._registrations.items())
        }

    def export_manifest(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(
            json.dumps(self.to_manifest(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return target


class RiskCalcExecutor:
    def __init__(self, registry: RiskCalcRegistry) -> None:
        self.registry = registry
        self._compiled_functions: dict[str, Callable[..., Any]] = {}

    def get_parameter_names(self, calc_id: str) -> list[str]:
        registration = self.registry.get(calc_id)
        return list(registration.parameter_names)

    def _compile(self, calc_id: str) -> Callable[..., Any]:
        normalized_id = str(calc_id)
        cached = self._compiled_functions.get(normalized_id)
        if cached is not None:
            return cached

        registration = self.registry.get(normalized_id)
        namespace: dict[str, Any] = {
            "_normalize_bool_like": _normalize_bool_like,
            "_compare_numeric": _compare_numeric,
            "_matches_male_like": _matches_male_like,
            "_matches_non_white_ethnicity_like": _matches_non_white_ethnicity_like,
        }
        exec(registration.code, namespace)

        function = namespace.get(registration.function_name)
        if not callable(function):
            raise ValueError(
                f"Compiled calculator {normalized_id} is missing callable {registration.function_name}."
            )

        self._compiled_functions[normalized_id] = function
        return function

    def run(self, calc_id: str, inputs: dict[str, Any]) -> dict[str, Any]:
        registration = self.registry.get(calc_id)
        resolved_inputs = dict(inputs)
        defaults_used: dict[str, Any] = {}
        for name in registration.parameter_names:
            if name in resolved_inputs:
                continue
            default_value = resolve_healthy_default(name, calculator_defaults=registration.healthy_defaults)
            if default_value is None:
                continue
            resolved_inputs[name] = default_value
            defaults_used[name] = default_value

        missing = [name for name in registration.parameter_names if name not in resolved_inputs]
        if missing:
            return {
                "status": "missing_inputs",
                "calc_id": registration.calc_id,
                "title": registration.title,
                "function_name": registration.function_name,
                "missing_inputs": missing,
                "defaults_used": defaults_used,
            }

        function = self._compile(registration.calc_id)
        call_inputs = {name: resolved_inputs[name] for name in registration.parameter_names}
        result = function(**call_inputs)
        return {
            "status": "completed",
            "calc_id": registration.calc_id,
            "title": registration.title,
            "function_name": registration.function_name,
            "inputs": call_inputs,
            "result": result,
            "interpretation": registration.interpretation,
            "defaults_used": defaults_used,
        }


def _normalize_calc_id(calculator: str | dict[str, Any]) -> str:
    """
    将外部传入的 calculator 引用统一整理成标准的计算器 ID。

    这里允许两种输入形式：
    1. 直接传字符串，例如 PMID 或内部使用的 ``calc_id``
    2. 传一个字典对象，只要其中带有 ``pmid`` 或 ``calc_id`` 字段即可

    这个函数只做“提取 + 去空白”的轻量规范化，不负责判断该 ID
    是否真的存在于注册表中。真正的有效性校验由上层方法负责。

    参数:
        calculator: 计算器引用，可以是字符串，也可以是包含
            ``pmid`` / ``calc_id`` 的字典。

    返回:
        规范化后的字符串 ID；如果没有提取到有效值，则返回空字符串。

    小例子:
        >>> _normalize_calc_id(" 123456 ")
        '123456'
        >>> _normalize_calc_id({"pmid": "987654"})
        '987654'
        >>> _normalize_calc_id({"calc_id": "ascvd-2013"})
        'ascvd-2013'
    """
    if isinstance(calculator, dict):
        return str(calculator.get("pmid") or calculator.get("calc_id") or "").strip()
    return str(calculator or "").strip()


class RiskCalcExecutionTool:
    """
    面向 Agent 层的“风险计算器执行工具”。

    这个类把一次完整的 calculator 调用流程串起来：
    1. 根据 ``pmid`` 或 ``calc_id`` 找到注册过的计算器
    2. 从临床文本中抽取计算所需参数
    3. 调用底层执行器进行真实计算
    4. 把结构化结果整理成自然语言答案或摘要

    它本身不实现医学公式，而是负责把“检索到的计算器”真正变成
    一个可执行、可返回结果的工具入口。

    小例子:
        >>> tool = RiskCalcExecutionTool(registry, chat_client=chat_client)
        >>> result = tool.run(
        ...     calculator="123456",
        ...     clinical_text="65岁男性，吸烟，SBP 150，TC 5.2，HDL 1.0",
        ... )
        >>> isinstance(result, dict) or result is None
        True
    """

    def __init__(
        self,
        registry: RiskCalcRegistry,
        *,
        chat_client: ChatClient,
        executor: RiskCalcExecutor | None = None,
    ) -> None:
        """
        初始化执行工具所需的核心依赖。

        参数:
            registry: 风险计算器注册表。负责根据 ``calc_id``/``pmid``
                提供 calculator 的元数据与可执行映射。
            chat_client: 对话模型客户端。主要用于：
                1. 从临床文本里抽取 calculator 参数
                2. 把计算结果整理成最终回复
            executor: 真正执行注册计算器的底层执行器；如果不传，
                会自动基于 registry 创建一个默认执行器。

        小例子:
            >>> tool = RiskCalcExecutionTool(
            ...     registry,
            ...     chat_client=chat_client,
            ... )
            >>> tool.registry is registry
            True
        """
        self.registry = registry
        self.chat_client = chat_client
        self.executor = executor or RiskCalcExecutor(registry)

    def resolve_calc_id(self, calculator: str | dict[str, Any]) -> str:
        """
        解析并校验 calculator 引用，返回可直接用于注册表查询的 ID。

        和 ``_normalize_calc_id`` 的区别是：
        - ``_normalize_calc_id`` 只负责提取字符串
        - ``resolve_calc_id`` 额外保证结果不能为空，否则立刻抛错

        这能让后续流程在一开始就发现输入不合法，而不是把错误拖到更深层。

        参数:
            calculator: 字符串形式的计算器 ID，或者包含 ``pmid`` /
                ``calc_id`` 的字典。

        返回:
            非空的 calculator ID。

        异常:
            ValueError: 当输入里提取不出 ``pmid`` 或 ``calc_id`` 时抛出。

        小例子:
            >>> tool.resolve_calc_id(" 123456 ")
            '123456'
            >>> tool.resolve_calc_id({"pmid": "123456"})
            '123456'
        """
        calc_id = _normalize_calc_id(calculator)
        if not calc_id:
            raise ValueError("Calculator reference must include a pmid or calc_id.")
        return calc_id

    def has_registration(self, calculator: str | dict[str, Any]) -> bool:
        """
        判断给定的 calculator 是否已经注册到系统中。

        这个方法适合在真正执行前做一次快速存在性检查，避免直接调用
        ``get_registration`` 时因为不存在而抛异常。

        参数:
            calculator: 字符串 ID 或带 ``pmid`` / ``calc_id`` 的字典。

        返回:
            ``True`` 表示注册表中存在该 calculator，``False`` 表示不存在。

        小例子:
            >>> tool.has_registration("123456")
            True
            >>> tool.has_registration({"pmid": "not-exist"})
            False
        """
        return self.registry.has(self.resolve_calc_id(calculator))

    def get_registration(self, calculator: str | dict[str, Any]) -> RiskCalcRegistration:
        """
        读取某个 calculator 在注册表中的完整注册信息。

        返回的 ``RiskCalcRegistration`` 一般会包含标题、函数名、参数名、
        解释文本、默认值等执行时需要的元数据。

        参数:
            calculator: 字符串 ID 或带 ``pmid`` / ``calc_id`` 的字典。

        返回:
            对应 calculator 的注册对象。

        小例子:
            >>> registration = tool.get_registration("123456")
            >>> registration.calc_id
            '123456'
        """
        return self.registry.get(self.resolve_calc_id(calculator))

    def describe(self, calculator: str | dict[str, Any]) -> dict[str, Any]:
        """
        把注册对象整理成适合序列化和展示的字典结构。

        这个方法适合给上层 Agent、调试面板、API 响应或日志系统使用，
        因为它返回的是纯字典，而不是内部 dataclass / 对象。

        参数:
            calculator: 需要查询的 calculator 引用。

        返回:
            一个只包含公开元数据的字典，例如：
            ``calc_id``、标题、参数名、用途、适用条件、解释说明、健康默认值。

        小例子:
            >>> info = tool.describe("123456")
            >>> sorted(["calc_id", "title", "parameter_names"]) <= sorted(info.keys())
            True
        """
        registration = self.get_registration(calculator)
        return {
            "calc_id": registration.calc_id,
            "title": registration.title,
            "function_name": registration.function_name,
            "parameter_names": list(registration.parameter_names),
            "purpose": registration.purpose,
            "eligibility": registration.eligibility,
            "interpretation": registration.interpretation,
            "healthy_defaults": dict(registration.healthy_defaults),
        }

    def extract_inputs(
        self,
        *,
        calculator: str | dict[str, Any],
        clinical_text: str,
        structured_case: dict[str, Any] | None = None,
        calculator_payload: dict[str, Any] | None = None,
        llm_model: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """
        利用大模型从临床文本中抽取 calculator 所需的输入参数。

        工作流程大致是：
        1. 从注册表读取 calculator 的参数清单
        2. 组装提示词，把参数名和原始临床文本一起交给 LLM
        3. 要求 LLM 只返回 JSON
        4. 解析 JSON，并且只保留当前 calculator 真正声明过的参数

        这个“只保留注册参数”的步骤很重要，可以避免模型额外编造字段，
        从而让执行阶段更稳定。

        参数:
            calculator: 要执行的 calculator 引用。
            clinical_text: 临床病历文本、问题文本或病例摘要。
            llm_model: 可选，指定抽参时使用的模型名称。
            temperature: 抽参时的采样温度，默认使用更保守的 ``0.0``。

        返回:
            一个字典，里面通常包含：
            - ``calc_id``: 计算器 ID
            - ``title``: 标题
            - ``function_name``: 绑定的执行函数名
            - ``parameter_names``: 参数名列表
            - ``inputs``: 抽取出的参数字典；如果一个都没有，则为 ``None``
            - ``raw_response``: LLM 原始返回文本，便于调试

        小例子:
            >>> extracted = tool.extract_inputs(
            ...     calculator="123456",
            ...     clinical_text="患者 65 岁，男性，收缩压 150 mmHg。",
            ... )
            >>> "inputs" in extracted
            True
        """
        registration = self.get_registration(calculator)
        selected_calculator_payload = dict(registration.to_dict())
        if isinstance(calculator_payload, dict):
            selected_calculator_payload.update(dict(calculator_payload))
        selected_calculator_payload["pmid"] = registration.calc_id
        selected_calculator_payload["calc_id"] = registration.calc_id
        selected_calculator_payload["title"] = (
            str(selected_calculator_payload.get("title") or registration.title).strip()
        )
        selected_calculator_payload["function_name"] = registration.function_name
        selected_calculator_payload["parameter_names"] = list(registration.parameter_names)

        normalized_structured_case = dict(structured_case or {})
        system_prompt = (
            "You are a calculator-specific clinical parameter extraction agent. "
            "You must extract values only for the single selected calculator provided to you. "
            "Use the selected calculator metadata, the structured case JSON, and the raw clinical text together. "
            "Return JSON only in the format {\"inputs\": {\"parameter_name\": value}}. "
            "Only include parameters from the selected calculator's required parameter list. "
            "Do not add extra keys. Do not guess unsupported values."
        )
        prompt = (
            "Extract inputs for the selected calculator.\n\n"
            f"Selected Calculator PMID: {registration.calc_id}\n"
            f"Selected Calculator Title: {registration.title}\n"
            f"Selected Calculator Function: {registration.function_name}\n"
            f"Selected Calculator Required Parameters: {', '.join(registration.parameter_names)}\n\n"
            "Selected Calculator Full Payload:\n"
            f"{json.dumps(selected_calculator_payload, ensure_ascii=False, indent=2)}\n\n"
            "Structured Case JSON:\n"
            f"{json.dumps(normalized_structured_case, ensure_ascii=False, indent=2)}\n\n"
            "Raw Clinical Text:\n"
            f"{clinical_text}\n\n"
            "Return JSON only in the format "
            '{"inputs": {"parameter_name": value}}.'
        )
        answer = self.chat_client.complete(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model=llm_model,
            temperature=temperature,
        )
        payload = maybe_load_json(answer)
        raw_inputs = payload.get("inputs", payload) if isinstance(payload, dict) else None

        normalized_inputs: dict[str, Any] = {}
        if isinstance(raw_inputs, dict):
            for name in registration.parameter_names:
                if name in raw_inputs:
                    normalized_inputs[name] = raw_inputs[name]

        return {
            "calc_id": registration.calc_id,
            "title": registration.title,
            "function_name": registration.function_name,
            "parameter_names": list(registration.parameter_names),
            "inputs": normalized_inputs or None,
            "structured_case": normalized_structured_case,
            "calculator_payload": selected_calculator_payload,
            "raw_response": answer,
        }

    def execute_registered(
        self,
        *,
        calculator: str | dict[str, Any],
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """
        调用底层执行器，真正运行某个已经注册的 calculator。

        这里默认认为 ``inputs`` 已经经过了前置清洗或抽取，因此这个方法
        主要做两件事：
        1. 把 calculator 引用解析成标准 ID
        2. 把输入字典交给 ``RiskCalcExecutor.run``

        参数:
            calculator: 目标 calculator 的引用。
            inputs: 传给计算公式的参数字典。

        返回:
            底层执行器返回的结果字典，通常包含执行状态、实际输入、
            使用到的默认值以及计算结果。

        小例子:
            >>> execution = tool.execute_registered(
            ...     calculator="123456",
            ...     inputs={"age": 65, "sbp": 150},
            ... )
            >>> "status" in execution
            True
        """
        return self.executor.run(self.resolve_calc_id(calculator), dict(inputs or {}))

    def summarize_result(
        self,
        *,
        calculator: str | dict[str, Any],
        clinical_text: str,
        mode: str,
        execution: dict[str, Any],
        llm_model: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, str]:
        """
        把结构化计算结果整理成用户可直接阅读的自然语言文本。

        这个方法不是重新计算，而是“结果解释层”。它会把：
        - 计算器名称
        - 实际使用的输入
        - 自动补上的健康默认值
        - 原始数值结果
        - 注册表中的解释说明
        一起交给 LLM，请模型输出更自然的答案。

        ``mode`` 决定文案目标：
        - ``question``: 更像“回答问题”
        - 其他值: 更像“病例摘要/风险总结”

        参数:
            calculator: 已执行过的 calculator 引用。
            clinical_text: 原始临床文本或问题文本，用于给总结补上下文。
            mode: 输出模式，常见值如 ``question`` 或 ``patient_note``。
            execution: 底层执行结果字典。
            llm_model: 可选，指定总结阶段使用的模型。
            temperature: 总结阶段的采样温度。

        返回:
            一个字典，至少包含：
            - ``final_text``: 给最终用户看的文本
            - ``raw_response``: LLM 原始返回

        小例子:
            >>> summary = tool.summarize_result(
            ...     calculator="123456",
            ...     clinical_text="患者胸痛 2 小时。",
            ...     mode="question",
            ...     execution={"inputs": {"age": 65}, "defaults_used": {}, "result": 0.18},
            ... )
            >>> "final_text" in summary
            True
        """
        registration = self.get_registration(calculator)
        if mode == "question":
            finish_prefix = "Answer: "
            task = "Use the computed calculator result to answer the question."
            prompt_subject = "question"
        else:
            finish_prefix = "Summary: "
            task = "Use the computed calculator result to summarize the patient's risk."
            prompt_subject = "patient information"

        prompt = (
            f"{task}\n\n"
            f"Calculator: {registration.title}\n"
            f"Function: {registration.function_name}\n"
            f"Inputs: {json.dumps(execution.get('inputs') or {}, ensure_ascii=False)}\n"
            f"Healthy Defaults Used: {json.dumps(execution.get('defaults_used') or {}, ensure_ascii=False)}\n"
            f"Raw Result: {execution.get('result')}\n"
            f"Interpretation Reference: {registration.interpretation}\n\n"
            f"{prompt_subject.title()}:\n{clinical_text}\n\n"
            f'Respond with "{finish_prefix}" followed by the final response.'
        )
        answer = self.chat_client.complete(
            [{"role": "user", "content": prompt}],
            model=llm_model,
            temperature=temperature,
        )
        if finish_prefix in answer:
            final_text = answer.split(finish_prefix, 1)[-1].strip()
        else:
            final_text = str(answer).strip() or f"Computed result: {execution.get('result')}"

        return {
            "final_text": final_text,
            "raw_response": answer,
        }

    @tool(
        name="riskcalc_executor",
        description="Execute a registered calculator after a match is selected.",
        input_schema={
            "calculator": "str|dict",
            "clinical_text": "str",
            "structured_case": "dict|None",
            "calculator_payload": "dict|None",
            "mode": "str",
            "llm_model": "str|None",
            "temperature": "float",
        },
        required=False,
    )
    def run(
        self,
        *,
        calculator: str | dict[str, Any],
        clinical_text: str,
        structured_case: dict[str, Any] | None = None,
        calculator_payload: dict[str, Any] | None = None,
        mode: str = "patient_note",
        llm_model: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any] | None:
        """
        端到端执行一个注册好的风险计算器。

        这是整个执行工具最常用的高层入口，内部顺序如下：
        1. 检查 calculator 是否存在
        2. 从临床文本中抽取参数
        3. 如果没有抽出任何参数，则提前返回 ``None``
        4. 调用底层执行器完成公式计算
        5. 如果执行未完成，则返回 ``None``
        6. 调用总结器生成最终自然语言输出

        因此，这个方法适合上层 Agent 直接调用；而更底层的场景则可以分别
        调用 ``extract_inputs``、``execute_registered``、``summarize_result``。

        参数:
            calculator: 目标 calculator 引用。
            clinical_text: 提供给抽参与总结阶段的原始文本。
            mode: 输出模式，默认 ``patient_note``。
            llm_model: 可选，统一指定抽参与总结阶段用到的模型。
            temperature: 抽参与总结阶段统一使用的采样温度。

        返回:
            成功时返回一个完整结果字典，包含：
            ``pmid``、标题、执行状态、函数名、输入、默认值、原始结果、
            最终文本等字段。
            如果任一关键步骤失败，则返回 ``None``。

        小例子:
            >>> result = tool.run(
            ...     calculator={"pmid": "123456"},
            ...     clinical_text="65岁男性，吸烟，收缩压 150。",
            ...     mode="patient_note",
            ... )
            >>> result is None or result["status"] == "completed"
            True
        """
        if not self.has_registration(calculator):
            return None

        registration = self.get_registration(calculator)
        extracted = self.extract_inputs(
            calculator=calculator,
            clinical_text=clinical_text,
            structured_case=structured_case,
            calculator_payload=calculator_payload,
            llm_model=llm_model,
            temperature=temperature,
        )
        extracted_inputs = dict(extracted.get("inputs") or {})
        if not extracted_inputs:
            return None

        execution = self.execute_registered(
            calculator=calculator,
            inputs=extracted_inputs,
        )
        if execution.get("status") != "completed":
            return None

        summary = self.summarize_result(
            calculator=calculator,
            clinical_text=clinical_text,
            mode=mode,
            execution=execution,
            llm_model=llm_model,
            temperature=temperature,
        )
        return {
            "pmid": registration.calc_id,
            "title": registration.title,
            "status": "completed",
            "rounds": 1,
            "execution_mode": "registered",
            "function_name": registration.function_name,
            "inputs": execution.get("inputs") or {},
            "defaults_used": execution.get("defaults_used") or {},
            "result": execution.get("result"),
            "final_text": summary["final_text"],
            "interpretation": registration.interpretation,
            "messages": [],
        }


def create_execution_tool(
    registry: RiskCalcRegistry,
    *,
    chat_client: ChatClient,
    executor: RiskCalcExecutor | None = None,
) -> RiskCalcExecutionTool:
    """
    创建一个标准的执行工具实例，供 Agent 层直接使用。

    这是一个轻量工厂函数，目的是把实例化方式统一起来。这样上层代码
    不需要每次都显式写 ``RiskCalcExecutionTool(...)``。

    参数:
        registry: 风险计算器注册表。
        chat_client: 用于抽参与结果总结的聊天模型客户端。
        executor: 可选的底层执行器；不传时自动创建默认执行器。

    返回:
        一个可直接执行 calculator 的 ``RiskCalcExecutionTool`` 实例。

    小例子:
        >>> tool = create_execution_tool(registry, chat_client=chat_client)
        >>> isinstance(tool, RiskCalcExecutionTool)
        True
    """
    return RiskCalcExecutionTool(
        registry,
        chat_client=chat_client,
        executor=executor,
    )


__all__ = [
    "AGENT_HEALTHY_DEFAULTS",
    "BoundTool",
    "ChatClient",
    "OpenAIChatClient",
    "RiskCalcExecutionTool",
    "RiskCalcExecutor",
    "RiskCalcRegistration",
    "RiskCalcRegistry",
    "ToolDefinition",
    "append_agent_trace",
    "build_agent_trace",
    "build_default_chat_client",
    "build_tool_call",
    "collect_tools",
    "create_execution_tool",
    "discover_healthy_defaults_path",
    "execute_python_code",
    "export_tool_specs",
    "extract_python_code_blocks",
    "get_tool_definition",
    "json_ready",
    "load_calculator_healthy_defaults",
    "maybe_load_json",
    "resolve_healthy_default",
    "summarize_text",
    "tool",
]
