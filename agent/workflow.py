from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    # Support direct execution via `python agent/workflow.py ...` by ensuring the
    # project root is on sys.path before importing the package module.
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from agent.graph import build_graph_with_memory
    from agent.graph.types import GraphState
else:
    from .graph import build_graph_with_memory
    from .graph.types import GraphState


logger = logging.getLogger(__name__)
graph = build_graph_with_memory()
_DEFAULT_RETRIEVAL_TOP_K = 30
DEFAULT_DEMO_TEXT = (
    "78-year-old male with atrial fibrillation, hypertension, diabetes, and prior TIA; "
    "which stroke risk calculator should be used and what would it compute?"
)


def _load_json_payload(raw_value: str | None, *, label: str) -> Any:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for {label}: {exc}") from exc


def _load_json_file(path_value: str | None, *, label: str) -> Any:
    if path_value is None:
        return None
    path = Path(path_value).expanduser().resolve()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Unable to read {label} file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {label} file {path}: {exc}") from exc


def _coalesce_json_sources(
    inline_value: str | None,
    file_value: str | None,
    *,
    label: str,
) -> Any:
    inline_payload = _load_json_payload(inline_value, label=label)
    if inline_payload is not None:
        return inline_payload
    return _load_json_file(file_value, label=label)


def _extract_cli_summary(result: dict[str, Any]) -> dict[str, Any]:
    final_output = dict(result.get("final_output") or {})
    child_result = dict(final_output.get("clinical_tool_agent") or {})
    calculation_bundle = dict(final_output.get("calculation_bundle") or {})
    child_from_bundle = dict(calculation_bundle.get("child_result") or {})
    child_result = child_result or child_from_bundle
    raw_job = result.get("clinical_tool_job")
    if isinstance(raw_job, dict):
        raw_job_mode = raw_job.get("mode")
    else:
        raw_job_mode = getattr(raw_job, "mode", None)

    selected_tool = dict(child_result.get("selected_tool") or {})
    execution = dict(child_result.get("execution") or {})
    executions = [dict(item) for item in list(child_result.get("executions") or []) if isinstance(item, dict)]
    clinical_answer = [
        dict(item) if isinstance(item, dict) else {"value": item}
        for item in list(result.get("clinical_answer") or final_output.get("clinical_answer") or [])
    ]

    summary: dict[str, Any] = {
        "status": result.get("status"),
        "review_passed": result.get("review_passed"),
        "mode": str(
            (
                child_result.get("mode")
                or raw_job_mode
                or dict(result.get("final_output") or {}).get("orchestrator_result", {}).get("mode")
                or ""
            )
        ).strip(),
        "selected_tool": selected_tool or None,
        "execution": execution or None,
        "executions": executions,
        "clinical_answer": clinical_answer,
        "errors": list(result.get("errors") or []),
    }
    if selected_tool:
        summary["selected_tool"] = {
            "pmid": selected_tool.get("pmid"),
            "title": selected_tool.get("title"),
            "purpose": selected_tool.get("purpose"),
            "parameter_names": list(selected_tool.get("parameter_names") or []),
            "reason": selected_tool.get("reason"),
        }
    if execution:
        summary["execution"] = {
            "pmid": execution.get("pmid"),
            "title": execution.get("title"),
            "status": execution.get("status"),
            "function_name": execution.get("function_name"),
            "inputs": dict(execution.get("inputs") or {}),
            "defaults_used": dict(execution.get("defaults_used") or {}),
            "result": execution.get("result"),
            "final_text": execution.get("final_text"),
        }
    if executions:
        normalized_executions = []
        for item in executions:
            normalized_executions.append(
                {
                    "pmid": item.get("pmid"),
                    "title": item.get("title"),
                    "status": item.get("status"),
                    "function_name": item.get("function_name"),
                    "inputs": dict(item.get("inputs") or {}),
                    "defaults_used": dict(item.get("defaults_used") or {}),
                    "result": item.get("result"),
                    "final_text": item.get("final_text"),
                }
            )
        summary["executions"] = normalized_executions
    return summary


def _resolve_retrieval_top_k(top_k: int | None) -> int:
    if top_k is not None:
        return max(int(top_k), 1)

    for env_name in ("MEDAI_RETRIEVAL_TOP_K", "MEDAI_TOP_K"):
        raw_value = str(os.getenv(env_name) or "").strip()
        if not raw_value:
            continue
        try:
            return max(int(raw_value), 1)
        except ValueError as exc:
            raise ValueError(f"{env_name} must be an integer, got {raw_value!r}.") from exc

    return _DEFAULT_RETRIEVAL_TOP_K


def _normalize_optional_mapping(raw_value: Any, *, label: str) -> dict[str, Any] | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, dict):
        return dict(raw_value)
    if is_dataclass(raw_value):
        return {item.name: getattr(raw_value, item.name) for item in fields(raw_value)}
    raise ValueError(f"{label} must be a JSON object / dict, got {type(raw_value).__name__}.")


def _normalize_optional_list(raw_value: Any, *, label: str) -> list[Any] | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, list):
        return list(raw_value)
    if isinstance(raw_value, tuple):
        return list(raw_value)
    if isinstance(raw_value, (str, bytes)):
        raise ValueError(f"{label} must be a JSON array / list, got a string.")
    raise ValueError(f"{label} must be a JSON array / list, got {type(raw_value).__name__}.")


def _normalize_optional_string_list(raw_value: Any, *, label: str) -> list[str] | None:
    items = _normalize_optional_list(raw_value, label=label)
    if items is None:
        return None
    return [str(item).strip() for item in items if str(item).strip()]


def _normalize_optional_mapping_list(raw_value: Any, *, label: str) -> list[dict[str, Any]] | None:
    items = _normalize_optional_list(raw_value, label=label)
    if items is None:
        return None

    normalized_items: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        normalized_items.append(
            _normalize_optional_mapping(
                item,
                label=f"{label}[{index}]",
            )
            or {}
        )
    return normalized_items


def _normalize_optional_existing_path(raw_value: str | None, *, label: str) -> str | None:
    text = str(raw_value or "").strip()
    if not text:
        return None

    path = Path(text).expanduser()
    if not path.exists():
        raise ValueError(f"{label} file not found: {path}")
    return str(path.resolve())


def _configure_logging(debug: bool) -> None:
    if debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        return

    if logging.getLogger().handlers:
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _ensure_workflow_config(config: dict[str, Any] | None) -> dict[str, Any]:
    workflow_config = dict(config or {})
    configurable = dict(workflow_config.get("configurable") or {})
    configurable.setdefault("thread_id", str(uuid.uuid4()))
    workflow_config["configurable"] = configurable
    return workflow_config


def _copy_state_input(state: GraphState | dict[str, Any]) -> dict[str, Any]:
    if isinstance(state, dict):
        return dict(state)
    if is_dataclass(state):
        return {item.name: getattr(state, item.name) for item in fields(state)}
    raise TypeError("`state` must be a GraphState dataclass instance or a dict.")


def _merge_tool_registry(
    state_input: GraphState | dict[str, Any],
    tool_registry: dict[str, Any] | None,
) -> GraphState | dict[str, Any]:
    state_payload = _copy_state_input(state_input)
    if not tool_registry:
        return state_payload

    merged_registry = dict(state_payload.get("tool_registry") or {})
    merged_registry.update(tool_registry)
    state_payload["tool_registry"] = merged_registry
    return state_payload


def _build_initial_state(
    request: str,
    *,
    messages: list[dict[str, str]] | None = None,
    patient_case: dict[str, Any] | None = None,
    clinical_tool_job: dict[str, Any] | None = None,
    tool_registry: dict[str, Any] | None = None,
    deep_thinking_mode: bool = True,
    search_before_planning: bool = False,
    pass_through_expert: bool = False,
    auto_accepted_plan: bool = False,
) -> dict[str, Any]:
    state_messages = list(messages or [])
    if not state_messages:
        state_messages.append({"role": "user", "content": request})

    return {
        "request": request,
        "messages": state_messages,
        "patient_case": patient_case,
        "clinical_tool_job": clinical_tool_job,
        "tool_registry": tool_registry or {},
        "deep_thinking_mode": deep_thinking_mode,
        "search_before_planning": search_before_planning,
        "pass_through_expert": pass_through_expert,
        "auto_accepted_plan": auto_accepted_plan,
    }


def _invoke_graph(
    state_input: GraphState | dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
    debug: bool = False,
    tool_registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _configure_logging(debug)
    workflow_config = _ensure_workflow_config(config)
    graph_input = _merge_tool_registry(state_input, tool_registry)

    logger.info("Starting MedAI workflow")
    result = graph.invoke(input=graph_input, config=workflow_config)
    logger.info("Workflow completed with status=%s", result.get("status"))
    return result


def _build_clinical_tool_job_payload(
    text: str,
    *,
    mode: str = "patient_note",
    structured_case: dict[str, Any] | None = None,
    riskcalcs_path: str | None = None,
    pmid_metadata_path: str | None = None,
    llm_model: str | None = None,
    retriever_backend: str = "hybrid",
    top_k: int | None = None,
    risk_hints: list[str] | None = None,
    retrieval_queries: list[dict[str, Any]] | None = None,
    max_selected_tools: int | None = 5,
    forced_tool_pmid: str | None = None,
) -> dict[str, Any]:
    retrieval_top_k = _resolve_retrieval_top_k(top_k)
    structured_case_payload = _normalize_optional_mapping(structured_case, label="structured_case") or {}
    normalized_risk_hints = _normalize_optional_string_list(risk_hints, label="risk_hints") or []
    normalized_retrieval_queries = (
        _normalize_optional_mapping_list(retrieval_queries, label="retrieval_queries") or []
    )
    normalized_riskcalcs_path = _normalize_optional_existing_path(
        riskcalcs_path,
        label="riskcalcs_path",
    )
    normalized_pmid_metadata_path = _normalize_optional_existing_path(
        pmid_metadata_path,
        label="pmid_metadata_path",
    )
    normalized_llm_model = str(llm_model or "").strip() or None
    normalized_forced_tool_pmid = str(forced_tool_pmid or "").strip() or None
    return {
        "mode": mode,
        "text": text,
        "case_summary": str(structured_case_payload.get("case_summary") or ""),
        "structured_case": structured_case_payload,
        "risk_hints": normalized_risk_hints,
        "retrieval_queries": normalized_retrieval_queries,
        "riskcalcs_path": normalized_riskcalcs_path,
        "pmid_metadata_path": normalized_pmid_metadata_path,
        "llm_model": normalized_llm_model,
        "retriever_backend": retriever_backend,
        "top_k": retrieval_top_k,
        "max_selected_tools": max_selected_tools,
        "forced_tool_pmid": normalized_forced_tool_pmid,
    }


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the MedAI workflow or clinical-tool workflow from the command line.",
    )
    parser.add_argument(
        "text",
        nargs="?",
        default=None,
        help="Clinical text or question to send into the workflow.",
    )
    parser.add_argument("--state-json", help="Inline JSON payload for a full workflow state.")
    parser.add_argument("--state-file", help="Path to a JSON file for a full workflow state.")
    parser.add_argument("--mode", default="question", choices=["question", "patient_note"], help="Clinical tool mode.")
    parser.add_argument("--structured-case-json", help="Inline JSON payload for structured_case.")
    parser.add_argument("--structured-case-file", help="Path to a JSON file for structured_case.")
    parser.add_argument("--retrieval-queries-json", help="Inline JSON array for retrieval_queries.")
    parser.add_argument("--retrieval-queries-file", help="Path to a JSON file for retrieval_queries.")
    parser.add_argument("--risk-hints-json", help="Inline JSON array for risk_hints.")
    parser.add_argument("--risk-hints-file", help="Path to a JSON file for risk_hints.")
    parser.add_argument("--llm-model", help="Optional model override for the clinical tool workflow.")
    parser.add_argument("--retriever-backend", default="hybrid", help="Retriever backend name.")
    parser.add_argument("--riskcalcs-path", help="Path to riskcalcs.json.")
    parser.add_argument("--pmid-metadata-path", help="Path to pmid2info.json.")
    parser.add_argument("--forced-tool-pmid", help="Optional oracle PMID to force during question-mode execution.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Retrieval recall top-k. Defaults to MEDAI_RETRIEVAL_TOP_K or MEDAI_TOP_K.",
    )
    parser.add_argument("--max-selected-tools", type=int, default=5, help="Max calculators to execute in patient_note mode.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--output",
        choices=["summary", "json"],
        default="summary",
        help="Print a concise summary or the full workflow JSON.",
    )
    parser.add_argument("--save-json", help="Optional output path for saving the full workflow JSON.")
    return parser


def _main(argv: list[str] | None = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)
    state_input = _coalesce_json_sources(args.state_json, args.state_file, label="state")

    structured_case = _coalesce_json_sources(
        args.structured_case_json,
        args.structured_case_file,
        label="structured_case",
    )
    retrieval_queries = _coalesce_json_sources(
        args.retrieval_queries_json,
        args.retrieval_queries_file,
        label="retrieval_queries",
    )
    risk_hints = _coalesce_json_sources(
        args.risk_hints_json,
        args.risk_hints_file,
        label="risk_hints",
    )

    if state_input is not None and args.text:
        parser.error("Do not pass positional text together with --state-json or --state-file.")

    if state_input is not None:
        result = run_workflow(
            state=state_input,
            debug=args.debug,
        )
    else:
        text = str(args.text or DEFAULT_DEMO_TEXT).strip()
        result = run_workflow(
            case_text=text,
            mode=args.mode,
            structured_case=structured_case,
            riskcalcs_path=args.riskcalcs_path,
            pmid_metadata_path=args.pmid_metadata_path,
            llm_model=args.llm_model,
            retriever_backend=args.retriever_backend,
            top_k=args.top_k,
            risk_hints=risk_hints,
            retrieval_queries=retrieval_queries,
            max_selected_tools=args.max_selected_tools,
            forced_tool_pmid=args.forced_tool_pmid,
            debug=args.debug,
        )

    if args.save_json:
        output_path = Path(args.save_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved full workflow JSON to %s", output_path)

    if args.output == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(_extract_cli_summary(result), ensure_ascii=False, indent=2))
    return 0


def run_agent_workflow(
    user_input: str | None = None,
    debug: bool = False,
    deep_thinking_mode: bool = True,
    search_before_planning: bool = False,
    pass_through_expert: bool = False,
    auto_accepted_plan: bool = False,
    config: dict[str, Any] | None = None,
    *,
    patient_case: dict[str, Any] | None = None,
    clinical_tool_job: dict[str, Any] | None = None,
    tool_registry: dict[str, Any] | None = None,
    messages: list[dict[str, str]] | None = None,
    state: GraphState | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the MedAI graph with either placeholder or clinical-tool mode."""
    if state is not None:
        if user_input is not None and str(user_input).strip():
            raise ValueError("Provide either `state` or `user_input`, not both.")
        if patient_case is not None or clinical_tool_job is not None or messages is not None:
            raise ValueError("Provide either `state` or fresh workflow inputs, not both.")
        return _invoke_graph(
            state,
            config=config,
            debug=debug,
            tool_registry=tool_registry,
        )

    request = str(user_input or "").strip()
    if not request:
        raise ValueError("Input could not be empty")

    initial_state = _build_initial_state(
        request,
        messages=messages,
        patient_case=patient_case,
        clinical_tool_job=clinical_tool_job,
        tool_registry=tool_registry,
        deep_thinking_mode=deep_thinking_mode,
        search_before_planning=search_before_planning,
        pass_through_expert=pass_through_expert,
        auto_accepted_plan=auto_accepted_plan,
    )
    return _invoke_graph(
        initial_state,
        config=config,
        debug=debug,
    )


def run_workflow(
    case_text: str | None = None,
    *,
    state: GraphState | dict[str, Any] | None = None,
    request: str | None = None,
    mode: str = "patient_note",
    structured_case: dict[str, Any] | None = None,
    riskcalcs_path: str | None = None,
    pmid_metadata_path: str | None = None,
    llm_model: str | None = None,
    retriever_backend: str = "hybrid",
    top_k: int | None = None,
    risk_hints: list[str] | None = None,
    retrieval_queries: list[dict[str, Any]] | None = None,
    max_selected_tools: int | None = 5,
    forced_tool_pmid: str | None = None,
    debug: bool = False,
    config: dict[str, Any] | None = None,
    tool_registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Unified MedAI workflow entrypoint.

    - Pass `case_text` to start a full workflow directly from a case/question.
    - Pass `state` to continue or integrate from an existing GraphState-style payload.
    """
    if state is not None:
        if case_text is not None and str(case_text).strip():
            raise ValueError("Provide either `case_text` or `state`, not both.")
        return _invoke_graph(
            state,
            config=config,
            debug=debug,
            tool_registry=tool_registry,
        )

    normalized_case_text = str(case_text or "").strip()
    if not normalized_case_text:
        raise ValueError("`case_text` could not be empty when `state` is not provided.")

    clinical_tool_job = _build_clinical_tool_job_payload(
        normalized_case_text,
        mode=mode,
        structured_case=structured_case,
        riskcalcs_path=riskcalcs_path,
        pmid_metadata_path=pmid_metadata_path,
        llm_model=llm_model,
        retriever_backend=retriever_backend,
        top_k=top_k,
        risk_hints=risk_hints,
        retrieval_queries=retrieval_queries,
        max_selected_tools=max_selected_tools,
        forced_tool_pmid=forced_tool_pmid,
    )
    workflow_request = str(request or f"Run MedAI clinical tool workflow in {mode} mode.").strip()
    return run_agent_workflow(
        workflow_request,
        debug=debug,
        config=config,
        clinical_tool_job=clinical_tool_job,
        tool_registry=tool_registry,
        messages=[{"role": "user", "content": normalized_case_text}],
    )


def run_clinical_tool_workflow(
    text: str,
    *,
    mode: str = "patient_note",
    structured_case: dict[str, Any] | None = None,
    riskcalcs_path: str | None = None,
    pmid_metadata_path: str | None = None,
    llm_model: str | None = None,
    retriever_backend: str = "hybrid",
    top_k: int | None = None,
    risk_hints: list[str] | None = None,
    retrieval_queries: list[dict[str, Any]] | None = None,
    max_selected_tools: int | None = 5,
    forced_tool_pmid: str | None = None,
    debug: bool = False,
    tool_registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return run_workflow(
        case_text=text,
        mode=mode,
        structured_case=structured_case,
        riskcalcs_path=riskcalcs_path,
        pmid_metadata_path=pmid_metadata_path,
        llm_model=llm_model,
        retriever_backend=retriever_backend,
        top_k=top_k,
        risk_hints=risk_hints,
        retrieval_queries=retrieval_queries,
        max_selected_tools=max_selected_tools,
        forced_tool_pmid=forced_tool_pmid,
        debug=debug,
        tool_registry=tool_registry,
    )


def main(argv: list[str] | None = None) -> int:
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
