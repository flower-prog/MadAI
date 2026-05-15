from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from agent.graph.types import ClinicalToolJob, GraphState


BUNDLE_KEYS: tuple[str, ...] = (
    "trial_retrieval_bundle",
    "eligibility_assessment_bundle",
    "patient_evidence_bundle",
    "calculator_evidence_bundle",
    "medical_knowledge_bundle",
    "missing_data_bundle",
    "protocol_decision_bundle",
)


ProtocolRunner = Callable[[GraphState], GraphState]


def to_plain_data(value: Any) -> Any:
    if is_dataclass(value):
        return to_plain_data(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_plain_data(item) for item in value]
    return value


def load_eval_cases(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Eval case line {line_number} must be a JSON object.")
            case_id = str(payload.get("case_id") or "").strip()
            if not case_id:
                raise ValueError(f"Eval case line {line_number} is missing case_id.")
            structured_case = payload.get("structured_case")
            if not isinstance(structured_case, dict):
                raise ValueError(f"Eval case {case_id!r} must include structured_case object.")
            payload.setdefault("expected", {})
            cases.append(payload)
            if limit is not None and len(cases) >= max(int(limit), 0):
                break
    return cases


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_plain_data(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _candidate_trials(trial_retrieval_bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in list(trial_retrieval_bundle.get("candidate_ranking") or []):
        if isinstance(item, Mapping):
            rows.append(dict(item))
    return rows


def _candidate_id(candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("nct_id") or candidate.get("trial_id") or candidate.get("id") or "").strip()


def _case_expected(eval_case: Mapping[str, Any]) -> dict[str, Any]:
    expected = eval_case.get("expected")
    return dict(expected) if isinstance(expected, Mapping) else {}


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _list_texts(value: Any) -> list[str]:
    return [_normalize_text(item) for item in list(value or []) if _normalize_text(item)]


def _candidate_text(candidate: Mapping[str, Any]) -> str:
    pieces: list[str] = []
    for key in (
        "nct_id",
        "title",
        "brief_title",
        "condition",
        "conditions",
        "intervention",
        "interventions",
        "summary",
        "brief_summary",
        "overall_status",
        "status",
    ):
        value = candidate.get(key)
        if isinstance(value, (list, tuple, set)):
            pieces.extend(str(item) for item in value)
        elif value is not None:
            pieces.append(str(value))
    return _normalize_text(" ".join(pieces))


def _contains_any(candidate: Mapping[str, Any], terms: Iterable[str]) -> bool:
    haystack = _candidate_text(candidate)
    return any(term and term in haystack for term in terms)


def _is_open_or_recruiting(candidate: Mapping[str, Any]) -> bool:
    status = _normalize_text(candidate.get("overall_status") or candidate.get("status"))
    if bool(candidate.get("enrollment_open")):
        return True
    return any(term in status for term in ("recruiting", "not yet recruiting", "enrolling by invitation"))


def _iter_assessed_trials(eligibility_assessment_bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in list(eligibility_assessment_bundle.get("assessed_trials") or [])
        if isinstance(item, Mapping)
    ]


def _criteria_counts(assessed_trials: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for trial in assessed_trials:
        for criterion in list(trial.get("criteria") or []):
            if not isinstance(criterion, Mapping):
                continue
            counts["criteria_count"] += 1
            label = str(criterion.get("label") or "unknown").strip().lower() or "unknown"
            counts[f"{label}_count"] += 1
    for trial in assessed_trials:
        counts["blocking_criteria_count"] += len(list(trial.get("blocking_criteria") or []))
        counts["unknown_criteria_count"] += len(list(trial.get("unknown_criteria") or []))
        counts["missing_question_count"] += len(list(trial.get("missing_questions") or []))
    return counts


def _parse_status_counts(assessed_trials: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for trial in assessed_trials:
        status = str(trial.get("eligibility_section_parse_status") or "unknown").strip() or "unknown"
        counts[status] += 1
    return dict(counts)


def _query_profile(trial_retrieval_bundle: Mapping[str, Any]) -> dict[str, Any]:
    payload = trial_retrieval_bundle.get("query_profile")
    return dict(payload) if isinstance(payload, Mapping) else {}


def _trial_retrieval_context(query_profile: Mapping[str, Any]) -> dict[str, Any]:
    payload = query_profile.get("trial_retrieval_context")
    if isinstance(payload, Mapping):
        return dict(payload)
    return {
        "raw_text": _normalize_text(query_profile.get("raw_text")),
        "case_summary": _normalize_text(query_profile.get("case_summary")),
        "source_terms": dict(query_profile.get("source_terms") or {}),
        "expanded_terms": dict(query_profile.get("expanded_terms") or {}),
        "retrieval_queries": list(query_profile.get("retrieval_queries") or []),
    }


def _retrieval_queries(query_profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    queries = query_profile.get("retrieval_queries")
    if not isinstance(queries, list):
        queries = dict(query_profile.get("trial_retrieval_context") or {}).get("retrieval_queries")
    return [dict(item) for item in list(queries or []) if isinstance(item, Mapping)]


def _expanded_terms(query_profile: Mapping[str, Any]) -> dict[str, list[str]]:
    raw_terms = query_profile.get("expanded_terms")
    if not isinstance(raw_terms, Mapping):
        raw_terms = dict(query_profile.get("trial_retrieval_context") or {}).get("expanded_terms")
    expanded: dict[str, list[str]] = {}
    for key, value in dict(raw_terms or {}).items():
        expanded[str(key)] = [
            str(item).strip()
            for item in list(value or [])
            if str(item).strip()
        ]
    return expanded


def _trial_retrieval_diagnostic_metrics(trial_retrieval_bundle: Mapping[str, Any]) -> dict[str, Any]:
    profile = _query_profile(trial_retrieval_bundle)
    queries = _retrieval_queries(profile)
    expanded = _expanded_terms(profile)
    query_names = [str(item.get("name") or "").strip() for item in queries if str(item.get("name") or "").strip()]
    return {
        "retrieval_query_count": len(queries),
        "retrieval_query_names": query_names,
        "has_raw_fallback_query": "raw_fallback" in set(query_names),
        "expanded_condition_count": len(expanded.get("conditions") or []),
        "expanded_biomarker_count": len(expanded.get("biomarkers") or []),
        "expanded_treatment_count": len(expanded.get("treatments") or []),
        "expanded_treatment_context_count": len(expanded.get("treatment_context") or []),
        "expanded_eligibility_count": len(expanded.get("eligibility") or []),
        "expanded_negative_count": len(expanded.get("negative") or []),
    }


def evaluate_case_result(
    eval_case: Mapping[str, Any],
    bundles: Mapping[str, Any],
    *,
    trace_dir: str | Path | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    expected = _case_expected(eval_case)
    trial_bundle = dict(bundles.get("trial_retrieval_bundle") or {})
    eligibility_bundle = dict(bundles.get("eligibility_assessment_bundle") or {})
    retrieval_diagnostics = _trial_retrieval_diagnostic_metrics(trial_bundle)
    candidates = _candidate_trials(trial_bundle)
    top10 = candidates[:10]
    top20 = candidates[:20]
    top10_ids = [_candidate_id(item) for item in top10 if _candidate_id(item)]
    top20_ids = [_candidate_id(item) for item in top20 if _candidate_id(item)]

    must_ids = {str(item).strip() for item in list(expected.get("must_retrieve_nct_ids") or []) if str(item).strip()}
    acceptable_ids = {
        str(item).strip() for item in list(expected.get("acceptable_nct_ids") or []) if str(item).strip()
    }
    acceptable_conditions = _list_texts(expected.get("acceptable_conditions"))
    acceptable_interventions = _list_texts(expected.get("acceptable_interventions"))

    assessed_trials = _iter_assessed_trials(eligibility_bundle)
    criterion_counts = _criteria_counts(assessed_trials)
    condition_match_count_at_10 = sum(1 for candidate in top10 if _contains_any(candidate, acceptable_conditions))
    intervention_match_count_at_10 = sum(
        1 for candidate in top10 if _contains_any(candidate, acceptable_interventions)
    )
    open_or_recruiting_count_at_10 = sum(1 for candidate in top10 if _is_open_or_recruiting(candidate))
    must_hit_top10 = must_ids.intersection(top10_ids)
    must_hit_top20 = must_ids.intersection(top20_ids)
    acceptable_hit_top10 = acceptable_ids.intersection(top10_ids)

    failure_buckets: list[str] = []
    if error:
        failure_buckets.append("protocol_exception")
    if not candidates:
        failure_buckets.append("no_candidates")
    if must_ids and not must_hit_top20:
        failure_buckets.append("must_retrieve_miss")
    if acceptable_conditions and candidates and condition_match_count_at_10 == 0:
        failure_buckets.append("wrong_condition_top10")
    if candidates and open_or_recruiting_count_at_10 == 0:
        failure_buckets.append("no_recruiting_trials_top10")
    if candidates and not assessed_trials:
        failure_buckets.append("eligibility_not_assessed")
    if assessed_trials and int(criterion_counts.get("criteria_count") or 0) == 0:
        failure_buckets.append("criteria_parse_empty")
    if int(criterion_counts.get("unknown_count") or 0) > (
        int(criterion_counts.get("met_count") or 0) + int(criterion_counts.get("not_met_count") or 0)
    ):
        failure_buckets.append("too_many_unknowns")
    if int(criterion_counts.get("unknown_count") or 0) and not int(criterion_counts.get("missing_question_count") or 0):
        failure_buckets.append("missing_questions_empty_despite_unknowns")

    case_metrics = {
        "case_id": str(eval_case.get("case_id") or ""),
        "status": "failed" if error else "completed",
        "error": error or "",
        "retrieved_trial_count": len(candidates),
        "candidate_ids_top10": top10_ids,
        "candidate_ids_top20": top20_ids,
        "must_retrieve_hit_count_at_10": len(must_hit_top10),
        "must_retrieve_hit_count_at_20": len(must_hit_top20),
        "must_retrieve_recall_at_10": (len(must_hit_top10) / len(must_ids)) if must_ids else None,
        "must_retrieve_recall_at_20": (len(must_hit_top20) / len(must_ids)) if must_ids else None,
        "acceptable_nct_hit_count_at_10": len(acceptable_hit_top10),
        "condition_match_count_at_10": condition_match_count_at_10,
        "intervention_match_count_at_10": intervention_match_count_at_10,
        "open_or_recruiting_count_at_10": open_or_recruiting_count_at_10,
        "assessed_trial_count": len(assessed_trials),
        "criteria_count": int(criterion_counts.get("criteria_count") or 0),
        "met_count": int(criterion_counts.get("met_count") or 0),
        "not_met_count": int(criterion_counts.get("not_met_count") or 0),
        "unknown_count": int(criterion_counts.get("unknown_count") or 0),
        "blocking_criteria_count": int(criterion_counts.get("blocking_criteria_count") or 0),
        "missing_question_count": int(criterion_counts.get("missing_question_count") or 0),
        "parse_status_counts": _parse_status_counts(assessed_trials),
        "parse_warning_count": sum(
            len(list(trial.get("eligibility_section_parse_warnings") or [])) for trial in assessed_trials
        ),
        **retrieval_diagnostics,
        "failure_buckets": failure_buckets,
        "trace_dir": str(trace_dir) if trace_dir is not None else "",
    }
    return case_metrics


def summarize_eval_results(case_metrics: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    cases = [dict(item) for item in case_metrics]
    failure_counts: Counter[str] = Counter()
    for case in cases:
        failure_counts.update(str(item) for item in list(case.get("failure_buckets") or []))

    numeric_keys = (
        "retrieved_trial_count",
        "retrieval_query_count",
        "expanded_condition_count",
        "expanded_biomarker_count",
        "expanded_treatment_count",
        "expanded_treatment_context_count",
        "expanded_eligibility_count",
        "expanded_negative_count",
        "condition_match_count_at_10",
        "intervention_match_count_at_10",
        "open_or_recruiting_count_at_10",
        "assessed_trial_count",
        "criteria_count",
        "unknown_count",
        "missing_question_count",
    )
    means: dict[str, float] = {}
    for key in numeric_keys:
        values = [float(case.get(key) or 0) for case in cases]
        means[f"mean_{key}"] = sum(values) / len(values) if values else 0.0

    return {
        "schema_version": 1,
        "case_count": len(cases),
        "completed_case_count": sum(1 for case in cases if case.get("status") == "completed"),
        "failed_case_count": sum(1 for case in cases if case.get("status") == "failed"),
        "metrics": means,
        "failure_buckets": dict(sorted(failure_counts.items())),
        "cases": cases,
    }


def extract_protocol_bundles(state: GraphState) -> dict[str, Any]:
    bundles: dict[str, Any] = {}
    for key in BUNDLE_KEYS:
        bundles[key] = to_plain_data(state.final_output.get(key) or getattr(state, key, {}) or {})
    return bundles


def write_case_trace(
    *,
    trace_dir: str | Path,
    eval_case: Mapping[str, Any],
    bundles: Mapping[str, Any],
    case_metrics: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> Path:
    target = Path(trace_dir)
    target.mkdir(parents=True, exist_ok=True)
    _write_json(target / "input_case.json", eval_case)
    query_profile = dict((bundles.get("trial_retrieval_bundle") or {}).get("query_profile") or {})
    if not query_profile:
        query_profile = {"query_text": (bundles.get("trial_retrieval_bundle") or {}).get("query_text", "")}
    _write_json(target / "query_profile.json", query_profile)
    _write_json(target / "trial_retrieval_context.json", _trial_retrieval_context(query_profile))
    _write_json(target / "retrieval_queries.json", _retrieval_queries(query_profile))
    _write_json(target / "expanded_terms.json", _expanded_terms(query_profile))
    for key in BUNDLE_KEYS:
        _write_json(target / f"{key}.json", bundles.get(key) or {})
    if case_metrics is not None:
        _write_json(target / "case_metrics.json", case_metrics)
    if error:
        _write_json(target / "error.json", {"error": error})
    return target


def build_protocol_eval_state(
    eval_case: Mapping[str, Any],
    *,
    backend: str = "hybrid",
    top_k: int = 20,
    tool_registry: Mapping[str, Any] | None = None,
) -> GraphState:
    structured_case = dict(eval_case.get("structured_case") or {})
    request = str(eval_case.get("description") or structured_case.get("case_summary") or structured_case.get("raw_text") or "")
    department_tags = [
        str(item).strip()
        for item in list(structured_case.get("department_tags") or eval_case.get("department_tags") or [])
        if str(item).strip()
    ]
    return GraphState(
        request=request,
        structured_case_json=structured_case,
        department_tags=department_tags,
        clinical_tool_job=ClinicalToolJob(
            mode="patient_note",
            text=str(structured_case.get("raw_text") or request),
            structured_case=structured_case,
            top_k=max(int(top_k), 1),
            retriever_backend=backend,  # type: ignore[arg-type]
        ),
        tool_registry=dict(tool_registry or {}),
    )


def run_protocol_trial_eval(
    cases: Iterable[Mapping[str, Any]],
    *,
    output_path: str | Path,
    trace_dir: str | Path,
    protocol_runner: ProtocolRunner,
    backend: str = "hybrid",
    top_k: int = 20,
    tool_registry: Mapping[str, Any] | None = None,
    fail_fast: bool = False,
) -> dict[str, Any]:
    output = Path(output_path)
    trace_root = Path(trace_dir)
    case_metrics: list[dict[str, Any]] = []

    for eval_case in cases:
        case_id = str(eval_case.get("case_id") or "case").strip()
        current_trace_dir = trace_root / case_id
        bundles: dict[str, Any] = {key: {} for key in BUNDLE_KEYS}
        error = ""
        try:
            state = build_protocol_eval_state(
                eval_case,
                backend=backend,
                top_k=top_k,
                tool_registry=tool_registry,
            )
            result_state = protocol_runner(state)
            bundles = extract_protocol_bundles(result_state)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if fail_fast:
                metrics = evaluate_case_result(eval_case, bundles, trace_dir=current_trace_dir, error=error)
                write_case_trace(
                    trace_dir=current_trace_dir,
                    eval_case=eval_case,
                    bundles=bundles,
                    case_metrics=metrics,
                    error=error,
                )
                raise

        metrics = evaluate_case_result(eval_case, bundles, trace_dir=current_trace_dir, error=error or None)
        write_case_trace(
            trace_dir=current_trace_dir,
            eval_case=eval_case,
            bundles=bundles,
            case_metrics=metrics,
            error=error or None,
        )
        case_metrics.append(metrics)

    summary = summarize_eval_results(case_metrics)
    _write_json(output, summary)
    return summary
