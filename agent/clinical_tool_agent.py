from __future__ import annotations

import json
import math
import re
from pathlib import Path
import threading
from typing import Any

from .graph.types import ClinicalToolJob, RetrievalQuery
from .tools import (
    ChatClient,
    RiskCalcCatalog,
    RiskCalcComputationRetrievalTool,
    RiskCalcExecutionTool,
    RiskCalcExecutor,
    RiskCalcRegistration,
    RiskCalcRegistry,
    build_default_chat_client,
    build_patient_note_queries,
    build_tool_call,
    collect_tools,
    create_computation_retrieval_tool,
    create_execution_tool,
    create_retrieval_tool,
    discover_healthy_defaults_path,
    execute_python_code,
    extract_python_code_blocks,
    extract_risk_hints_from_queries,
    generate_risk_hints,
    is_risk_hint_query,
    load_calculator_healthy_defaults,
    maybe_load_json,
    summarize_text,
)


def _normalize_yes_no(value: Any, *, default: str = "no") -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"yes", "y", "true", "1"}:
        return "yes"
    if normalized in {"no", "n", "false", "0"}:
        return "no"
    return default


def _decision_passes_gate(decision: dict[str, Any] | None) -> bool:
    if not isinstance(decision, dict):
        return False

    gate_status = str(decision.get("gate_status") or "").strip().lower()
    execution_status = str(decision.get("execution_status") or "").strip().lower()
    if gate_status:
        return gate_status == "passed" or execution_status == "completed"
    if execution_status:
        return execution_status == "completed"

    return (
        str(decision.get("patient_eligible") or "").strip().lower() == "yes"
        and str(decision.get("missing_all_parameters") or "").strip().lower() == "no"
    )


def _coerce_text_list(value: Any, *, limit: int | None = None) -> list[str]:
    values: list[str] = []
    if value is None:
        return values
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, dict):
        values = [str(item or "").strip() for item in value.values()]
    else:
        try:
            values = [str(item or "").strip() for item in value]
        except TypeError:
            values = [str(value or "").strip()]

    deduped: list[str] = []
    seen: set[str] = set()
    for raw in values:
        normalized = str(raw or "").strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)

    if limit is None:
        return deduped
    return deduped[: max(int(limit), 0)]


def _build_recommended_channels(
    *,
    bm25_raw_top5: list[dict[str, Any]],
    vector_raw_top5: list[dict[str, Any]],
    per_channel_limit: int = 2,
) -> dict[str, list[str]]:
    recommended: dict[str, list[str]] = {}
    for channel, rows in (("bm25", bm25_raw_top5), ("vector", vector_raw_top5)):
        for row in list(rows)[: max(int(per_channel_limit), 0)]:
            pmid = str(row.get("pmid") or "").strip()
            if not pmid:
                continue
            channels = recommended.setdefault(pmid, [])
            if channel not in channels:
                channels.append(channel)
    return {
        pmid: sorted(channels)
        for pmid, channels in recommended.items()
        if channels
    }


def _public_candidate_view(
    candidate: dict[str, Any],
    *,
    rank: int | None = None,
    recommended_channels: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    pmid = str(candidate.get("pmid") or "").strip()
    payload: dict[str, Any] = {
        "pmid": pmid,
        "title": str(candidate.get("title") or "").strip(),
        "purpose": str(candidate.get("purpose") or "").strip(),
        "specialty": str(candidate.get("specialty") or "").strip(),
        "eligibility": str(candidate.get("eligibility") or "").strip(),
        "parameter_names": list(candidate.get("parameter_names") or []),
        "recommended": list((recommended_channels or {}).get(pmid) or []),
    }
    if rank is not None:
        payload["rank"] = rank
    return payload


def _public_context_hit_view(
    row: dict[str, Any],
    *,
    recommended_channels: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    pmid = str(row.get("pmid") or "").strip()
    payload: dict[str, Any] = {
        "rank": int(row.get("rank") or 0),
        "channel": str(row.get("channel") or "").strip(),
        "pmid": pmid,
        "parameter_names": list(row.get("parameter_names") or []),
        "matched_parameter_names": list(row.get("matched_parameter_names") or []),
        "matched_aliases": list(row.get("matched_aliases") or []),
        "match_sources": sorted(set(list(row.get("match_sources") or []))),
        "query_text": str(row.get("query_text") or ""),
    }
    title = str(row.get("title") or "").strip()
    if title:
        payload["title"] = title
    purpose = str(row.get("purpose") or "").strip()
    if purpose:
        payload["purpose"] = purpose
    eligibility = str(row.get("eligibility") or "").strip()
    if eligibility:
        payload["eligibility"] = eligibility
    calculator_payload = dict(row.get("calculator_payload") or {})
    if calculator_payload:
        payload["calculator_payload"] = calculator_payload
    recommended = list((recommended_channels or {}).get(pmid) or [])
    if recommended:
        payload["recommended"] = recommended
    return payload


_QUERY_CHANNEL_WEIGHTS: dict[str, float] = {
    "case_summary": 6.0,
    "parameter_match": 4.5,
    "problem_anchor": 3.0,
    "calculator_category": 2.0,
    "protocol_mapping": 1.5,
    "risk_hint": 1.0,
    "general": 1.25,
}

_QUERY_CHANNEL_ORDER: dict[str, int] = {
    "case_summary": 0,
    "parameter_match": 1,
    "problem_anchor": 2,
    "calculator_category": 3,
    "protocol_mapping": 4,
    "risk_hint": 5,
    "general": 6,
}

_FUSION_RRF_K = 30
_PATIENT_NOTE_SELECTION_SCAN_MULTIPLIER = 3
_PATIENT_NOTE_SELECTION_EXTRA_BUFFER = 5
_PATIENT_NOTE_MIN_SELECTION_SCAN = 8
_QUESTION_SELECTION_POOL_LIMIT = 6

_RUNTIME_CACHE_LOCK = threading.RLock()
_REGISTRY_CACHE: dict[tuple[str, str], RiskCalcRegistry] = {}


def clear_runtime_caches() -> None:
    with _RUNTIME_CACHE_LOCK:
        _REGISTRY_CACHE.clear()


def _path_signature(path: str | Path | None) -> str:
    if path is None:
        return "<missing>"
    file_path = Path(path).resolve()
    try:
        stat = file_path.stat()
    except OSError:
        return str(file_path)
    return f"{file_path}|{stat.st_size}|{stat.st_mtime_ns}"


def _registry_cache_key(catalog: RiskCalcCatalog, *, healthy_defaults_path: str | Path | None) -> tuple[str, str]:
    return (
        catalog.runtime_cache_key,
        _path_signature(healthy_defaults_path),
    )


def _get_cached_registry(
    catalog: RiskCalcCatalog,
    *,
    healthy_defaults_by_calc: dict[str, dict[str, Any]],
    healthy_defaults_path: str | Path | None,
) -> RiskCalcRegistry:
    cache_key = _registry_cache_key(catalog, healthy_defaults_path=healthy_defaults_path)
    with _RUNTIME_CACHE_LOCK:
        cached = _REGISTRY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    registry = RiskCalcRegistry.from_catalog_with_defaults(
        catalog,
        healthy_defaults_by_calc=healthy_defaults_by_calc,
    )
    with _RUNTIME_CACHE_LOCK:
        cached = _REGISTRY_CACHE.setdefault(cache_key, registry)
    return cached


def prewarm_clinical_tool_job(job: ClinicalToolJob) -> None:
    riskcalcs_path = job.riskcalcs_path
    pmid_metadata_path = job.pmid_metadata_path
    if not riskcalcs_path or not pmid_metadata_path:
        discovered_riskcalcs, discovered_pmid = RiskCalcCatalog.discover_default_paths()
        riskcalcs_path = riskcalcs_path or str(discovered_riskcalcs)
        pmid_metadata_path = pmid_metadata_path or str(discovered_pmid)

    catalog = RiskCalcCatalog.from_paths(riskcalcs_path, pmid_metadata_path)
    create_retrieval_tool(catalog, backend=job.retriever_backend)
    create_computation_retrieval_tool(catalog)
    healthy_defaults_path = discover_healthy_defaults_path(
        search_root=Path(riskcalcs_path).resolve().parents[1]
    )
    healthy_defaults_by_calc = load_calculator_healthy_defaults(healthy_defaults_path)
    _get_cached_registry(
        catalog,
        healthy_defaults_by_calc=healthy_defaults_by_calc,
        healthy_defaults_path=healthy_defaults_path,
    )

_LOW_SIGNAL_QUERY_TOKENS = {
    "adult",
    "adults",
    "admission",
    "admitted",
    "assessment",
    "associated",
    "calculator",
    "care",
    "category",
    "chronic",
    "clinical",
    "complication",
    "complications",
    "concern",
    "concerns",
    "critical",
    "diagnosis",
    "disease",
    "emergency",
    "followup",
    "general",
    "history",
    "hospital",
    "mapping",
    "mode",
    "monitoring",
    "oncology",
    "patient",
    "patients",
    "priority",
    "problem",
    "progressive",
    "protocol",
    "query",
    "related",
    "risk",
    "screening",
    "selection",
    "severe",
    "score",
    "stage",
    "status",
    "symptom",
    "symptoms",
    "treatment",
}

_TOKEN_CANONICAL_MAP = {
    "astrocytomas": "astrocytoma",
    "cancers": "cancer",
    "cancerous": "cancer",
    "catheters": "catheter",
    "compressions": "compression",
    "corticosteroid": "steroid",
    "corticosteroids": "steroid",
    "glioblastoma": "glioma",
    "glioblastomas": "glioma",
    "malignancies": "cancer",
    "malignancy": "cancer",
    "reirradiations": "reirradiation",
    "retentions": "retention",
    "shocks": "shock",
    "spinal": "spine",
    "tumors": "tumor",
}


def _canonicalize_alignment_token(token: str) -> str:
    normalized = str(token or "").strip().lower()
    if not normalized:
        return ""
    normalized = _TOKEN_CANONICAL_MAP.get(normalized, normalized)
    if len(normalized) > 4 and normalized.endswith("ies"):
        normalized = normalized[:-3] + "y"
    elif len(normalized) > 4 and normalized.endswith("s"):
        normalized = normalized[:-1]
    return _TOKEN_CANONICAL_MAP.get(normalized, normalized)


def _extract_alignment_tokens(text: str) -> list[str]:
    raw_tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())
    tokens = []
    for raw_token in raw_tokens:
        normalized = _canonicalize_alignment_token(raw_token)
        if len(normalized) < 3:
            continue
        tokens.append(normalized)
    return tokens


def _build_query_alignment_signature(text: str) -> tuple[dict[str, float], dict[str, float]]:
    tokens = _extract_alignment_tokens(text)
    token_weights: dict[str, float] = {}
    weighted_tokens_for_phrases: list[str] = []
    for token in tokens:
        if token in _LOW_SIGNAL_QUERY_TOKENS:
            weight = 0.15
        else:
            weight = 1.0
            weighted_tokens_for_phrases.append(token)
        token_weights[token] = max(token_weights.get(token, 0.0), weight)

    bigram_weights: dict[str, float] = {}
    for left, right in zip(weighted_tokens_for_phrases, weighted_tokens_for_phrases[1:]):
        if left == right:
            continue
        bigram = f"{left} {right}"
        bigram_weights[bigram] = 1.5

    return token_weights, bigram_weights


def _build_alignment_features(text: str) -> tuple[set[str], set[str]]:
    tokens = _extract_alignment_tokens(text)
    token_set = set(tokens)
    bigram_set = {f"{left} {right}" for left, right in zip(tokens, tokens[1:]) if left != right}
    return token_set, bigram_set


class ClinicalToolAgent:
    def __init__(
        self,
        *,
        catalog: RiskCalcCatalog,
        retriever: Any,
        chat_client: ChatClient,
        computation_retriever: RiskCalcComputationRetrievalTool | None = None,
        registry: RiskCalcRegistry | None = None,
        executor: RiskCalcExecutor | None = None,
        execution_tool: RiskCalcExecutionTool | None = None,
    ) -> None:
        self.catalog = catalog
        self.retriever = retriever
        self.computation_retriever = computation_retriever
        self.chat_client = chat_client
        self.registry = registry
        self.execution_tool = execution_tool or (
            create_execution_tool(
                registry,
                chat_client=chat_client,
                executor=executor,
            )
            if registry is not None
            else None
        )
        self.tools = collect_tools(self.retriever, self.computation_retriever, self.execution_tool)
        if self.execution_tool is not None and "riskcalc_executor" not in self.tools:
            self.tools["riskcalc_executor"] = self.execution_tool
        self._trace_tool_calls: list[dict[str, Any]] = []
        self._latest_retrieval_bundle: dict[str, Any] | None = None
        self._latest_coarse_retrieval_bundle: dict[str, Any] | None = None

    @classmethod
    def from_job(cls, job: ClinicalToolJob) -> "ClinicalToolAgent":
        riskcalcs_path = job.riskcalcs_path
        pmid_metadata_path = job.pmid_metadata_path
        if not riskcalcs_path or not pmid_metadata_path:
            discovered_riskcalcs, discovered_pmid = RiskCalcCatalog.discover_default_paths()
            riskcalcs_path = riskcalcs_path or str(discovered_riskcalcs)
            pmid_metadata_path = pmid_metadata_path or str(discovered_pmid)

        catalog = RiskCalcCatalog.from_paths(riskcalcs_path, pmid_metadata_path)
        retriever = create_retrieval_tool(catalog, backend=job.retriever_backend)
        computation_retriever = create_computation_retrieval_tool(catalog)
        chat_client = build_default_chat_client(model=job.llm_model)
        healthy_defaults_path = discover_healthy_defaults_path(
            search_root=Path(riskcalcs_path).resolve().parents[1]
        )
        healthy_defaults_by_calc = load_calculator_healthy_defaults(healthy_defaults_path)
        registry = _get_cached_registry(
            catalog,
            healthy_defaults_by_calc=healthy_defaults_by_calc,
            healthy_defaults_path=healthy_defaults_path,
        )
        executor = RiskCalcExecutor(registry)
        execution_tool = create_execution_tool(
            registry,
            chat_client=chat_client,
            executor=executor,
        )
        return cls(
            catalog=catalog,
            retriever=retriever,
            computation_retriever=computation_retriever,
            chat_client=chat_client,
            registry=registry,
            executor=executor,
            execution_tool=execution_tool,
        )

    def _reset_trace(self) -> None:
        self._trace_tool_calls = []
        self._latest_retrieval_bundle = None
        self._latest_coarse_retrieval_bundle = None

    def _record_tool_call(
        self,
        tool_name: str,
        *,
        status: str = "completed",
        input_payload: Any = None,
        output_payload: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._trace_tool_calls.append(
            build_tool_call(
                tool_name,
                status=status,
                input_payload=input_payload,
                output_payload=output_payload,
                metadata=metadata,
            )
        )

    def _build_trace_bundle(self, *, status: str = "completed") -> dict[str, Any]:
        return {
            "agent_name": "clinical_tool_agent",
            "status": status,
            "tool_calls": list(self._trace_tool_calls),
        }

    def run(self, job: ClinicalToolJob) -> dict[str, Any]:
        self._reset_trace()
        if job.mode == "question":
            result = self._run_question(job)
        else:
            result = self._run_patient_note(job)
        result["trace"] = self._build_trace_bundle()
        return result

    @staticmethod
    def _selection_override_pmid(job: ClinicalToolJob) -> str:
        return str(job.selected_tool_pmid or job.forced_tool_pmid or "").strip()

    def _should_use_preselected_path(self, job: ClinicalToolJob) -> bool:
        if self._selection_override_pmid(job):
            return True
        return bool(job.selection_context)

    def plan_selection(self, job: ClinicalToolJob) -> dict[str, Any]:
        self._reset_trace()
        retrieval_context = self._retrieve_and_rank_candidates(job)
        selection_candidates = list(retrieval_context.get("question_selection_candidates") or [])
        raw_bm25_top5 = list(
            retrieval_context.get("bm25_raw_top3")
            or retrieval_context.get("bm25_raw_top5")
            or []
        )
        raw_vector_top5 = list(
            retrieval_context.get("vector_raw_top3")
            or retrieval_context.get("vector_raw_top5")
            or []
        )
        recommended_channels = _build_recommended_channels(
            bm25_raw_top5=raw_bm25_top5,
            vector_raw_top5=raw_vector_top5,
        )
        if not selection_candidates:
            retrieved_tools = list(
                retrieval_context.get("retrieved_tools")
                or retrieval_context.get("candidate_ranking")
                or []
            )
            selection_candidates = self._build_question_selection_candidates(
                retrieved_tools=retrieved_tools,
                bm25_raw_top5=raw_bm25_top5,
                vector_raw_top5=raw_vector_top5,
                extra_candidates=list(retrieval_context.get("candidate_ranking") or []),
                limit=_QUESTION_SELECTION_POOL_LIMIT,
            )
        else:
            selection_candidates = selection_candidates[:_QUESTION_SELECTION_POOL_LIMIT]

        override_pmid = self._selection_override_pmid(job)
        if override_pmid:
            selected_tool = self._build_forced_question_selection(
                job,
                selection_candidates,
                forced_tool_pmid=override_pmid,
            )
            selection_mode = "oracle_forced" if job.forced_tool_pmid else "parent_selected"
        else:
            selected_tool = self._select_tool_for_question(
                job,
                selection_candidates,
            )
            selection_mode = "model_selected"

        selected_pmid = str(selected_tool.get("pmid") or "").strip()
        selected_candidate = next(
            (candidate for candidate in selection_candidates if str(candidate.get("pmid") or "").strip() == selected_pmid),
            None,
        )
        dispatch_query_text = ""
        dispatch_query_source = "unavailable"
        if selected_pmid:
            dispatch_query_text, dispatch_query_source = self._resolve_parameter_extraction_query_text(
                job,
                selected_pmid=selected_pmid,
                selected_candidate=selected_candidate,
            )

        public_candidates = [
            _public_candidate_view(
                dict(candidate),
                recommended_channels=recommended_channels,
            )
            for candidate in selection_candidates
        ]
        return {
            "mode": job.mode,
            "risk_hints": list(retrieval_context.get("risk_hints") or []),
            "retrieval_batches": list(retrieval_context.get("retrieval_batches") or []),
            "retrieved_tools": list(public_candidates),
            "candidate_ranking": list(public_candidates),
            "selection_candidates": [dict(candidate) for candidate in selection_candidates],
            "bm25_raw_top5": [
                _public_context_hit_view(
                    dict(item),
                    recommended_channels=recommended_channels,
                )
                for item in raw_bm25_top5
            ],
            "vector_raw_top5": [
                _public_context_hit_view(
                    dict(item),
                    recommended_channels=recommended_channels,
                )
                for item in raw_vector_top5
            ],
            "recommended_pmids": list(recommended_channels.keys()),
            "selected_tool": dict(selected_tool),
            "selection_mode": selection_mode,
            "dispatch_query_text": dispatch_query_text,
            "dispatch_query_source": dispatch_query_source,
        }

    def _build_preselected_tool(
        self,
        job: ClinicalToolJob,
        *,
        selected_pmid: str,
        selected_candidate: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        selection_context = dict(job.selection_context or {})
        existing = dict(job.selected_tool or selection_context.get("selected_tool") or {})
        if str(existing.get("pmid") or "").strip() != selected_pmid:
            existing = {}

        selected_doc = self.catalog.get(selected_pmid)
        parameter_names = self._resolve_candidate_parameter_names(
            pmid=selected_pmid,
            candidate=selected_candidate or existing,
        )
        return {
            "pmid": selected_pmid,
            "title": str(existing.get("title") or selected_doc.title).strip(),
            "purpose": str(existing.get("purpose") or selected_doc.purpose).strip(),
            "parameter_names": list(parameter_names),
            "model_parameter_names": list(existing.get("model_parameter_names") or []),
            "model_selected_tool_id": existing.get("model_selected_tool_id"),
            "fallback_used": bool(existing.get("fallback_used") or False),
            "reason": str(
                existing.get("reason")
                or (
                    "Forced oracle tool selection for evaluation."
                    if job.forced_tool_pmid and not job.selected_tool_pmid
                    else "Calculator PMID selected by clinical_assisstment."
                )
            ).strip(),
            "raw_response": existing.get("raw_response"),
            "present_in_retrieved_candidates": bool(
                existing.get("present_in_retrieved_candidates")
                or selected_candidate
            ),
        }

    def _run_preselected(self, job: ClinicalToolJob) -> dict[str, Any]:
        selected_pmid = self._selection_override_pmid(job)
        selection_context = dict(job.selection_context or {})
        raw_selection_candidates = [
            dict(item)
            for item in list(selection_context.get("selection_candidates") or [])
            if isinstance(item, dict)
        ]
        selected_candidate = next(
            (candidate for candidate in raw_selection_candidates if str(candidate.get("pmid") or "").strip() == selected_pmid),
            None,
        )

        selected_tool: dict[str, Any]
        if selected_pmid:
            selected_tool = self._build_preselected_tool(
                job,
                selected_pmid=selected_pmid,
                selected_candidate=selected_candidate,
            )
        else:
            selected_tool = dict(job.selected_tool or selection_context.get("selected_tool") or {})

        retrieved_tools = [
            dict(item)
            for item in list(
                selection_context.get("retrieved_tools")
                or selection_context.get("candidate_ranking")
                or []
            )
            if isinstance(item, dict)
        ]
        if not retrieved_tools and selected_tool:
            retrieved_tools = [
                _public_candidate_view(
                    dict(selected_candidate or selected_tool),
                    recommended_channels={selected_pmid: []} if selected_pmid else {},
                )
            ]

        execution: dict[str, Any] = {}
        executions: list[dict[str, Any]] = []
        if selected_pmid:
            execution = self._execute_calculator(
                job,
                selected_pmid,
                selected_candidate=selected_candidate,
            )
            self._record_tool_call(
                "python_calculator_executor",
                status=str(execution.get("status") or "unknown"),
                input_payload={
                    "mode": job.mode,
                    "pmid": selected_pmid,
                    "title": selected_tool.get("title"),
                    "clinical_text": summarize_text(job.text),
                },
                output_payload=execution,
                metadata={"dispatch": f"{job.mode}_mode", "source": "parent_preselected_execution"},
            )
            execution = dict(execution)
            executions.append(dict(execution))
        if not execution:
            execution = {
                "status": "skipped",
                "final_text": "No calculator PMID was selected by clinical_assisstment.",
            }

        result = {
            "mode": job.mode,
            "retrieval_batches": list(selection_context.get("retrieval_batches") or []),
            "retrieved_tools": list(retrieved_tools),
            "candidate_ranking": list(selection_context.get("candidate_ranking") or retrieved_tools),
            "bm25_raw_top5": [
                dict(item)
                for item in list(selection_context.get("bm25_raw_top5") or [])
                if isinstance(item, dict)
            ],
            "vector_raw_top5": [
                dict(item)
                for item in list(selection_context.get("vector_raw_top5") or [])
                if isinstance(item, dict)
            ],
            "selection_decisions": [],
            "eligible_tools": [],
            "selected_tool": selected_tool,
            "executions": executions,
            "execution": execution,
            "recommended_pmids": list(selection_context.get("recommended_pmids") or ([selected_pmid] if selected_pmid else [])),
            "selection_mode": str(
                selection_context.get("selection_mode")
                or ("oracle_forced" if job.forced_tool_pmid and not job.selected_tool_pmid else "parent_selected")
            ).strip(),
            "dispatch_query_text": str(
                job.dispatch_query_text
                or selection_context.get("dispatch_query_text")
                or ""
            ),
            "dispatch_query_source": str(
                selection_context.get("dispatch_query_source")
                or ("job.dispatch_query_text" if job.dispatch_query_text else "")
            ),
        }
        if job.mode == "patient_note":
            result["risk_hints"] = list(selection_context.get("risk_hints") or job.risk_hints)
        return result

    def _run_question(self, job: ClinicalToolJob) -> dict[str, Any]:
        if self._should_use_preselected_path(job):
            return self._run_preselected(job)

        selection_bundle = self.plan_selection(job)
        selection_candidates = list(selection_bundle.get("selection_candidates") or [])
        selected_tool = dict(selection_bundle.get("selected_tool") or {})
        selected_pmid = str(selected_tool.get("pmid") or "").strip()
        execution: dict[str, Any] = {}
        executions: list[dict[str, Any]] = []
        if selected_pmid:
            selected_candidate = next(
                (candidate for candidate in selection_candidates if str(candidate.get("pmid") or "").strip() == selected_pmid),
                None,
            )
            execution = self._execute_calculator(
                job,
                selected_pmid,
                selected_candidate=selected_candidate,
            )
            self._record_tool_call(
                "python_calculator_executor",
                status=str(execution.get("status") or "unknown"),
                input_payload={
                    "mode": job.mode,
                    "pmid": selected_pmid,
                    "title": selected_tool.get("title"),
                    "clinical_text": summarize_text(job.text),
                },
                output_payload=execution,
                metadata={"dispatch": "question_mode", "source": "post_selection_execution"},
            )
            execution = dict(execution)
            executions.append(dict(execution))
        if not execution:
            execution = {
                "status": "skipped",
                "final_text": "No calculator was selected from the second-stage candidate pool.",
            }
        return {
            "mode": job.mode,
            "retrieved_tools": list(selection_bundle.get("retrieved_tools") or []),
            "candidate_ranking": list(selection_bundle.get("candidate_ranking") or []),
            "retrieval_batches": list(selection_bundle.get("retrieval_batches") or []),
            "bm25_raw_top5": list(selection_bundle.get("bm25_raw_top5") or []),
            "vector_raw_top5": list(selection_bundle.get("vector_raw_top5") or []),
            "selection_decisions": [],
            "eligible_tools": [],
            "selected_tool": selected_tool,
            "executions": executions,
            "execution": execution,
            "recommended_pmids": list(selection_bundle.get("recommended_pmids") or []),
            "selection_mode": str(selection_bundle.get("selection_mode") or "model_selected"),
        }

    def _build_forced_question_selection(
        self,
        job: ClinicalToolJob,
        retrieved: list[dict[str, Any]],
        *,
        forced_tool_pmid: str,
    ) -> dict[str, Any]:
        selected_tool_id = str(forced_tool_pmid or "").strip()
        selected_candidate = next(
            (tool for tool in retrieved if str(tool.get("pmid") or "").strip() == selected_tool_id),
            None,
        )
        selected_doc = self.catalog.get(selected_tool_id)
        parameter_names = self._resolve_candidate_parameter_names(
            pmid=selected_tool_id,
            candidate=selected_candidate,
        )
        selection = {
            "pmid": selected_tool_id,
            "title": selected_doc.title.strip(),
            "purpose": selected_doc.purpose.strip(),
            "parameter_names": parameter_names,
            "model_parameter_names": [],
            "model_selected_tool_id": None,
            "fallback_used": False,
            "reason": "Forced oracle tool selection for evaluation.",
            "raw_response": None,
            "present_in_retrieved_candidates": bool(selected_candidate),
        }
        self._record_tool_call(
            "riskcalc_selector",
            input_payload={
                "mode": job.mode,
                "clinical_text": summarize_text(job.text),
                "candidate_tools": retrieved,
            },
            output_payload=selection,
            metadata={"stage": "question_selection", "selection_mode": "oracle_forced"},
        )
        return selection

    def _run_patient_note(self, job: ClinicalToolJob) -> dict[str, Any]:
        if self._should_use_preselected_path(job):
            return self._run_preselected(job)

        selection_bundle = self.plan_selection(job)
        risk_hints = list(selection_bundle.get("risk_hints") or [])
        selection_candidates = list(selection_bundle.get("selection_candidates") or [])
        selected_tool = dict(selection_bundle.get("selected_tool") or {})

        executions: list[dict[str, Any]] = []
        selected_pmid = str(selected_tool.get("pmid") or "").strip()
        selected_execution: dict[str, Any] = {}
        if selected_pmid:
            selected_candidate = next(
                (candidate for candidate in selection_candidates if str(candidate.get("pmid") or "").strip() == selected_pmid),
                None,
            )
            execution = self._execute_calculator(
                job,
                selected_pmid,
                selected_candidate=selected_candidate,
            )
            self._record_tool_call(
                "python_calculator_executor",
                status=str(execution.get("status") or "unknown"),
                input_payload={
                    "mode": job.mode,
                    "pmid": selected_pmid,
                    "title": selected_tool.get("title"),
                    "clinical_text": summarize_text(job.text),
                },
                output_payload=execution,
                metadata={"dispatch": "patient_note_mode", "source": "post_selection_execution"},
            )
            execution = dict(execution)
            executions.append(execution)
            selected_execution = dict(execution)
        if not selected_execution:
            selected_execution = {
                "status": "skipped",
                "final_text": "No calculator was selected from the second-stage candidate pool.",
            }

        return {
            "mode": job.mode,
            "risk_hints": risk_hints,
            "retrieval_batches": list(selection_bundle.get("retrieval_batches") or []),
            "retrieved_tools": list(selection_bundle.get("retrieved_tools") or []),
            "candidate_ranking": list(selection_bundle.get("candidate_ranking") or []),
            "recommended_pmids": list(selection_bundle.get("recommended_pmids") or []),
            "bm25_raw_top5": list(selection_bundle.get("bm25_raw_top5") or []),
            "vector_raw_top5": list(selection_bundle.get("vector_raw_top5") or []),
            "selection_decisions": [],
            "eligible_tools": [],
            "selected_tool": selected_tool,
            "executions": executions,
            "execution": selected_execution,
            "selection_mode": str(selection_bundle.get("selection_mode") or "model_selected"),
        }

    @staticmethod
    def _compute_patient_note_selection_limit(*, total_candidates: int, candidate_budget: int) -> int:
        if total_candidates <= 0:
            return 0
        if candidate_budget <= 0:
            return min(total_candidates, _PATIENT_NOTE_MIN_SELECTION_SCAN)
        return min(
            total_candidates,
            max(
                candidate_budget,
                candidate_budget * _PATIENT_NOTE_SELECTION_SCAN_MULTIPLIER,
                candidate_budget + _PATIENT_NOTE_SELECTION_EXTRA_BUFFER,
                _PATIENT_NOTE_MIN_SELECTION_SCAN,
            ),
        )

    def _build_candidate_alignment_text(self, pmid: str) -> str:
        doc = self.catalog.get(pmid)
        parts = [
            doc.title,
            doc.purpose,
            doc.eligibility,
            doc.utility,
            doc.abstract,
            doc.specialty,
        ]
        return " ".join(part for part in parts if str(part or "").strip())

    def _resolve_candidate_parameter_names(
        self,
        *,
        pmid: str,
        candidate: dict[str, Any] | None = None,
    ) -> list[str]:
        normalized_pmid = str(pmid or "").strip()
        if normalized_pmid and self.registry is not None and self.registry.has(normalized_pmid):
            registration = self.registry.get(normalized_pmid)
            return [str(name).strip() for name in registration.parameter_names if str(name or "").strip()]

        raw_parameter_names = []
        if isinstance(candidate, dict):
            raw_parameter_names = list(candidate.get("parameter_names") or [])

        normalized_parameter_names: list[str] = []
        for name in raw_parameter_names:
            normalized_name = str(name or "").strip()
            if normalized_name and normalized_name not in normalized_parameter_names:
                normalized_parameter_names.append(normalized_name)
        return normalized_parameter_names

    def _build_selected_calculator_payload(self, registration: RiskCalcRegistration) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        try:
            document = self.catalog.get(registration.calc_id)
        except Exception:
            document = None

        if document is not None and hasattr(document, "to_dict"):
            try:
                payload = dict(document.to_dict() or {})
            except Exception:
                payload = {}

        if not payload:
            payload = {
                "pmid": registration.calc_id,
                "title": registration.title,
                "purpose": registration.purpose,
                "eligibility": registration.eligibility,
                "interpretation": registration.interpretation,
                "utility": registration.utility,
                "computation": registration.code,
            }

        payload["pmid"] = registration.calc_id
        payload["calc_id"] = registration.calc_id
        payload.setdefault("title", registration.title)
        payload.setdefault("purpose", registration.purpose)
        payload.setdefault("eligibility", registration.eligibility)
        payload.setdefault("interpretation", registration.interpretation)
        payload.setdefault("utility", registration.utility)
        payload.setdefault("function_name", registration.function_name)
        payload.setdefault("parameter_names", list(registration.parameter_names))
        payload.setdefault("computation", registration.code)
        return payload

    def _resolve_parameter_extraction_query_text(
        self,
        job: ClinicalToolJob,
        *,
        selected_pmid: str,
        selected_candidate: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        normalized_pmid = str(selected_pmid or "").strip()
        structured_case = dict(job.structured_case or {})
        source_entries = dict((selected_candidate or {}).get("source_entries") or {})
        query_candidates: list[tuple[str, str]] = []

        def _append_candidate(label: str, raw_text: Any) -> None:
            text = str(raw_text or "").strip()
            if text:
                query_candidates.append((label, text))

        _append_candidate("job.dispatch_query_text", job.dispatch_query_text)
        _append_candidate(
            "job.selection_context.dispatch_query_text",
            dict(job.selection_context or {}).get("dispatch_query_text"),
        )
        coarse_bundle = dict(self._latest_coarse_retrieval_bundle or {})
        _append_candidate("coarse_retrieval.query_text", coarse_bundle.get("query_text"))
        _append_candidate(
            "selected_candidate.query_text",
            (selected_candidate or {}).get("query_text"),
        )
        for source_name in ("retrieved_union", "bm25", "vector", "legacy_question_pool"):
            source_entry = dict(source_entries.get(source_name) or {})
            if normalized_pmid and str(source_entry.get("pmid") or "").strip() not in {"", normalized_pmid}:
                continue
            _append_candidate(f"selected_candidate.source_entries.{source_name}", source_entry.get("query_text"))

        for index, query in enumerate(list(job.retrieval_queries or []), start=1):
            _append_candidate(f"job.retrieval_queries[{index}]", getattr(query, "text", ""))

        _append_candidate("job.case_summary", job.case_summary)
        _append_candidate("structured_case.case_summary", structured_case.get("case_summary"))
        _append_candidate("job.text", job.text)
        _append_candidate("structured_case.raw_text", structured_case.get("raw_text"))

        for source_name, query_text in query_candidates:
            if query_text:
                return query_text, source_name
        return "", "unavailable"

    def _score_query_alignment(
        self,
        query_text: str,
        candidate_token_set: set[str],
        candidate_bigram_set: set[str],
    ) -> float:
        token_weights, bigram_weights = _build_query_alignment_signature(query_text)
        total_weight = float(sum(token_weights.values()) + sum(bigram_weights.values()))
        if total_weight <= 0.0:
            return 0.0

        matched_weight = 0.0
        for token, weight in token_weights.items():
            if token in candidate_token_set:
                matched_weight += float(weight)
        for bigram, weight in bigram_weights.items():
            if bigram in candidate_bigram_set:
                matched_weight += float(weight)
        support_score = matched_weight / total_weight
        specificity_factor = min(total_weight / 4.0, 1.0)
        return support_score * specificity_factor

    def _retrieve_and_rank_candidates(self, job: ClinicalToolJob) -> dict[str, Any]:
        retrieval_queries, risk_hints = self._resolve_retrieval_queries(job)
        retrieval_batches = []
        retrieved_tools: list[dict[str, Any]] = []
        candidate_pmids = None

        coarse_retrieved = self._retrieve_coarse_candidates(
            job,
            candidate_pmids=candidate_pmids,
        )
        if coarse_retrieved:
            coarse_bundle = dict(self._latest_retrieval_bundle or {})
            self._latest_coarse_retrieval_bundle = dict(coarse_bundle)
            self._record_tool_call(
                "riskcalc_coarse_retriever",
                input_payload={
                    "structured_case": dict(job.structured_case or {}),
                    "top_k": job.top_k,
                },
                output_payload={
                    "result_count": len(coarse_retrieved),
                    "tools": [dict(candidate) for candidate in coarse_retrieved],
                },
                metadata={"backend": str(coarse_bundle.get("backend_used") or job.retriever_backend)},
            )
            retrieval_batches.append(
                {
                    "stage": "coarse_recall",
                    "query": str(coarse_bundle.get("query_text") or job.text),
                    "intent": "coarse_recall",
                    "channel": "coarse_recall",
                    "backend": str(coarse_bundle.get("backend_used") or job.retriever_backend),
                    "priority": 1,
                    "tools": [dict(candidate) for candidate in coarse_retrieved],
                }
            )

            coarse_candidate_pmids = [
                str(candidate.get("pmid") or "").strip()
                for candidate in coarse_retrieved
                if str(candidate.get("pmid") or "").strip()
            ]
            rerank_pool_size = max(len(coarse_candidate_pmids), 1)
            refined_retrieved = self._retrieve_computation_candidates(
                job,
                candidate_pmids=coarse_candidate_pmids,
                top_k_per_channel=rerank_pool_size,
            )
            if refined_retrieved:
                refined_bundle = dict(self._latest_retrieval_bundle or {})
                self._record_tool_call(
                    "riskcalc_computation_retriever",
                    input_payload={
                        "structured_case": dict(job.structured_case or {}),
                        "candidate_pmids": list(coarse_candidate_pmids),
                        "top_k_per_channel": rerank_pool_size,
                    },
                    output_payload={
                        "result_count": len(refined_retrieved),
                        "tools": [
                            _public_candidate_view(dict(candidate), rank=index)
                            for index, candidate in enumerate(refined_retrieved, start=1)
                        ],
                    },
                    metadata={"backend": str(refined_bundle.get("backend_used") or "bm25")},
                )
                retrieval_batches.append(
                    {
                        "stage": "computation_parameter_match",
                        "query": str(refined_bundle.get("query_text") or job.text),
                        "intent": "computation_parameter_match",
                        "channel": "computation_parameter_match",
                        "backend": str(refined_bundle.get("backend_used") or "bm25"),
                        "priority": 2,
                        "tools": [
                            _public_candidate_view(dict(candidate), rank=index)
                            for index, candidate in enumerate(refined_retrieved, start=1)
                        ],
                    }
                )
                retrieved_tools = list(refined_retrieved)
            else:
                retrieved_tools = self._hydrate_coarse_candidates(coarse_retrieved)
        else:
            selection_budget = max(int(job.top_k), 1)
            latest_bundle: dict[str, Any] = {}
            for query in retrieval_queries:
                query_text = str(query.text or "").strip()
                if not query_text:
                    continue

                bundle_builder = getattr(self.retriever, "_build_retrieval_bundle", None)
                if callable(bundle_builder):
                    try:
                        bundle = bundle_builder(
                            query_text=query_text,
                            top_k=selection_budget,
                            candidate_pmids=candidate_pmids,
                            backend=job.retriever_backend,
                        )
                    except TypeError:
                        bundle = bundle_builder(
                            query_text=query_text,
                            top_k=selection_budget,
                            candidate_pmids=candidate_pmids,
                        )
                    latest_bundle = dict(bundle or {})
                    current_retrieved = [
                        dict(row)
                        for row in list(
                            latest_bundle.get("candidate_ranking") or latest_bundle.get("retrieved_tools") or []
                        )
                    ]
                else:
                    retrieve = getattr(self.retriever, "retrieve", None)
                    if not callable(retrieve):
                        continue
                    try:
                        rows = retrieve(
                            query_text,
                            top_k=selection_budget,
                            candidate_pmids=candidate_pmids,
                            backend=job.retriever_backend,
                        )
                    except TypeError:
                        try:
                            rows = retrieve(
                                query_text,
                                top_k=selection_budget,
                                candidate_pmids=candidate_pmids,
                            )
                        except TypeError:
                            rows = retrieve(query_text, top_k=selection_budget)
                    current_retrieved = [dict(row) for row in list(rows or [])]
                    latest_bundle = {
                        "query_text": query_text,
                        "backend_used": job.retriever_backend,
                        "available_backends": list(getattr(self.retriever, "available_backends", ()) or []),
                        "bm25_raw_top5": [],
                        "vector_raw_top5": [],
                        "model_context": {"bm25_top5": {}, "vector_top5": {}},
                        "retrieved_tools": list(current_retrieved),
                        "candidate_ranking": list(current_retrieved),
                    }

                if latest_bundle:
                    self._latest_retrieval_bundle = dict(latest_bundle)
                retrieval_batches.append(
                    {
                        "stage": str(query.stage or "query_retrieval"),
                        "query": query_text,
                        "intent": str(query.intent or "general"),
                        "tools": [
                            _public_candidate_view(dict(candidate), rank=index)
                            for index, candidate in enumerate(current_retrieved, start=1)
                        ],
                    }
                )
                if current_retrieved and not retrieved_tools:
                    retrieved_tools = list(current_retrieved)

        retrieval_bundle = dict(self._latest_retrieval_bundle or {})
        bm25_raw_top3 = list(retrieval_bundle.get("bm25_raw_top3") or [])
        vector_raw_top3 = list(retrieval_bundle.get("vector_raw_top3") or [])
        bm25_raw_top5 = list(retrieval_bundle.get("bm25_raw_top5") or bm25_raw_top3 or [])
        vector_raw_top5 = list(retrieval_bundle.get("vector_raw_top5") or vector_raw_top3 or [])
        if bm25_raw_top3 or vector_raw_top3:
            retrieved_tools = [
                dict(row)
                for row in list(retrieval_bundle.get("candidate_ranking") or retrieval_bundle.get("retrieved_tools") or retrieved_tools)
            ]
        else:
            retrieved_tools = self._build_question_selection_candidates(
                retrieved_tools=retrieved_tools,
                bm25_raw_top5=bm25_raw_top5,
                vector_raw_top5=vector_raw_top5,
                extra_candidates=list(retrieval_bundle.get("candidate_ranking") or []),
                limit=_QUESTION_SELECTION_POOL_LIMIT,
            )
        return {
            "risk_hints": risk_hints,
            "retrieval_queries": retrieval_queries,
            "retrieval_batches": retrieval_batches,
            "bm25_raw_top3": bm25_raw_top3,
            "vector_raw_top3": vector_raw_top3,
            "bm25_raw_top5": bm25_raw_top5,
            "vector_raw_top5": vector_raw_top5,
            "question_selection_candidates": list(retrieved_tools),
            "retrieved_tools": list(retrieved_tools),
            "candidate_ranking": list(retrieved_tools),
        }

    @staticmethod
    def _build_question_selection_candidates(
        *,
        retrieved_tools: list[dict[str, Any]],
        bm25_raw_top5: list[dict[str, Any]],
        vector_raw_top5: list[dict[str, Any]],
        extra_candidates: list[dict[str, Any]] | None = None,
        limit: int = _QUESTION_SELECTION_POOL_LIMIT,
    ) -> list[dict[str, Any]]:
        max_candidates = max(int(limit), 1)
        ordered_candidates: list[dict[str, Any]] = []
        by_pmid: dict[str, dict[str, Any]] = {}

        def _merge_candidate(
            raw_candidate: dict[str, Any] | None,
            *,
            source_name: str,
            source_entry: dict[str, Any] | None = None,
            allow_append: bool,
        ) -> None:
            candidate_payload = dict(raw_candidate or {})
            pmid = str(
                candidate_payload.get("pmid")
                or (source_entry or {}).get("pmid")
                or ""
            ).strip()
            if not pmid:
                return

            calculator_payload = dict((source_entry or {}).get("calculator_payload") or {})
            candidate = by_pmid.get(pmid)
            if candidate is None:
                if not allow_append or len(ordered_candidates) >= max_candidates:
                    return
                candidate = {
                    "pmid": pmid,
                    "title": "",
                    "purpose": "",
                    "eligibility": "",
                    "example": "",
                    "parameter_names": [],
                    "calculator_payload": {},
                    "source_channels": [],
                    "source_entries": {},
                }
                by_pmid[pmid] = candidate
                ordered_candidates.append(candidate)

            candidate["title"] = str(
                candidate.get("title")
                or candidate_payload.get("title")
                or calculator_payload.get("title")
                or (source_entry or {}).get("title")
                or ""
            ).strip()
            candidate["purpose"] = str(
                candidate.get("purpose")
                or candidate_payload.get("purpose")
                or calculator_payload.get("purpose")
                or (source_entry or {}).get("purpose")
                or ""
            ).strip()
            candidate["eligibility"] = str(
                candidate.get("eligibility")
                or candidate_payload.get("eligibility")
                or calculator_payload.get("eligibility")
                or (source_entry or {}).get("eligibility")
                or ""
            ).strip()
            candidate["example"] = str(
                candidate.get("example")
                or candidate_payload.get("example")
                or calculator_payload.get("example")
                or (source_entry or {}).get("example")
                or ""
            ).strip()
            if not candidate.get("parameter_names"):
                candidate["parameter_names"] = list(
                    candidate_payload.get("parameter_names")
                    or (source_entry or {}).get("parameter_names")
                    or []
                )
            if calculator_payload and not candidate.get("calculator_payload"):
                candidate["calculator_payload"] = calculator_payload
            if source_name not in list(candidate.get("source_channels") or []):
                candidate["source_channels"] = list(candidate.get("source_channels") or []) + [source_name]
            if source_entry is not None:
                source_entries = dict(candidate.get("source_entries") or {})
                source_entries[source_name] = dict(source_entry)
                candidate["source_entries"] = source_entries

        for candidate in list(retrieved_tools or [])[:max_candidates]:
            _merge_candidate(
                candidate,
                source_name="retrieved_union",
                allow_append=True,
            )

        if not ordered_candidates:
            for candidate in list(extra_candidates or [])[:max_candidates]:
                _merge_candidate(
                    candidate,
                    source_name="legacy_question_pool",
                    allow_append=True,
                )

        for channel, raw_hits in (("bm25", bm25_raw_top5), ("vector", vector_raw_top5)):
            for entry in list(raw_hits or []):
                _merge_candidate(
                    dict(entry.get("calculator_payload") or {}),
                    source_name=channel,
                    source_entry=dict(entry),
                    allow_append=True,
                )

        return ordered_candidates[:max_candidates]

    def _accumulate_retrieved_batch(
        self,
        candidates: dict[str, dict[str, Any]],
        candidate_alignment_cache: dict[str, tuple[set[str], set[str]]],
        *,
        query_text: str,
        query_stage: str,
        query_channel: str,
        query_priority: int,
        retrieved: list[dict[str, Any]],
    ) -> None:
        for rank, tool in enumerate(retrieved, start=1):
            pmid = str(tool["pmid"])
            current = candidates.setdefault(
                pmid,
                {
                    "pmid": pmid,
                    "title": tool.get("title", ""),
                    "purpose": tool.get("purpose", ""),
                    "count": 0,
                    "best_score": 0.0,
                    "best_priority": query_priority,
                    "best_priority_score": 0.0,
                    "best_stage": query_stage,
                    "channel_scores": {},
                    "channel_hits": {},
                },
            )
            score = float(tool.get("score") or 0.0)
            current["count"] += 1
            current["best_score"] = max(float(current["best_score"]), score)
            candidate_token_set, candidate_bigram_set = candidate_alignment_cache.setdefault(
                pmid,
                _build_alignment_features(self._build_candidate_alignment_text(pmid)),
            )
            alignment_score = self._score_query_alignment(
                query_text,
                candidate_token_set,
                candidate_bigram_set,
            )
            if query_priority < int(current["best_priority"]):
                current["best_priority"] = query_priority
                current["best_priority_score"] = score
                current["best_stage"] = query_stage
            elif query_priority == int(current["best_priority"]):
                if score > float(current["best_priority_score"]):
                    current["best_priority_score"] = score
                    current["best_stage"] = query_stage
            channel_scores = dict(current.get("channel_scores") or {})
            channel_hits = dict(current.get("channel_hits") or {})
            channel_alignment_max = dict(current.get("channel_alignment_max") or {})
            priority_factor = 1.0 / max(float(query_priority), 1.0) ** 0.5
            channel_weight = _QUERY_CHANNEL_WEIGHTS.get(query_channel, _QUERY_CHANNEL_WEIGHTS["general"])
            if query_channel == "case_summary":
                alignment_floor = 0.6
            elif query_channel == "parameter_match":
                alignment_floor = 0.5
            elif query_channel in {"problem_anchor", "risk_hint"}:
                alignment_floor = 0.15
            else:
                alignment_floor = 0.3
            contribution = (
                channel_weight
                * priority_factor
                * (1.0 / (_FUSION_RRF_K + rank))
                * (alignment_floor + ((1.0 - alignment_floor) * alignment_score))
            )
            channel_scores[query_channel] = float(channel_scores.get(query_channel) or 0.0) + contribution
            channel_hits[query_channel] = int(channel_hits.get(query_channel) or 0) + 1
            channel_alignment_max[query_channel] = max(
                float(channel_alignment_max.get(query_channel) or 0.0),
                alignment_score,
            )
            current["channel_scores"] = channel_scores
            current["channel_hits"] = channel_hits
            current["channel_alignment_max"] = channel_alignment_max

    def _coerce_bound_tool_rows(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            self._latest_retrieval_bundle = dict(payload)
            candidate_rows = (
                payload.get("retrieved_tools")
                or payload.get("candidate_ranking")
                or payload.get("bm25_candidates")
                or []
            )
            return [dict(row) for row in list(candidate_rows)]
        return [dict(row) for row in list(payload or [])]

    def _retrieve_coarse_candidates(
        self,
        job: ClinicalToolJob,
        *,
        candidate_pmids: set[str] | None,
    ) -> list[dict[str, Any]]:
        structured_case = dict(job.structured_case or {})
        if not structured_case:
            return []

        retrieve_tool = self.tools.get("riskcalc_coarse_retriever")
        tool_invoke = getattr(retrieve_tool, "invoke", None)
        if not callable(tool_invoke):
            return []
        kwargs = {"candidate_pmids": candidate_pmids} if candidate_pmids is not None else {}
        retrieved = tool_invoke(job, **kwargs)
        retrieval_rows = self._coerce_bound_tool_rows(retrieved)
        if candidate_pmids is None:
            return retrieval_rows
        return [row for row in retrieval_rows if str(row.get("pmid") or "").strip() in candidate_pmids]

    def _retrieve_computation_candidates(
        self,
        job: ClinicalToolJob,
        *,
        candidate_pmids: list[str],
        top_k_per_channel: int,
    ) -> list[dict[str, Any]]:
        structured_case = dict(job.structured_case or {})
        if not structured_case or not candidate_pmids:
            return []

        retrieve_tool = self.tools.get("riskcalc_computation_retriever")
        tool_invoke = getattr(retrieve_tool, "invoke", None)
        if not callable(tool_invoke):
            return []

        retrieved = tool_invoke(
            job,
            candidate_pmids=list(candidate_pmids),
            top_k_per_channel=max(int(top_k_per_channel), 1),
        )
        return self._coerce_bound_tool_rows(retrieved)

    def _hydrate_coarse_candidates(self, coarse_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        hydrated: list[dict[str, Any]] = []
        for row in list(coarse_candidates or []):
            pmid = str(row.get("pmid") or "").strip()
            if not pmid:
                continue
            document = self.catalog.get(pmid)
            hydrated.append(
                {
                    "pmid": pmid,
                    "title": str(row.get("title") or getattr(document, "title", "") or "").strip(),
                    "purpose": str(getattr(document, "purpose", "") or "").strip(),
                    "specialty": str(getattr(document, "specialty", "") or "").strip(),
                    "eligibility": str(getattr(document, "eligibility", "") or "").strip(),
                    "parameter_names": [],
                    "source_channels": ["coarse"],
                }
            )
        return hydrated

    def _resolve_retrieval_queries(self, job: ClinicalToolJob) -> tuple[list[RetrievalQuery], list[str]]:
        if job.retrieval_queries:
            queries = [query for query in job.retrieval_queries if str(query.text).strip()]
            if job.mode == "patient_note":
                risk_queries = [query for query in queries if is_risk_hint_query(query)]
                if risk_queries:
                    risk_hints = self._extract_risk_hints_from_queries(job, risk_queries)
                    return queries, risk_hints[: max(job.risk_count, 1)]
                return queries, list(job.risk_hints)[: max(job.risk_count, 1)]

            risk_hints = self._extract_risk_hints_from_queries(job, queries)
            return queries, risk_hints

        if job.mode == "patient_note":
            risk_hints = list(job.risk_hints) or self._generate_risk_hints(job)
            structured_case = dict(job.structured_case or {})
            problem_list = _coerce_text_list(structured_case.get("problem_list"))
            case_summary = str(job.case_summary or structured_case.get("case_summary") or "").strip()
            queries = build_patient_note_queries(
                case_summary=case_summary,
                risk_hints=risk_hints,
                problem_list=problem_list,
                risk_count=job.risk_count,
            )
            return queries, risk_hints

        return (
            [
                RetrievalQuery(
                    stage="direct_question",
                    text=job.text,
                    intent="clinical_question",
                    rationale="Fallback direct retrieval path when no staged queries were prepared upstream.",
                    priority=1,
                )
            ],
            [],
        )

    @staticmethod
    def _extract_risk_hints_from_queries(job: ClinicalToolJob, queries: list[RetrievalQuery]) -> list[str]:
        if job.risk_hints:
            return list(job.risk_hints)
        return extract_risk_hints_from_queries(queries)

    def _generate_risk_hints(self, job: ClinicalToolJob) -> list[str]:
        hints, answer = generate_risk_hints(
            clinical_text=job.text,
            risk_count=job.risk_count,
            chat_client=self.chat_client,
            model=job.llm_model,
            temperature=job.temperature,
        )
        self._record_tool_call(
            "risk_hint_generator",
            input_payload={
                "clinical_text": summarize_text(job.text),
                "risk_count": job.risk_count,
            },
            output_payload={
                "risk_hints": hints,
                "raw_response": answer,
            },
        )
        return hints

    def _select_tool_for_question(
        self,
        job: ClinicalToolJob,
        retrieved: list[dict[str, Any]],
        *,
        selection_decisions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        del selection_decisions
        if not retrieved:
            selection = {
                "pmid": "",
                "title": "",
                "purpose": "",
                "parameter_names": [],
                "model_parameter_names": [],
                "model_selected_tool_id": None,
                "fallback_used": False,
                "reason": "No candidate calculator was available for second-stage selection.",
                "raw_response": "",
            }
            self._record_tool_call(
                "riskcalc_selector",
                input_payload={
                    "mode": job.mode,
                    "clinical_text": summarize_text(job.text),
                    "candidate_tools": [],
                },
                output_payload=selection,
                metadata={"stage": "question_selection", "selection_pool": "empty_candidate_pool"},
            )
            return selection

        prompt_lines = [
            "Please choose the most appropriate clinical calculator for the question below.",
            job.text,
            "",
            "You must choose from the provided second-stage candidate pool only.",
            "Do not invent a new calculator and do not rely on any shortlist outside this pool.",
            "Do not pre-execute calculators or infer cross-candidate inputs before selection.",
            "If you return parameter_names, they must belong to the single selected calculator only.",
        ]
        retrieval_bundle = dict(self._latest_retrieval_bundle or {})
        raw_bm25_top3 = list(retrieval_bundle.get("bm25_raw_top3") or [])
        raw_vector_top3 = list(retrieval_bundle.get("vector_raw_top3") or [])
        raw_bm25_top5 = list(raw_bm25_top3 or retrieval_bundle.get("bm25_raw_top5") or [])
        raw_vector_top5 = list(raw_vector_top3 or retrieval_bundle.get("vector_raw_top5") or [])
        retrieval_window_label = "top-3" if raw_bm25_top3 or raw_vector_top3 else "top-5"
        prompt_lines.append(
            f"You will receive up to {_QUESTION_SELECTION_POOL_LIMIT} second-stage candidates, assembled from the raw BM25 {retrieval_window_label} calculators and raw vector {retrieval_window_label} calculators."
        )
        recommended_channels = _build_recommended_channels(
            bm25_raw_top5=raw_bm25_top5,
            vector_raw_top5=raw_vector_top5,
        )
        model_context = retrieval_bundle.get("model_context")
        if isinstance(model_context, dict) and model_context:
            prompt_lines.extend(
                [
                    "",
                    "Retriever model context:",
                    json.dumps(model_context, ensure_ascii=False, indent=2),
                    "",
                    "Use the BM25 and vector retrieval context below to compare candidate calculators.",
                ]
            )
        allowed_pmids = [
            str(tool.get("pmid") or "").strip()
            for tool in retrieved
            if str(tool.get("pmid") or "").strip()
        ]
        def _build_selector_view(raw_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
            view: list[dict[str, Any]] = []
            for item in list(raw_hits):
                pmid = str(item.get("pmid") or "").strip()
                if not pmid or pmid not in allowed_pmids:
                    continue
                view.append(
                    {
                        "pmid": pmid,
                        "channel": item.get("channel"),
                        "calculator_payload": dict(item.get("calculator_payload") or {}),
                    }
                )
            return view

        def _build_candidate_pool_view(raw_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
            view: list[dict[str, Any]] = []
            for item in list(raw_candidates):
                pmid = str(item.get("pmid") or "").strip()
                if not pmid:
                    continue
                view.append(
                    {
                        "pmid": pmid,
                        "title": str(item.get("title") or "").strip(),
                        "purpose": str(item.get("purpose") or "").strip(),
                        "eligibility": str(item.get("eligibility") or "").strip(),
                        "parameter_names": list(item.get("parameter_names") or []),
                        "source_channels": list(item.get("source_channels") or []),
                    }
                )
            return view

        prompt_lines.extend(
            [
                "",
                "Retrieved calculator union pool:",
                json.dumps(_build_candidate_pool_view(retrieved), ensure_ascii=False, indent=2),
                "",
                f"BM25 {retrieval_window_label} raw calculators:",
                json.dumps(_build_selector_view(raw_bm25_top5), ensure_ascii=False, indent=2),
                "",
                f"Vector {retrieval_window_label} raw calculators:",
                json.dumps(_build_selector_view(raw_vector_top5), ensure_ascii=False, indent=2),
                "",
                "Allowed calculator PMIDs:",
                json.dumps(allowed_pmids, ensure_ascii=False),
                "",
                "Choose exactly one PMID from the allowed list only.",
            ]
        )
        prompt_lines.append(
            'Return JSON: {"selected_tool_id": "<pmid>", "parameter_names": ["<param1>"], "reason": "<short reason>"}'
        )
        answer = self.chat_client.complete(
            [{"role": "user", "content": "\n".join(prompt_lines)}],
            model=job.llm_model,
            temperature=job.temperature,
        )
        payload = maybe_load_json(answer)
        selected_tool_id = None
        selection_reason = ""
        model_parameter_names: list[str] = []
        candidate_pmids = {
            str(tool.get("pmid") or "").strip()
            for tool in retrieved
            if str(tool.get("pmid") or "").strip()
        }
        if isinstance(payload, dict):
            selected_tool_id = str(payload.get("selected_tool_id") or "").strip()
            selection_reason = str(payload.get("reason") or "").strip()
            raw_parameter_names = payload.get("parameter_names", payload.get("required_parameters"))
            if isinstance(raw_parameter_names, str):
                raw_parameter_names = [part.strip() for part in raw_parameter_names.split(",")]
            if isinstance(raw_parameter_names, list):
                for name in raw_parameter_names:
                    normalized_name = str(name or "").strip()
                    if normalized_name and normalized_name not in model_parameter_names:
                        model_parameter_names.append(normalized_name)
        model_selected_tool_id = selected_tool_id
        if selected_tool_id and selected_tool_id not in candidate_pmids:
            selected_tool_id = ""
        if not selected_tool_id:
            for tool in retrieved:
                tool_pmid = str(tool["pmid"])
                if tool_pmid in answer:
                    selected_tool_id = str(tool["pmid"])
                    break
        if not selected_tool_id and retrieved:
            selected_tool_id = str(retrieved[0]["pmid"])
        if not selected_tool_id:
            selection = {
                "pmid": "",
                "title": "",
                "purpose": "",
                "parameter_names": [],
                "model_parameter_names": model_parameter_names,
                "model_selected_tool_id": model_selected_tool_id or None,
                "fallback_used": bool(model_selected_tool_id),
                "reason": selection_reason or "No valid calculator was selected from the second-stage candidate pool.",
                "raw_response": answer,
            }
            self._record_tool_call(
                "riskcalc_selector",
                input_payload={
                    "mode": job.mode,
                    "clinical_text": summarize_text(job.text),
                    "candidate_tools": retrieved,
                    "bm25_raw_top5": [
                        _public_context_hit_view(
                            dict(item),
                            recommended_channels=recommended_channels,
                        )
                        for item in raw_bm25_top5
                    ],
                    "vector_raw_top5": [
                        _public_context_hit_view(
                            dict(item),
                            recommended_channels=recommended_channels,
                        )
                        for item in raw_vector_top5
                    ],
                },
                output_payload=selection,
                metadata={"stage": "question_selection", "selection_pool": "no_valid_selection"},
            )
            return selection
        selected_candidate = next(
            (tool for tool in retrieved if str(tool.get("pmid") or "").strip() == selected_tool_id),
            None,
        )
        authoritative_parameter_names = self._resolve_candidate_parameter_names(
            pmid=selected_tool_id,
            candidate=selected_candidate,
        )
        if authoritative_parameter_names and model_parameter_names:
            normalized_model_parameter_names = {name.casefold() for name in model_parameter_names}
            validated_parameter_names = [
                name
                for name in authoritative_parameter_names
                if name.casefold() in normalized_model_parameter_names
            ]
            parameter_names = validated_parameter_names or authoritative_parameter_names
        else:
            parameter_names = authoritative_parameter_names or model_parameter_names
        selected_doc = self.catalog.get(selected_tool_id)
        selection = {
            "pmid": selected_tool_id,
            "title": selected_doc.title.strip(),
            "purpose": selected_doc.purpose.strip(),
            "parameter_names": parameter_names,
            "model_parameter_names": model_parameter_names,
            "model_selected_tool_id": model_selected_tool_id or None,
            "fallback_used": bool(model_selected_tool_id and model_selected_tool_id != selected_tool_id),
            "reason": selection_reason,
            "raw_response": answer,
        }
        self._record_tool_call(
            "riskcalc_selector",
            input_payload={
                "mode": job.mode,
                "clinical_text": summarize_text(job.text),
                "candidate_tools": retrieved,
                "bm25_raw_top5": [
                    _public_context_hit_view(
                        dict(item),
                        recommended_channels=recommended_channels,
                    )
                    for item in raw_bm25_top5
                ],
                "vector_raw_top5": [
                    _public_context_hit_view(
                        dict(item),
                        recommended_channels=recommended_channels,
                    )
                    for item in raw_vector_top5
                ],
            },
            output_payload=selection,
            metadata={"stage": "question_selection"},
        )
        return selection

    def _assess_patient_eligibility(self, job: ClinicalToolJob, pmid: str) -> dict[str, Any]:
        doc = self.catalog.get(pmid)
        case_label = "clinical question" if job.mode == "question" else "patient admission note"
        prompt = (
            f"Here is the {case_label}:\n"
            f"{job.text}\n\n"
            "Here is the calculator:\n"
            f"{json.dumps(doc.to_dict(), ensure_ascii=False, indent=2)}\n\n"
            'Return JSON: {"patient_eligible":"yes|no","missing_all_parameters":"yes|no","rationale":"..."}'
        )
        answer = self.chat_client.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a critical evaluator. Judge whether the case belongs to the eligible "
                        "population of the calculator and whether all parameters are missing. "
                        "Hard reject the calculator when the primary disease domain, target population, "
                        "or treatment context clearly mismatches the case. For example, reject a "
                        "glioblastoma prognostic tool for an atrial-fibrillation stroke-risk question."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=job.llm_model,
            temperature=job.temperature,
        )
        payload = maybe_load_json(answer)
        if not isinstance(payload, dict):
            payload = {}
        decision = {
            "pmid": pmid,
            "title": doc.title.strip(),
            "patient_eligible": _normalize_yes_no(payload.get("patient_eligible"), default="no"),
            "missing_all_parameters": _normalize_yes_no(payload.get("missing_all_parameters"), default="yes"),
            "rationale": str(payload.get("rationale") or answer).strip(),
        }
        self._record_tool_call(
            "riskcalc_selector",
            input_payload={
                "mode": job.mode,
                "pmid": pmid,
                "title": doc.title.strip(),
                "clinical_text": summarize_text(job.text),
            },
            output_payload=decision,
            metadata={"stage": "patient_eligibility"},
        )
        return decision

    def _assess_candidate_execution_gate(
        self,
        job: ClinicalToolJob,
        candidate: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        pmid = str(candidate.get("pmid") or "").strip()
        title = str(candidate.get("title") or "").strip()
        parameter_names = self._resolve_candidate_parameter_names(pmid=pmid, candidate=candidate)
        decision: dict[str, Any] = {
            "pmid": pmid,
            "title": title,
            "patient_eligible": "yes",
            "missing_all_parameters": "no",
            "gate_status": "failed_invalid_candidate",
            "execution_status": "failed",
            "parameter_names": list(parameter_names),
            "available_inputs": [],
            "missing_inputs": list(parameter_names),
            "defaults_used": {},
            "result": None,
            "rationale": "Candidate is missing a valid calculator PMID.",
        }
        if not pmid:
            return decision, None

        execution_tool = self.tools.get("riskcalc_executor")
        if execution_tool is None or self.registry is None or not execution_tool.has_registration(pmid):
            decision.update(
                patient_eligible="no",
                missing_all_parameters="yes" if parameter_names else "no",
                gate_status="failed_unregistered",
                execution_status="unregistered",
                rationale="Calculator is not registered for structured execution.",
            )
            return decision, None

        registration = execution_tool.get_registration(pmid)
        parameter_names = list(registration.parameter_names)
        decision["title"] = registration.title or title
        decision["parameter_names"] = parameter_names
        extracted_inputs = self._extract_registered_inputs(
            job,
            registration,
            selected_candidate=candidate,
        ) or {}
        available_inputs = sorted(str(name) for name in extracted_inputs.keys())
        missing_before_defaults = [
            name for name in parameter_names if name not in extracted_inputs
        ]
        decision["available_inputs"] = available_inputs
        decision["missing_inputs"] = list(missing_before_defaults)
        if not extracted_inputs:
            decision.update(
                patient_eligible="no",
                missing_all_parameters="yes" if parameter_names else "no",
                gate_status="failed_missing_inputs",
                execution_status="missing_inputs",
                rationale="No calculator parameters could be extracted from the case text.",
            )
            return decision, None

        execution = execution_tool.execute_registered(
            calculator=pmid,
            inputs=extracted_inputs,
        )
        self._record_tool_call(
            "registered_riskcalc_executor",
            status=str(execution.get("status") or "unknown"),
            input_payload={
                "calc_id": pmid,
                "title": registration.title,
                "inputs": extracted_inputs,
            },
            output_payload=execution,
            metadata={"stage": "candidate_execution_gate"},
        )

        execution_status = str(execution.get("status") or "failed").strip().lower()
        defaults_used = dict(execution.get("defaults_used") or {})
        missing_after_defaults = list(execution.get("missing_inputs") or [])
        decision["defaults_used"] = defaults_used
        decision["result"] = execution.get("result")
        decision["execution_status"] = execution_status
        if execution_status == "completed":
            final_text = self._summarize_registered_execution(job, registration, execution)
            normalized_execution = {
                "pmid": pmid,
                "title": registration.title,
                "status": "completed",
                "rounds": 1,
                "execution_mode": "registered",
                "function_name": registration.function_name,
                "inputs": execution.get("inputs") or {},
                "defaults_used": defaults_used,
                "result": execution.get("result"),
                "final_text": final_text,
                "interpretation": registration.interpretation,
                "messages": [],
            }
            decision.update(
                patient_eligible="yes",
                missing_all_parameters="no",
                gate_status="passed",
                missing_inputs=[],
                rationale="Calculator produced a concrete score from extracted inputs.",
            )
            return decision, normalized_execution

        if execution_status == "missing_inputs":
            missing_inputs = missing_after_defaults or missing_before_defaults or list(parameter_names)
            decision.update(
                patient_eligible="no",
                missing_all_parameters="yes" if len(missing_inputs) >= len(parameter_names) else "no",
                gate_status="failed_missing_inputs",
                missing_inputs=missing_inputs,
                rationale="Calculator could not run because required parameters are missing.",
            )
            return decision, None

        decision.update(
            patient_eligible="no",
            missing_all_parameters="no",
            gate_status="failed_execution",
            missing_inputs=missing_after_defaults,
            rationale=str(execution.get("error") or "Calculator execution failed."),
        )
        return decision, None

    def _extract_registered_inputs(
        self,
        job: ClinicalToolJob,
        registration: RiskCalcRegistration,
        *,
        selected_candidate: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        execution_tool = self.tools.get("riskcalc_executor")
        if execution_tool is None:
            return None

        calculator_payload = self._build_selected_calculator_payload(registration)
        structured_case = dict(job.structured_case or {})
        retrieval_query_text, retrieval_query_source = self._resolve_parameter_extraction_query_text(
            job,
            selected_pmid=registration.calc_id,
            selected_candidate=selected_candidate,
        )
        extraction = execution_tool.extract_inputs(
            calculator=registration.calc_id,
            clinical_text=job.text,
            structured_case=structured_case,
            calculator_payload=calculator_payload,
            retrieval_query_text=retrieval_query_text,
            llm_model=job.llm_model,
            temperature=job.temperature,
        )
        extracted_inputs = extraction.get("inputs")
        self._record_tool_call(
            "riskcalc_input_extractor",
            status="completed" if extracted_inputs else "missing_inputs",
            input_payload={
                "calc_id": registration.calc_id,
                "title": registration.title,
                "parameter_names": registration.parameter_names,
                "clinical_text": summarize_text(job.text),
                "structured_case": structured_case,
                "calculator_payload": calculator_payload,
                "retrieval_query_text": retrieval_query_text,
                "retrieval_query_source": retrieval_query_source,
            },
            output_payload={
                "inputs": extracted_inputs,
                "raw_response": extraction.get("raw_response"),
            },
        )
        if not isinstance(extracted_inputs, dict):
            return None
        return dict(extracted_inputs)

    def _summarize_registered_execution(
        self,
        job: ClinicalToolJob,
        registration: RiskCalcRegistration,
        execution: dict[str, Any],
    ) -> str:
        execution_tool = self.tools.get("riskcalc_executor")
        if execution_tool is None:
            return str(execution.get("result") or "").strip()

        summary = execution_tool.summarize_result(
            calculator=registration.calc_id,
            clinical_text=job.text,
            mode=job.mode,
            execution=execution,
            llm_model=job.llm_model,
            temperature=job.temperature,
        )
        final_text = str(summary.get("final_text") or "").strip() or str(execution.get("result") or "").strip()
        self._record_tool_call(
            "execution_result_summarizer",
            input_payload={
                "calc_id": registration.calc_id,
                "title": registration.title,
                "function_name": registration.function_name,
                "inputs": execution.get("inputs") or {},
                "result": execution.get("result"),
            },
            output_payload={
                "final_text": final_text,
                "raw_response": summary.get("raw_response"),
            },
        )
        return final_text

    def _try_registered_execution(
        self,
        job: ClinicalToolJob,
        pmid: str,
        *,
        selected_candidate: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        execution_tool = self.tools.get("riskcalc_executor")
        if execution_tool is None or self.registry is None:
            return None
        if not execution_tool.has_registration(pmid):
            return None

        registration = execution_tool.get_registration(pmid)
        extracted_inputs = self._extract_registered_inputs(
            job,
            registration,
            selected_candidate=selected_candidate,
        )
        if not extracted_inputs:
            return None

        execution = execution_tool.execute_registered(
            calculator=pmid,
            inputs=extracted_inputs,
        )
        self._record_tool_call(
            "registered_riskcalc_executor",
            status=str(execution.get("status") or "unknown"),
            input_payload={
                "calc_id": pmid,
                "inputs": extracted_inputs,
            },
            output_payload=execution,
        )
        if execution.get("status") != "completed":
            return None

        final_text = self._summarize_registered_execution(job, registration, execution)
        return {
            "pmid": pmid,
            "title": registration.title,
            "status": "completed",
            "rounds": 1,
            "execution_mode": "registered",
            "function_name": registration.function_name,
            "inputs": execution.get("inputs") or {},
            "defaults_used": execution.get("defaults_used") or {},
            "result": execution.get("result"),
            "final_text": final_text,
            "interpretation": registration.interpretation,
            "messages": [],
        }

    def _execute_calculator(
        self,
        job: ClinicalToolJob,
        pmid: str,
        *,
        selected_candidate: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        doc = self.catalog.get(pmid)
        registered_execution = self._try_registered_execution(
            job,
            pmid,
            selected_candidate=selected_candidate,
        )
        if registered_execution is not None:
            return registered_execution

        calculator_text = self.catalog.format_calculator_text(pmid)
        base_code = extract_python_code_blocks(calculator_text)

        if job.mode == "question":
            finish_prefix = "Answer: "
            system = (
                "You are a helpful assistant. Apply the medical calculator to solve the question. "
                'You may write Python, and the user will execute it. Finish with "Answer: ".'
            )
            prompt_subject = "question"
        else:
            finish_prefix = "Summary: "
            system = (
                "You are a helpful assistant. Apply the medical calculator to an imaginary patient and "
                'interpret the result. You may write Python, and the user will execute it. Finish with "Summary: ".'
            )
            prompt_subject = "patient information"

        prompt = (
            "Here is the calculator:\n"
            f"{calculator_text}\n\n"
            f"Here is the {prompt_subject}:\n"
            f"{job.text}\n\n"
            "Please apply this calculator. If needed, write Python scripts and print the results."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        for round_index in range(1, max(job.max_rounds, 1) + 1):
            answer = self.chat_client.complete(
                messages,
                model=job.llm_model,
                temperature=job.temperature,
            )
            self._record_tool_call(
                "llm_calculator_round",
                input_payload={
                    "pmid": pmid,
                    "round": round_index,
                    "message_count": len(messages),
                    "mode": job.mode,
                },
                output_payload={"assistant_message": answer},
            )
            messages.append({"role": "assistant", "content": answer})
            if finish_prefix in answer:
                return {
                    "pmid": pmid,
                    "title": doc.title.strip(),
                    "status": "completed",
                    "rounds": round_index,
                    "execution_mode": "legacy_llm_loop",
                    "final_text": answer.split(finish_prefix, 1)[-1].strip(),
                    "messages": messages,
                }

            code = extract_python_code_blocks(answer)
            if code:
                script = "\n\n".join(part for part in [base_code, code] if part.strip())
                output = execute_python_code(script)
                self._record_tool_call(
                    "python_code_runtime",
                    input_payload={
                        "pmid": pmid,
                        "round": round_index,
                        "script": script,
                    },
                    output_payload={"stdout_stderr": output or "[no output]"},
                )
                user_feedback = "I executed your code. Output:\n" + (output or "[no output]")
            else:
                user_feedback = f'If you are finished, reply with "{finish_prefix}" followed by the final result.'

            messages.append({"role": "user", "content": user_feedback})

        return {
            "pmid": pmid,
            "title": doc.title.strip(),
            "status": "failed",
            "rounds": job.max_rounds,
            "execution_mode": "legacy_llm_loop",
            "final_text": "Failed to finish within the round limit.",
            "messages": messages,
        }
