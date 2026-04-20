from __future__ import annotations

import json
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.retrieval import (
    FieldedBM25Index,
    MedCPTRetriever,
    build_structured_query_text,
    create_structured_retriever,
)

from .execution_tools import tool


_CACHE_LOCK = threading.RLock()
_TRIAL_CATALOG_CACHE: dict[tuple[str, str], "TrialCatalog"] = {}
_TRIAL_RETRIEVER_CACHE: dict[tuple[str, str, str], "TrialRetrievalTool"] = {}


def _normalize_whitespace(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _dedupe_texts(values: Iterable[Any]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in list(values or []):
        text = _normalize_whitespace(value)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


def _normalize_backend_name(value: str | None) -> str:
    normalized = str(value or "hybrid").strip().lower()
    if not normalized:
        normalized = "hybrid"
    return {
        "keyword": "bm25",
        "auto": "hybrid",
        "medcpt": "vector",
    }.get(normalized, normalized)


def _status_priority(status: str, enrollment_open: bool) -> int:
    normalized_status = str(status or "").strip().lower()
    if normalized_status == "trial_matched":
        return 3 if enrollment_open else 2
    if normalized_status == "manual_review":
        return 1
    return 0


def _normalize_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    raw_scores = {
        str(row.get("document_id") or row.get("nct_id") or "").strip(): float(row.get("score") or 0.0)
        for row in list(rows or [])
        if str(row.get("document_id") or row.get("nct_id") or "").strip()
    }
    if not raw_scores:
        return {}

    values = list(raw_scores.values())
    max_score = max(values)
    min_score = min(values)
    if max_score == min_score:
        if max_score <= 0.0:
            return {document_id: 0.0 for document_id in raw_scores}
        return {document_id: 1.0 for document_id in raw_scores}

    denominator = max_score - min_score
    return {
        document_id: (score - min_score) / denominator
        for document_id, score in raw_scores.items()
    }


def _resolve_trial_department_root(department_root: str | Path | None = None) -> Path:
    if department_root is not None:
        return Path(department_root).expanduser().resolve()

    project_root = Path(__file__).resolve().parents[2]
    candidates = (
        project_root / "数据" / "治疗方案",
        project_root / "data" / "治疗方案",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _build_trial_retrieval_text(payload: Mapping[str, Any]) -> str:
    lines: list[str] = []

    title = _normalize_whitespace(payload.get("name") or payload.get("brief_title") or payload.get("official_title"))
    if title:
        lines.append(f"title: {title}")

    status = _normalize_whitespace(payload.get("status"))
    overall_status = _normalize_whitespace(payload.get("overall_status"))
    if status or overall_status:
        lines.append(
            "trial status: "
            + "; ".join(
                item
                for item in (
                    f"medai={status}" if status else "",
                    f"overall={overall_status}" if overall_status else "",
                    f"enrollment_open={bool(payload.get('enrollment_open'))}",
                )
                if item
            )
        )

    department_tags = _dedupe_texts(payload.get("department_tags") or [])
    if department_tags:
        lines.append("department tags: " + ", ".join(department_tags))

    conditions = _dedupe_texts(payload.get("conditions") or [])
    if conditions:
        lines.append("conditions: " + ", ".join(conditions))

    mesh_terms = _dedupe_texts(payload.get("mesh_terms") or [])
    if mesh_terms:
        lines.append("mesh terms: " + ", ".join(mesh_terms))

    keywords = _dedupe_texts(payload.get("keywords") or [])
    if keywords:
        lines.append("keywords: " + ", ".join(keywords))

    interventions = _dedupe_texts(payload.get("interventions") or [])
    if interventions:
        lines.append("interventions: " + ", ".join(interventions))

    study_type = _normalize_whitespace(payload.get("study_type"))
    phase = _normalize_whitespace(payload.get("phase"))
    primary_purpose = _normalize_whitespace(payload.get("primary_purpose"))
    if study_type or phase or primary_purpose:
        lines.append(
            "study profile: "
            + "; ".join(item for item in (study_type, phase, primary_purpose) if item)
        )

    brief_summary = _normalize_whitespace(payload.get("brief_summary"))
    if brief_summary:
        lines.append(f"summary: {brief_summary}")

    status_reason = _normalize_whitespace(payload.get("status_reason"))
    if status_reason:
        lines.append(f"status reason: {status_reason}")

    eligibility_text = _normalize_whitespace(payload.get("eligibility_text"))
    if eligibility_text:
        lines.append(f"eligibility: {eligibility_text}")

    actions = _dedupe_texts(payload.get("actions") or [])
    if actions:
        lines.append("actions: " + " | ".join(actions))

    return "\n".join(line for line in lines if line.strip()).strip()


@dataclass(slots=True)
class TrialDocument:
    nct_id: str
    title: str
    summary: str = ""
    purpose: str = ""
    eligibility: str = ""
    abstract: str = ""
    retrieval_text: str = ""
    status: str = "manual_review"
    overall_status: str = ""
    enrollment_open: bool = False
    department_tag: str = ""
    department_role: str = ""
    department_tags: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    mesh_terms: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    interventions: list[str] = field(default_factory=list)
    phase: str = ""
    primary_purpose: str = ""
    status_reason: str = ""
    actions: list[str] = field(default_factory=list)
    source_payload: dict[str, Any] = field(default_factory=dict)
    pmid: str = field(init=False)

    def __post_init__(self) -> None:
        self.pmid = self.nct_id
        if not self.retrieval_text:
            self.retrieval_text = _build_trial_retrieval_text(self.source_payload)

    def to_brief(self) -> dict[str, Any]:
        return {
            "nct_id": self.nct_id,
            "title": self.title,
            "summary": self.summary,
            "purpose": self.purpose,
            "eligibility": self.eligibility,
            "status": self.status,
            "overall_status": self.overall_status,
            "enrollment_open": self.enrollment_open,
            "department_tag": self.department_tag,
            "department_tags": list(self.department_tags),
            "actions": list(self.actions),
            "status_reason": self.status_reason,
        }


class TrialCatalog:
    def __init__(
        self,
        *,
        documents: list[TrialDocument],
        department_index: Mapping[str, set[str]],
        preferred_department: str = "",
        department_root: Path,
    ) -> None:
        self._documents = list(documents)
        self._document_by_id = {document.nct_id: document for document in self._documents}
        self.department_index = {
            str(tag): set(values)
            for tag, values in dict(department_index or {}).items()
        }
        self.preferred_department = str(preferred_department or "").strip()
        self.department_root = Path(department_root).resolve()
        self.runtime_cache_key = f"trial-catalog:{self.department_root}:{self.preferred_department}"

    def documents(self) -> list[TrialDocument]:
        return list(self._documents)

    def get(self, nct_id: str) -> TrialDocument | None:
        return self._document_by_id.get(str(nct_id or "").strip())

    def dense_index_cache_paths(self, *, query_model_name: str, doc_model_name: str) -> None:
        del query_model_name, doc_model_name
        return None

    @classmethod
    def from_department_root(
        cls,
        department_root: str | Path | None = None,
        *,
        preferred_department: str = "",
    ) -> "TrialCatalog":
        resolved_root = _resolve_trial_department_root(department_root)
        cache_key = (str(resolved_root), str(preferred_department or "").strip())
        with _CACHE_LOCK:
            cached = _TRIAL_CATALOG_CACHE.get(cache_key)
        if cached is not None:
            return cached

        grouped_payloads: dict[str, list[dict[str, Any]]] = {}
        department_index: dict[str, set[str]] = {}
        if resolved_root.exists():
            for department_dir in sorted(path for path in resolved_root.iterdir() if path.is_dir()):
                department_tag = department_dir.name
                payload_path = department_dir / "treatment_trials.json"
                if not payload_path.exists():
                    continue
                department_index.setdefault(department_tag, set())
                try:
                    payload = json.loads(payload_path.read_text(encoding="utf-8"))
                except Exception:
                    continue

                for raw_nct_id, raw_trial in dict(payload or {}).items():
                    normalized_trial = dict(raw_trial or {})
                    nct_id = _normalize_whitespace(normalized_trial.get("nct_id") or raw_nct_id)
                    if not nct_id:
                        continue
                    normalized_trial["nct_id"] = nct_id
                    normalized_trial["department_tag"] = _normalize_whitespace(
                        normalized_trial.get("department_tag") or department_tag
                    )
                    normalized_trial["department_tags"] = _dedupe_texts(
                        [
                            normalized_trial.get("department_tag"),
                            *list(normalized_trial.get("department_tags") or []),
                        ]
                    )
                    normalized_trial["actions"] = _dedupe_texts(normalized_trial.get("actions") or [])
                    normalized_trial["linked_trials"] = _dedupe_texts(
                        [nct_id, *list(normalized_trial.get("linked_trials") or [])]
                    )
                    normalized_trial["conditions"] = _dedupe_texts(normalized_trial.get("conditions") or [])
                    normalized_trial["mesh_terms"] = _dedupe_texts(normalized_trial.get("mesh_terms") or [])
                    normalized_trial["keywords"] = _dedupe_texts(normalized_trial.get("keywords") or [])
                    normalized_trial["interventions"] = _dedupe_texts(normalized_trial.get("interventions") or [])
                    grouped_payloads.setdefault(nct_id, []).append(normalized_trial)
                    department_index.setdefault(department_tag, set()).add(nct_id)

        documents: list[TrialDocument] = []
        for nct_id in sorted(grouped_payloads):
            payloads = grouped_payloads[nct_id]
            selected_payload = _select_trial_payload(payloads, preferred_department=preferred_department)
            merged_payload = _merge_trial_payloads(selected_payload, payloads)
            documents.append(_build_trial_document(merged_payload))

        catalog = cls(
            documents=documents,
            department_index=department_index,
            preferred_department=preferred_department,
            department_root=resolved_root,
        )
        with _CACHE_LOCK:
            _TRIAL_CATALOG_CACHE.setdefault(cache_key, catalog)
        return catalog


def _select_trial_payload(payloads: list[dict[str, Any]], *, preferred_department: str) -> dict[str, Any]:
    normalized_preferred_department = str(preferred_department or "").strip()
    if normalized_preferred_department:
        for payload in payloads:
            if str(payload.get("department_tag") or "").strip() == normalized_preferred_department:
                return dict(payload)

    for payload in payloads:
        if str(payload.get("department_role") or "").strip().lower() == "primary":
            return dict(payload)

    return dict(payloads[0]) if payloads else {}


def _merge_trial_payloads(selected_payload: dict[str, Any], payloads: list[dict[str, Any]]) -> dict[str, Any]:
    merged = dict(selected_payload or {})
    merged["department_tags"] = _dedupe_texts(
        item
        for payload in payloads
        for item in [payload.get("department_tag"), *list(payload.get("department_tags") or [])]
    )
    merged["linked_trials"] = _dedupe_texts(
        item
        for payload in payloads
        for item in list(payload.get("linked_trials") or [])
    )
    for list_field in ("conditions", "mesh_terms", "keywords", "interventions", "actions", "secondary_departments"):
        merged[list_field] = _dedupe_texts(
            item
            for payload in payloads
            for item in list(payload.get(list_field) or [])
        )
    return merged


def _build_trial_document(payload: Mapping[str, Any]) -> TrialDocument:
    name = _normalize_whitespace(payload.get("name") or payload.get("brief_title") or payload.get("official_title"))
    summary = _normalize_whitespace(payload.get("brief_summary"))
    purpose = _normalize_whitespace(payload.get("primary_purpose"))
    study_type = _normalize_whitespace(payload.get("study_type"))
    phase = _normalize_whitespace(payload.get("phase"))
    purpose_text = "; ".join(item for item in (purpose, study_type, phase) if item) or summary
    eligibility = _normalize_whitespace(payload.get("eligibility_text"))

    return TrialDocument(
        nct_id=_normalize_whitespace(payload.get("nct_id")),
        title=name or _normalize_whitespace(payload.get("official_title")) or _normalize_whitespace(payload.get("nct_id")),
        summary=summary,
        purpose=purpose_text,
        eligibility=eligibility,
        abstract=summary,
        retrieval_text=_build_trial_retrieval_text(payload),
        status=_normalize_whitespace(payload.get("status")) or "manual_review",
        overall_status=_normalize_whitespace(payload.get("overall_status")),
        enrollment_open=bool(payload.get("enrollment_open")),
        department_tag=_normalize_whitespace(payload.get("department_tag")),
        department_role=_normalize_whitespace(payload.get("department_role")),
        department_tags=_dedupe_texts(payload.get("department_tags") or []),
        conditions=_dedupe_texts(payload.get("conditions") or []),
        mesh_terms=_dedupe_texts(payload.get("mesh_terms") or []),
        keywords=_dedupe_texts(payload.get("keywords") or []),
        interventions=_dedupe_texts(payload.get("interventions") or []),
        phase=_normalize_whitespace(payload.get("phase")),
        primary_purpose=_normalize_whitespace(payload.get("primary_purpose")),
        status_reason=_normalize_whitespace(payload.get("status_reason")),
        actions=_dedupe_texts(payload.get("actions") or []),
        source_payload=dict(payload or {}),
    )


class TrialKeywordRetriever:
    FIELD_WEIGHTS: dict[str, float] = {
        "title": 3.0,
        "retrieval_text": 3.8,
        "purpose": 0.8,
        "eligibility": 0.5,
        "summary": 1.4,
    }

    def __init__(self, catalog: TrialCatalog) -> None:
        self.catalog = catalog
        self._search_rows: list[dict[str, Any]] = []
        bm25_documents: list[dict[str, str]] = []
        for document in catalog.documents():
            self._search_rows.append(
                {
                    "document": document,
                    "title": document.title,
                    "retrieval_text": document.retrieval_text,
                    "purpose": document.purpose,
                    "eligibility": document.eligibility,
                    "summary": document.summary,
                }
            )
            bm25_documents.append(
                {
                    "title": document.title,
                    "retrieval_text": document.retrieval_text,
                    "purpose": document.purpose,
                    "eligibility": document.eligibility,
                    "summary": document.summary,
                }
            )
        self._bm25 = FieldedBM25Index(bm25_documents, field_weights=self.FIELD_WEIGHTS)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        *,
        candidate_ids: set[str] | list[str] | tuple[str, ...] | None = None,
        candidate_pmids: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        normalized_candidate_ids = {
            str(item).strip()
            for item in list(candidate_ids if candidate_ids is not None else candidate_pmids or [])
            if str(item).strip()
        }
        filtered_rows = (
            [
                row
                for row in self._search_rows
                if str(row["document"].nct_id) in normalized_candidate_ids
            ]
            if normalized_candidate_ids
            else list(self._search_rows)
        )
        if not str(query or "").strip():
            return []

        score_by_id = {
            str(self._search_rows[index]["document"].nct_id): float(score)
            for index, score in enumerate(self._bm25.score(query))
            if index < len(self._search_rows)
        }

        scored_rows: list[tuple[float, TrialDocument]] = []
        for row in filtered_rows:
            score = float(score_by_id.get(str(row["document"].nct_id)) or 0.0)
            if score <= 0.0:
                continue
            scored_rows.append((score, row["document"]))

        scored_rows.sort(key=lambda item: (-item[0], item[1].title.lower(), item[1].nct_id))
        return [self._serialize(document, score) for score, document in scored_rows[: max(int(top_k), 1)]]

    @staticmethod
    def _serialize(document: TrialDocument, score: float) -> dict[str, Any]:
        payload = document.to_brief()
        payload["score"] = float(score)
        return payload


class TrialRetrievalTool:
    def __init__(
        self,
        catalog: TrialCatalog,
        *,
        vector_retriever: Any | None = None,
        backend: str = "hybrid",
    ) -> None:
        self.catalog = catalog
        self.vector_retriever = vector_retriever
        self.keyword_retriever = TrialKeywordRetriever(catalog)
        self._retriever = create_structured_retriever(
            catalog,
            bm25_retriever=self.keyword_retriever,
            vector_retriever=vector_retriever,
            query_builder=build_structured_query_text,
            default_backend=backend,
            id_field="nct_id",
        )
        self.default_backend = self._retriever.resolve_backend(backend)

    @property
    def available_backends(self) -> tuple[str, ...]:
        return self._retriever.available_backends

    def resolve_backend(self, backend: str | None = None) -> str:
        return self._retriever.resolve_backend(backend or self.default_backend)

    @tool(
        name="trial_coarse_retriever",
        description=(
            "Coarsely recall local trial candidates from structured_case. "
            "Return a minimal NCT/title bundle for protocol-stage candidate pooling."
        ),
        input_schema={
            "structured_case": {
                "raw_text": "str",
                "case_summary": "str",
                "problem_list": "list[str]",
                "known_facts": "list[str]",
                "department_tags": "list[str]",
            },
            "top_k": "int",
            "department_tags": "list[str] | None",
            "candidate_ids": "list[str] | set[str] | None",
            "backend": "str | None",
        },
        state_fields={
            "structured_case": (
                "structured_case",
                "structured_case_json",
                "clinical_tool_job.structured_case",
            ),
        },
    )
    def retrieve_coarse(
        self,
        structured_case: Mapping[str, Any],
        *,
        top_k: int = 30,
        department_tags: list[str] | None = None,
        candidate_ids: list[str] | set[str] | tuple[str, ...] | None = None,
        backend: str | None = None,
    ) -> dict[str, Any]:
        return self.retrieve_coarse_from_structured_case(
            structured_case,
            top_k=top_k,
            department_tags=department_tags,
            candidate_ids=candidate_ids,
            backend=backend,
        )

    @tool(
        name="trial_candidate_retriever",
        description=(
            "Run the protocol-stage two-stage trial retrieval over local treatment-trial payloads. "
            "Return coarse candidate ids, bm25 top5, vector top5, and the merged top10 candidate ranking."
        ),
        input_schema={
            "structured_case": {
                "raw_text": "str",
                "case_summary": "str",
                "problem_list": "list[str]",
                "known_facts": "list[str]",
                "department_tags": "list[str]",
            },
            "top_k": "int",
            "coarse_top_k": "int",
            "department_tags": "list[str] | None",
            "candidate_ids": "list[str] | set[str] | None",
            "backend": "str | None",
        },
        state_fields={
            "structured_case": (
                "structured_case",
                "structured_case_json",
                "clinical_tool_job.structured_case",
            ),
        },
    )
    def retrieve_candidates(
        self,
        structured_case: Mapping[str, Any],
        *,
        top_k: int = 10,
        coarse_top_k: int = 30,
        department_tags: list[str] | None = None,
        candidate_ids: list[str] | set[str] | tuple[str, ...] | None = None,
        backend: str | None = None,
    ) -> dict[str, Any]:
        return self.retrieve_from_structured_case(
            structured_case,
            top_k=top_k,
            coarse_top_k=coarse_top_k,
            department_tags=department_tags,
            candidate_ids=candidate_ids,
            backend=backend,
        )

    def retrieve_coarse_from_structured_case(
        self,
        structured_case: Mapping[str, Any],
        *,
        top_k: int = 30,
        department_tags: list[str] | None = None,
        candidate_ids: list[str] | set[str] | tuple[str, ...] | None = None,
        backend: str | None = None,
    ) -> dict[str, Any]:
        case_payload = dict(structured_case or {})
        normalized_department_tags = self._normalize_department_tags(
            department_tags if department_tags is not None else case_payload.get("department_tags")
        )
        query_text = self._build_query_text(case_payload)
        resolved_backend = self.resolve_backend(backend)
        department_candidate_ids = self._resolve_department_candidate_ids(normalized_department_tags)
        scoped_candidate_ids = self._resolve_candidate_ids(
            candidate_ids=candidate_ids,
            department_tags=normalized_department_tags,
        )

        if not query_text:
            return {
                "query_text": "",
                "backend_used": resolved_backend,
                "available_backends": list(self.available_backends),
                "department_tags": normalized_department_tags,
                "department_candidate_ids": sorted(department_candidate_ids) if department_candidate_ids else None,
                "fallback_to_full_catalog": False,
                "candidate_ranking": [],
                "coarse_candidate_ids": [],
            }

        payload = self._retriever.retrieve_from_query(
            query_text,
            top_k=max(int(top_k), 1),
            candidate_ids=scoped_candidate_ids,
            backend=resolved_backend,
            include_scores=True,
        )
        fallback_triggered = False
        if department_candidate_ids is not None and not list(payload.get("candidate_ranking") or []):
            fallback_triggered = True
            fallback_candidate_ids = (
                {
                    str(item).strip()
                    for item in list(candidate_ids or [])
                    if str(item).strip()
                }
                if candidate_ids is not None
                else None
            )
            payload = self._retriever.retrieve_from_query(
                query_text,
                top_k=max(int(top_k), 1),
                candidate_ids=fallback_candidate_ids,
                backend=resolved_backend,
                include_scores=True,
            )

        coarse_ids = [
            str(row.get("nct_id") or row.get("document_id") or "").strip()
            for row in list(payload.get("candidate_ranking") or [])
            if str(row.get("nct_id") or row.get("document_id") or "").strip()
        ]
        return {
            "query_text": query_text,
            "backend_used": str(payload.get("backend_used") or resolved_backend),
            "available_backends": list(payload.get("available_backends") or self.available_backends),
            "department_tags": normalized_department_tags,
            "department_candidate_ids": sorted(department_candidate_ids) if department_candidate_ids is not None else None,
            "fallback_to_full_catalog": fallback_triggered,
            "candidate_ranking": [
                {
                    "nct_id": document_id,
                    "title": self._candidate_title(document_id),
                }
                for document_id in coarse_ids
            ],
            "coarse_candidate_ids": coarse_ids,
        }

    def retrieve_from_structured_case(
        self,
        structured_case: Mapping[str, Any],
        *,
        top_k: int = 10,
        coarse_top_k: int = 30,
        department_tags: list[str] | None = None,
        candidate_ids: list[str] | set[str] | tuple[str, ...] | None = None,
        backend: str | None = None,
    ) -> dict[str, Any]:
        case_payload = dict(structured_case or {})
        normalized_department_tags = self._normalize_department_tags(
            department_tags if department_tags is not None else case_payload.get("department_tags")
        )
        query_text = self._build_query_text(case_payload)
        resolved_backend = self.resolve_backend(backend)
        coarse_bundle = self.retrieve_coarse_from_structured_case(
            case_payload,
            top_k=coarse_top_k,
            department_tags=normalized_department_tags,
            candidate_ids=candidate_ids,
            backend=backend,
        )

        coarse_candidate_ids = [
            str(item).strip()
            for item in list(coarse_bundle.get("coarse_candidate_ids") or [])
            if str(item).strip()
        ]
        if not query_text or not coarse_candidate_ids:
            return {
                "query_text": query_text,
                "backend_used": "bm25" if resolved_backend == "bm25" or self.vector_retriever is None else "hybrid",
                "available_backends": list(self.available_backends),
                "department_tags": normalized_department_tags,
                "fallback_to_full_catalog": bool(coarse_bundle.get("fallback_to_full_catalog")),
                "coarse_candidate_ids": coarse_candidate_ids,
                "bm25_top5": [],
                "vector_top5": [],
                "candidate_ranking": [],
            }

        bm25_bundle = self._retriever.retrieve_from_query(
            query_text,
            top_k=min(max(int(top_k), 1), 5),
            candidate_ids=set(coarse_candidate_ids),
            backend="bm25",
            include_scores=True,
        )
        bm25_rows = [dict(row) for row in list(bm25_bundle.get("candidate_ranking") or [])]

        vector_rows: list[dict[str, Any]] = []
        if self.vector_retriever is not None and resolved_backend != "bm25":
            vector_bundle = self._retriever.retrieve_from_query(
                query_text,
                top_k=min(max(int(top_k), 1), 5),
                candidate_ids=set(coarse_candidate_ids),
                backend="vector",
                include_scores=True,
            )
            vector_rows = [dict(row) for row in list(vector_bundle.get("candidate_ranking") or [])]

        bm25_top5 = self._hydrate_ranked_rows(bm25_rows, channel="bm25")
        vector_top5 = self._hydrate_ranked_rows(vector_rows, channel="vector")
        candidate_ranking = self._merge_ranked_rows(
            bm25_rows=bm25_rows,
            vector_rows=vector_rows,
            limit=max(int(top_k), 1),
        )

        return {
            "query_text": query_text,
            "backend_used": "bm25" if not vector_rows else "hybrid",
            "available_backends": list(self.available_backends),
            "department_tags": normalized_department_tags,
            "fallback_to_full_catalog": bool(coarse_bundle.get("fallback_to_full_catalog")),
            "coarse_candidate_ids": coarse_candidate_ids,
            "bm25_top5": bm25_top5,
            "vector_top5": vector_top5,
            "candidate_ranking": candidate_ranking,
        }

    def _build_query_text(self, structured_case: Mapping[str, Any]) -> str:
        return build_structured_query_text(
            raw_text=structured_case.get("raw_text") or structured_case.get("raw_request") or "",
            case_summary=structured_case.get("case_summary"),
            problem_list=structured_case.get("problem_list"),
            known_facts=structured_case.get("known_facts"),
        )

    def _normalize_department_tags(self, department_tags: Any) -> list[str]:
        return _dedupe_texts(department_tags or [])

    def _resolve_department_candidate_ids(self, department_tags: list[str]) -> set[str] | None:
        if not department_tags:
            return None
        candidate_ids: set[str] = set()
        found_department = False
        for department_tag in department_tags:
            department_ids = self.catalog.department_index.get(str(department_tag))
            if department_ids is None:
                continue
            found_department = True
            candidate_ids.update(department_ids)
        if not found_department:
            return None
        return candidate_ids

    def _resolve_candidate_ids(
        self,
        *,
        candidate_ids: list[str] | set[str] | tuple[str, ...] | None,
        department_tags: list[str],
    ) -> set[str] | None:
        normalized_candidate_ids = (
            {
                str(item).strip()
                for item in list(candidate_ids or [])
                if str(item).strip()
            }
            if candidate_ids is not None
            else None
        )
        department_candidate_ids = self._resolve_department_candidate_ids(department_tags)
        if department_candidate_ids is None:
            return normalized_candidate_ids
        if normalized_candidate_ids is None:
            return set(department_candidate_ids)
        return normalized_candidate_ids.intersection(department_candidate_ids)

    def _candidate_title(self, nct_id: str) -> str:
        document = self.catalog.get(nct_id)
        if document is None:
            return str(nct_id)
        return document.title

    def _hydrate_ranked_rows(self, rows: list[dict[str, Any]], *, channel: str) -> list[dict[str, Any]]:
        hydrated: list[dict[str, Any]] = []
        for row in list(rows or []):
            document_id = str(row.get("nct_id") or row.get("document_id") or "").strip()
            if not document_id:
                continue
            document = self.catalog.get(document_id)
            if document is None:
                continue
            candidate = self._candidate_payload(document)
            candidate["channel"] = channel
            candidate["score"] = float(row.get("score") or 0.0)
            candidate["match_sources"] = sorted(set(list(row.get("match_sources") or [])))
            hydrated.append(candidate)
        return hydrated

    def _merge_ranked_rows(
        self,
        *,
        bm25_rows: list[dict[str, Any]],
        vector_rows: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        normalized_bm25_scores = _normalize_scores(bm25_rows)
        normalized_vector_scores = _normalize_scores(vector_rows)
        merged: dict[str, dict[str, Any]] = {}

        for row in list(bm25_rows or []):
            document_id = str(row.get("nct_id") or row.get("document_id") or "").strip()
            if not document_id:
                continue
            merged.setdefault(document_id, {})
            merged[document_id]["bm25_row"] = dict(row)

        for row in list(vector_rows or []):
            document_id = str(row.get("nct_id") or row.get("document_id") or "").strip()
            if not document_id:
                continue
            merged.setdefault(document_id, {})
            merged[document_id]["vector_row"] = dict(row)

        ranked: list[dict[str, Any]] = []
        for document_id, raw_bundle in merged.items():
            document = self.catalog.get(document_id)
            if document is None:
                continue
            bm25_row = dict(raw_bundle.get("bm25_row") or {})
            vector_row = dict(raw_bundle.get("vector_row") or {})
            candidate = self._candidate_payload(document)
            status_priority = _status_priority(str(candidate.get("status") or ""), bool(candidate.get("enrollment_open")))
            candidate["bm25_score"] = float(bm25_row.get("score") or 0.0)
            candidate["vector_score"] = float(vector_row.get("score") or 0.0)
            candidate["score"] = float(normalized_bm25_scores.get(document_id) or 0.0) + float(
                normalized_vector_scores.get(document_id) or 0.0
            )
            candidate["match_sources"] = sorted(
                {
                    *list(bm25_row.get("match_sources") or []),
                    *list(vector_row.get("match_sources") or []),
                }
            )
            ranked.append(candidate)

        ranked.sort(
            key=lambda row: (
                -_status_priority(str(row.get("status") or ""), bool(row.get("enrollment_open"))),
                -float(row.get("score") or 0.0),
                -len(list(row.get("match_sources") or [])),
                str(row.get("title") or "").lower(),
                str(row.get("nct_id") or ""),
            )
        )
        return ranked[: max(int(limit), 1)]

    @staticmethod
    def _candidate_payload(document: TrialDocument) -> dict[str, Any]:
        payload = dict(document.source_payload or {})
        payload["nct_id"] = document.nct_id
        payload["title"] = document.title
        payload["name"] = _normalize_whitespace(payload.get("name") or document.title)
        payload["brief_summary"] = _normalize_whitespace(payload.get("brief_summary") or document.summary)
        payload["status"] = document.status
        payload["overall_status"] = document.overall_status
        payload["enrollment_open"] = document.enrollment_open
        payload["department_tag"] = document.department_tag
        payload["department_tags"] = list(document.department_tags)
        payload["conditions"] = list(document.conditions)
        payload["mesh_terms"] = list(document.mesh_terms)
        payload["keywords"] = list(document.keywords)
        payload["interventions"] = list(document.interventions)
        payload["phase"] = document.phase
        payload["primary_purpose"] = document.primary_purpose
        payload["status_reason"] = document.status_reason
        payload["actions"] = list(document.actions)
        return payload


def create_trial_retrieval_tool(
    *,
    department_root: str | Path | None = None,
    backend: str = "hybrid",
    preferred_department: str = "",
    vector_retriever: Any | None = None,
) -> TrialRetrievalTool:
    resolved_root = _resolve_trial_department_root(department_root)
    normalized_backend = _normalize_backend_name(backend)
    cache_key = (
        str(resolved_root),
        str(preferred_department or "").strip(),
        normalized_backend,
    )
    if vector_retriever is None:
        with _CACHE_LOCK:
            cached = _TRIAL_RETRIEVER_CACHE.get(cache_key)
        if cached is not None:
            return cached

    catalog = TrialCatalog.from_department_root(
        resolved_root,
        preferred_department=preferred_department,
    )

    resolved_vector_retriever = vector_retriever
    if resolved_vector_retriever is None and normalized_backend in {"vector", "hybrid"}:
        try:
            resolved_vector_retriever = MedCPTRetriever(catalog)
        except Exception:
            resolved_vector_retriever = None

    tool_instance = TrialRetrievalTool(
        catalog,
        vector_retriever=resolved_vector_retriever,
        backend="bm25" if resolved_vector_retriever is None else normalized_backend,
    )
    if vector_retriever is None:
        with _CACHE_LOCK:
            _TRIAL_RETRIEVER_CACHE.setdefault(cache_key, tool_instance)
    return tool_instance


__all__ = [
    "TrialCatalog",
    "TrialDocument",
    "TrialKeywordRetriever",
    "TrialRetrievalTool",
    "create_trial_retrieval_tool",
]
