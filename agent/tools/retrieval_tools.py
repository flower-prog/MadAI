from __future__ import annotations

import ast
import functools
import hashlib
import json
import math
import re
import threading
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent.corpus_paths import discover_complete_corpus_pair
from agent.graph.types import RetrievalQuery
from agent.retrieval import FieldedBM25Index, HybridRetriever, MedCPTRetriever, tokenize_bm25_text
from agent.retrieval.vector import _retrieve_dense_scored_pmids

from .execution_tools import (
    ChatClient,
    extract_python_code_blocks,
    maybe_load_json,
    tool,
)
from .structured_retrieval_tools import StructuredRetrievalTool


# Shared runtime caches.
_CATALOG_CACHE: dict[tuple[str, str, str, str], "RiskCalcCatalog"] = {}
_RETRIEVER_CACHE: dict[tuple[str, str], Any] = {}
_CACHE_LOCK = threading.RLock()
_CACHE_KEY_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_RISKCALCS_PAYLOAD_CACHE: dict[str, dict[str, dict[str, Any]]] = {}
_MEDCPT_RESOURCE_CACHE: dict[tuple[str, str, str], tuple[Any, Any, Any]] = {}
_PARAMETER_RETRIEVER_CACHE: dict[
    tuple[str, tuple[tuple[str, float], ...]],
    "RiskCalcParameterRetrievalTool",
] = {}
_INLINE_MEDCPT_RETRIEVER_CACHE: dict[str, Any] = {}
_DEPARTMENT_PMID_CACHE: dict[str, set[str]] = {}
_RETRIEVAL_TOOL_CACHE: dict[
    tuple[str, str, tuple[str, tuple[tuple[str, float], ...]], str],
    "RiskCalcRetrievalTool",
] = {}
_FUSION_RRF_K = 30
_DEFAULT_COARSE_RETRIEVAL_TOP_K = 30


def clear_runtime_caches() -> None:
    with _CACHE_LOCK:
        _CATALOG_CACHE.clear()
        _RETRIEVER_CACHE.clear()
        _CACHE_KEY_LOCKS.clear()
        _RISKCALCS_PAYLOAD_CACHE.clear()
        _MEDCPT_RESOURCE_CACHE.clear()
        _PARAMETER_RETRIEVER_CACHE.clear()
        _INLINE_MEDCPT_RETRIEVER_CACHE.clear()
        _DEPARTMENT_PMID_CACHE.clear()
        _RETRIEVAL_TOOL_CACHE.clear()


def _resolve_hybrid_branch_top_k(top_k: int) -> int:
    requested_top_k = max(int(top_k), 1)
    return max(1, math.ceil(requested_top_k / 2.0))


# Retrieval text and document helpers.
_CALCULATOR_FAMILY_LABELS: dict[str, str] = {
    "severity_score": "severity stratification",
    "diagnostic_rule": "diagnostic rule",
    "treatment_selection": "treatment selection",
    "prognostic_model": "prognosis estimation",
    "complication_risk": "complication risk",
    "risk_score": "risk prediction",
}

_PROTOCOL_FOCUS_LABELS: dict[str, str] = {
    "admission_triage": "admission triage",
    "screening_diagnosis": "screening and diagnosis",
    "treatment_selection": "treatment selection",
    "perioperative_management": "perioperative management",
    "monitoring_followup": "monitoring and follow-up",
    "prognosis_counseling": "prognosis counseling",
}

_PARAMETER_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*•]|\d+\.)\s*(.+?)\s*$")
_PARAMETER_INTRO_PATTERNS = [
    re.compile(
        r"(?:variables?|factors?|criteria|predictors?|characteristics?|parameters?|measurements?|inputs?)"
        r"\s*(?:are|include|included|consist of|consists of)?\s*:\s*(.+?)"
        r"(?:Each\b|The total\b|The score\b|The risk score\b|\n\n|```|$)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"uses?\s+(?:the\s+)?(?:presence of\s+)?(?:\w+\s+)?"
        r"(?:variables?|factors?|criteria|predictors?|characteristics?|parameters?|measurements?|inputs?)"
        r"\s*:\s*(.+?)(?:Each\b|The total\b|The score\b|The risk score\b|\.|```|$)",
        re.IGNORECASE | re.DOTALL,
    ),
]
_FORMULA_PARAMETER_TERM_RE = re.compile(
    r"(?:^|[+\-])\s*\(?\s*-?\d+(?:\.\d+)?\s*(?:[x*×])\s*([^+\-\n)]+)",
    re.IGNORECASE,
)
_FORMULA_FUNCTION_ARG_RE = re.compile(r"\b(?:log|ln|sqrt)\s*\(\s*([^,)]+)", re.IGNORECASE)
_PARAMETER_PHRASE_CANONICAL_MAP: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bage\b", re.IGNORECASE), "age"),
    (re.compile(r"\b(?:male|female|sex|gender)\b", re.IGNORECASE), "sex"),
    (re.compile(r"\bnon[-\s]?white ethnicity\b", re.IGNORECASE), "non_white_ethnicity"),
    (re.compile(r"\bethnicity\b", re.IGNORECASE), "ethnicity"),
    (re.compile(r"\boxygen saturation(?:s)?\b|\bspo2\b", re.IGNORECASE), "oxygen_saturation"),
    (re.compile(r"\bradiological severity score\b", re.IGNORECASE), "radiological_severity_score"),
    (re.compile(r"\bneutrophil count\b", re.IGNORECASE), "neutrophil_count"),
    (re.compile(r"\b(?:c[- ]reactive protein|crp)\b", re.IGNORECASE), "crp"),
    (re.compile(r"\balbumin\b", re.IGNORECASE), "albumin"),
    (re.compile(r"\bcreatinine\b", re.IGNORECASE), "creatinine"),
    (re.compile(r"\bdiabetes(?: mellitus)?\b", re.IGNORECASE), "diabetes"),
    (re.compile(r"\bhypertension\b|\bhtn\b", re.IGNORECASE), "hypertension"),
    (re.compile(r"\bchronic lung disease\b", re.IGNORECASE), "chronic_lung_disease"),
    (re.compile(r"\bferritin(?: level)?\b", re.IGNORECASE), "ferritin"),
    (re.compile(r"\blactate dehydrogenase(?: level)?\b|\bldh\b", re.IGNORECASE), "ldh"),
    (re.compile(r"\brapidly progressive ild\b|\brpild\b", re.IGNORECASE), "rapidly_progressive_ild"),
    (re.compile(r"\bserum sodium\b", re.IGNORECASE), "serum_sodium"),
    (re.compile(r"\bperformance status\b", re.IGNORECASE), "performance_status"),
    (re.compile(r"\bphiladelphia chromosome\b|\bph[-\s]?positive\b", re.IGNORECASE), "ph_positive"),
    (re.compile(r"\bfrench[-\s]?american[-\s]?british\b.*\bmorphology\b", re.IGNORECASE), "morphology"),
    (re.compile(r"\bcourses?\s+to\s+(?:achieve\s+)?(?:complete response|cr)\b", re.IGNORECASE), "courses_to_cr"),
    (re.compile(r"\b(?:day\s*14\s+)?bone marrow blasts?\b|\bmarrow blasts?\b", re.IGNORECASE), "marrow_blasts"),
    (re.compile(r"\bcardiothoracic ratio\b", re.IGNORECASE), "cardiothoracic_ratio"),
    (re.compile(r"\bsdnn\b", re.IGNORECASE), "sdnn"),
    (re.compile(r"\bmaximum corrected qt interval\b|\bqtc\b", re.IGNORECASE), "corrected_qt_interval"),
    (re.compile(r"\bqrs dispersion\b", re.IGNORECASE), "qrs_dispersion"),
    (
        re.compile(r"\bnon[-\s]?sustained ventricular tachycardia\b", re.IGNORECASE),
        "non_sustained_ventricular_tachycardia",
    ),
    (re.compile(r"\bleft ventricular hypertrophy\b", re.IGNORECASE), "left_ventricular_hypertrophy"),
)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _first_sentence(text: str, *, max_chars: int = 220) -> str:
    normalized = _normalize_whitespace(text)
    if not normalized:
        return ""
    sentence = re.split(r"(?<=[.!?])\s+", normalized, maxsplit=1)[0].strip()
    if len(sentence) <= max_chars:
        return sentence
    return sentence[: max_chars - 3].rstrip(" ,;:") + "..."


def _normalize_parameter_name(name: str) -> str:
    normalized = re.sub(r"[_\-\s]+", "_", str(name or "")).strip("_").lower()
    return normalized


def extract_parameter_names_from_computation(computation: str, *, example: str = "") -> list[str]:
    code_parameter_names = _extract_code_parameter_names_from_computation(computation)
    prose_parameter_names = _extract_prose_parameter_names_from_text(
        computation=computation,
        example=example,
    )
    return _merge_parameter_name_sources(code_parameter_names, prose_parameter_names)


def _extract_code_parameter_names_from_computation(computation: str) -> list[str]:
    code = extract_python_code_blocks(str(computation or ""))
    if not code.strip():
        return []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    function_nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if not function_nodes:
        return []

    primary = function_nodes[0]
    deduped: list[str] = []
    seen: set[str] = set()
    for arg in primary.args.args:
        normalized = _normalize_parameter_name(arg.arg)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _split_inline_parameter_list(text: str) -> list[str]:
    working = _normalize_whitespace(text).replace("μ", "u").replace("µ", "u")
    if not working:
        return []
    working = re.sub(r"\band/or\b", ",", working, flags=re.IGNORECASE)
    working = re.sub(r"\s+\band\b\s+", ", ", working, flags=re.IGNORECASE)
    parts = re.split(r",(?![^()]*\))", working)
    return [part.strip(" .;:") for part in parts if part.strip(" .;:")]


def _canonicalize_parameter_candidate(raw_text: str) -> str:
    text = _normalize_whitespace(raw_text).replace("μ", "u").replace("µ", "u")
    if not text:
        return ""

    if ":" in text:
        text = text.split(":", 1)[0].strip()

    text = re.sub(r"^\(?[xyz]?\d+\)?\s*[:=-]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\([^)]*(?:points?|score|yes|no|positive|negative|present|absent)[^)]*\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bassign(?:ed)?\b.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bpoints?\b.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthe patient'?s\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:the|a|an)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*(?:[<>]=?|=|≥|≤).*$", "", text)
    text = text.strip(" -–—:;,.()[]")
    if not text:
        return ""

    lowered = text.lower()
    for pattern, canonical_name in _PARAMETER_PHRASE_CANONICAL_MAP:
        if pattern.search(lowered):
            return canonical_name

    identifier = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    identifier = re.sub(r"_+", "_", identifier)
    return identifier


def _extract_inline_parameter_candidates(text: str) -> list[str]:
    output: list[str] = []
    source = str(text or "")
    for pattern in _PARAMETER_INTRO_PATTERNS:
        for match in pattern.findall(source):
            if re.search(r"\n\s*(?:[-*•]|\d+\.)", match):
                continue
            for piece in _split_inline_parameter_list(match):
                parameter_name = _canonicalize_parameter_candidate(piece)
                if parameter_name:
                    output.append(parameter_name)
    return output


def _extract_list_parameter_candidates(text: str) -> list[str]:
    output: list[str] = []
    for raw_line in str(text or "").splitlines():
        match = _PARAMETER_LIST_ITEM_RE.match(raw_line)
        if not match:
            continue
        content = _normalize_whitespace(match.group(1))
        if not content:
            continue
        parameter_name = _canonicalize_parameter_candidate(content)
        if parameter_name:
            output.append(parameter_name)
    return output


def _extract_formula_parameter_candidates(text: str) -> list[str]:
    output: list[str] = []
    for raw_line in str(text or "").splitlines():
        working = _normalize_whitespace(raw_line).replace("×", "x")
        if "=" not in working:
            continue
        for match in _FORMULA_PARAMETER_TERM_RE.findall(working):
            cleaned = _normalize_whitespace(match)
            if not cleaned:
                continue
            function_arg = _FORMULA_FUNCTION_ARG_RE.search(cleaned)
            if function_arg:
                cleaned = function_arg.group(1)
            for piece in _split_inline_parameter_list(cleaned.replace(" x ", ", ")):
                parameter_name = _canonicalize_parameter_candidate(piece)
                if parameter_name:
                    output.append(parameter_name)
    return output


def _extract_prose_parameter_names_from_text(*, computation: str, example: str = "") -> list[str]:
    candidates: list[str] = []
    for text in (computation, example):
        candidates.extend(_extract_inline_parameter_candidates(text))
        candidates.extend(_extract_list_parameter_candidates(text))
        candidates.extend(_extract_formula_parameter_candidates(text))
    return _sanitize_parameter_names(candidates)


def _is_low_signal_parameter_name(name: str) -> bool:
    lowered = str(name or "").strip().lower()
    if not lowered:
        return True
    if lowered in {"score", "risk", "risk_score", "result", "output", "value", "parameter", "input"}:
        return True
    return bool(
        re.fullmatch(r"[a-z]", lowered)
        or re.fullmatch(r"[a-z]\d+", lowered)
        or re.fullmatch(r"(?:arg|var|value|feature|input|score)_?\d+", lowered)
        or re.fullmatch(r"\d+(?:_\d+)*", lowered)
    )


def _merge_parameter_name_sources(primary: Iterable[Any], supplemental: Iterable[Any]) -> list[str]:
    primary_names = _sanitize_parameter_names(primary)
    supplemental_names = _sanitize_parameter_names(supplemental)
    if not primary_names:
        return supplemental_names
    if not supplemental_names:
        return primary_names

    prefer_supplemental = (
        len(supplemental_names) >= len(primary_names)
        and sum(1 for name in primary_names if _is_low_signal_parameter_name(name)) >= max(1, len(primary_names) // 2)
    )
    ordered_groups = (
        (supplemental_names, primary_names)
        if prefer_supplemental
        else (primary_names, supplemental_names)
    )

    merged: list[str] = []
    seen: set[str] = set()
    for group in ordered_groups:
        for name in group:
            normalized = str(name or "").strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(str(name).strip())
    return merged


def build_riskcalc_retrieval_text(
    *,
    title: str,
    purpose: str,
    specialty: str,
    eligibility: str,
    interpretation: str,
    utility: str,
    abstract: str,
    computation: str,
    taxonomy: dict[str, list[str]] | None = None,
) -> str:
    taxonomy = dict(taxonomy or {})
    lines: list[str] = []

    cleaned_title = _normalize_whitespace(title)
    if cleaned_title:
        lines.append(f"title: {cleaned_title}")

    purpose_sentence = _first_sentence(purpose)
    if purpose_sentence:
        lines.append(f"primary task: {purpose_sentence}")

    eligibility_sentence = _first_sentence(eligibility)
    if eligibility_sentence:
        lines.append(f"eligible population: {eligibility_sentence}")

    specialty_text = _normalize_whitespace(specialty)
    if specialty_text:
        lines.append(f"specialty: {specialty_text}")

    domains = [_normalize_whitespace(item).replace("_", " ") for item in list(taxonomy.get("clinical_domains") or [])]
    if domains:
        lines.append("clinical domain: " + ", ".join(domains))

    patient_segments = [
        _normalize_whitespace(item).replace("_", " ")
        for item in list(taxonomy.get("patient_segments") or [])
    ]
    if patient_segments:
        lines.append("patient segment: " + ", ".join(patient_segments))

    calculator_families = [
        _CALCULATOR_FAMILY_LABELS.get(item, item.replace("_", " "))
        for item in list(taxonomy.get("calculator_families") or [])
    ]
    if calculator_families:
        lines.append("calculator family: " + ", ".join(calculator_families))

    protocol_focus_terms = [
        _PROTOCOL_FOCUS_LABELS.get(item, item.replace("_", " "))
        for item in list(taxonomy.get("protocol_buckets") or [])
    ]
    if protocol_focus_terms:
        lines.append("protocol focus: " + ", ".join(protocol_focus_terms))

    parameter_names = extract_parameter_names_from_computation(computation)
    if parameter_names:
        lines.append("inputs: " + ", ".join(parameter_names[:12]))

    interpretation_sentence = _first_sentence(interpretation)
    if interpretation_sentence:
        lines.append(f"decision focus: {interpretation_sentence}")
    elif utility:
        utility_sentence = _first_sentence(utility)
        if utility_sentence:
            lines.append(f"decision focus: {utility_sentence}")

    abstract_sentence = _first_sentence(abstract)
    if abstract_sentence:
        lines.append(f"evidence summary: {abstract_sentence}")

    return "\n".join(line for line in lines if line.strip()).strip()


def build_riskcalc_retrieval_text_from_payload(
    payload: dict[str, Any],
    *,
    abstract: str = "",
    taxonomy: dict[str, list[str]] | None = None,
) -> str:
    return build_riskcalc_retrieval_text(
        title=str(payload.get("title") or ""),
        purpose=str(payload.get("purpose") or ""),
        specialty=str(payload.get("specialty") or ""),
        eligibility=str(payload.get("eligibility") or ""),
        interpretation=str(payload.get("interpretation") or ""),
        utility=str(payload.get("utility") or ""),
        abstract=abstract,
        computation=str(payload.get("computation") or ""),
        taxonomy=taxonomy,
    )


@dataclass(slots=True)
class RiskCalcDocument:
    pmid: str
    title: str
    purpose: str
    specialty: str
    eligibility: str
    computation: str
    interpretation: str
    utility: str
    abstract: str
    taxonomy: dict[str, list[str]] = field(default_factory=dict)
    retrieval_text: str = ""

    def __post_init__(self) -> None:
        if not self.taxonomy:
            self.taxonomy = classify_calculator_document(self)
        if not self.retrieval_text.strip():
            self.retrieval_text = build_riskcalc_retrieval_text(
                title=self.title,
                purpose=self.purpose,
                specialty=self.specialty,
                eligibility=self.eligibility,
                interpretation=self.interpretation,
                utility=self.utility,
                abstract=self.abstract,
                computation=self.computation,
                taxonomy=self.taxonomy,
            )

    def to_brief(self) -> dict[str, Any]:
        return {
            "pmid": self.pmid,
            "title": self.title.strip(),
            "purpose": self.purpose.strip(),
            "eligibility": self.eligibility.strip(),
            "taxonomy": self.taxonomy,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Taxonomy inference for calculator and request text.
_DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "cardiology": ("cardiology", "cardiac", "heart", "atrial", "ventric", "coronary", "myocard", "stemi", "nstemi"),
    "pulmonology": ("pulmonology", "pulmonary", "pneumonia", "copd", "asthma", "respiratory", "lung", "hypoxem"),
    "oncology": ("oncology", "cancer", "tumor", "carcinoma", "lymphoma", "leukemia", "glioma", "metasta"),
    "neurology": ("neurology", "stroke", "brain", "neuro", "seizure", "dementia", "parkinson", "head injury"),
    "gastroenterology_hepatology": (
        "gastroenterology",
        "hepatology",
        "liver",
        "hepatic",
        "cirrhosis",
        "pancrea",
        "colon",
        "rectal",
        "gastric",
        "bowel",
    ),
    "nephrology_urology": ("nephrology", "renal", "kidney", "urology", "urinary", "bladder", "prostate"),
    "infectious_disease": ("infectious disease", "infection", "infectious", "sepsis", "bacter", "viral", "fung", "covid"),
    "hematology": ("hematology", "anemia", "coagul", "thromb", "hemoph", "stem cell", "transplant"),
    "surgery_perioperative": ("surgery", "surgical", "operative", "postoperative", "perioperative", "resection", "arthroplasty"),
    "emergency_critical_care": ("emergency", "triage", "icu", "critical care", "intensive care", "shock", "ventilation"),
    "obstetrics_gynecology": ("pregnan", "obstetric", "gynecology", "maternal", "preeclampsia", "ovarian"),
    "endocrinology_metabolic": ("diabetes", "glucose", "endocrinology", "thyroid", "metabolic", "obesity"),
    "rheumatology_immunology": ("rheumatology", "lupus", "autoimmune", "arthritis", "inflammatory", "vasculitis"),
}

_PATIENT_SEGMENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "pediatric": ("pediatric", "children", "adolescent", "infant", "young girls"),
    "neonatal": ("neonate", "neonatal", "newborn", "birth weight"),
    "older_adult": ("elderly", "older adult", "older adults", "geriatric", "aged 65", "age >= 65", "70 years"),
    "pregnancy": ("pregnan", "maternal", "postpartum", "women who have recently given birth"),
    "critical_care": ("icu", "critical care", "intensive care", "shock", "ventilation", "ecmo", "critically ill"),
    "emergency_care": ("emergency department", "presenting with", "presented with", "urgent", "triage"),
    "surgical_candidate": ("surgery", "surgical", "operative", "postoperative", "resection", "arthroplasty", "esophagectomy"),
    "oncology_patient": ("cancer", "oncology", "tumor", "carcinoma", "lymphoma", "leukemia", "glioma"),
    "transplant_patient": ("transplant", "stem cell transplant", "allo-hct", "auto-sct", "hct", "lvad"),
}

_PROTOCOL_BUCKET_KEYWORDS: dict[str, tuple[str, ...]] = {
    "admission_triage": ("triage", "admission", "icu", "level of care", "disposition", "hospitalized", "hospitalisation"),
    "screening_diagnosis": ("diagnos", "diagnostic", "screen", "detect", "differentiat", "ct", "mri", "imaging", "biopsy"),
    "treatment_selection": ("therapy", "treatment", "management", "protocol", "chemotherapy", "radiotherapy", "immunotherapy", "benefit"),
    "perioperative_management": ("surgery", "surgical", "perioperative", "operative", "postoperative", "resection", "arthroplasty"),
    "monitoring_followup": ("follow-up", "monitor", "readmission", "toxicity", "complication", "flare", "adverse", "failure"),
    "prognosis_counseling": ("prognosis", "survival", "mortality", "outcome", "recurrence", "progression-free"),
}

_CALCULATOR_FAMILY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "severity_score": ("severity", "triage", "icu", "admission", "disposition", "critical"),
    "diagnostic_rule": ("diagnos", "diagnostic", "screen", "detect", "differentiat", "ct", "mri", "imaging", "rule"),
    "treatment_selection": ("therapy", "treatment", "management", "protocol", "benefit", "surgery", "chemotherapy", "radiotherapy"),
    "prognostic_model": ("prognosis", "survival", "mortality", "outcome", "recurrence", "progression-free"),
    "complication_risk": ("complication", "toxicity", "bleeding", "infection", "readmission", "failure", "adverse"),
    "risk_score": ("risk", "probability", "likelihood", "predict"),
}


def _stable_digest(*parts: str) -> str:
    payload = "::".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _taxonomy_text(*parts: str) -> str:
    return " ".join(str(part or "").strip().lower() for part in parts if str(part or "").strip())


@functools.lru_cache(maxsize=512)
def _taxonomy_keyword_pattern(keyword: str) -> re.Pattern[str]:
    normalized_keyword = _taxonomy_text(keyword)
    if not normalized_keyword:
        return re.compile(r"(?!x)x")

    parts = [re.escape(part) for part in re.split(r"[\s/_-]+", normalized_keyword) if part]
    if not parts:
        return re.compile(r"(?!x)x")

    if len(parts) > 1:
        separator_pattern = r"[\s/_-]+"
        return re.compile(rf"\b{separator_pattern.join(parts)}\b")

    # Longer single-token keywords in the taxonomy tables are often stems such as
    # "diagnos" or "pregnan", so we allow suffix expansion after a word boundary.
    token = parts[0]
    if len(normalized_keyword) <= 4:
        return re.compile(rf"\b{token}\b")
    return re.compile(rf"\b{token}[a-z0-9]*\b")


def _match_taxonomy_labels(text: str, mapping: dict[str, tuple[str, ...]]) -> list[str]:
    labels: list[str] = []
    for label, keywords in mapping.items():
        if any(_taxonomy_keyword_pattern(keyword).search(text) for keyword in keywords):
            labels.append(label)
    return labels


def _dedupe_labels(labels: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for label in labels:
        normalized = str(label or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def infer_request_taxonomy(text: str) -> dict[str, list[str]]:
    return _classify_taxonomy_from_text(text=text, specialty="")


def classify_calculator_document(document: RiskCalcDocument) -> dict[str, list[str]]:
    return _classify_taxonomy_from_text(
        text=" ".join(
            [
                document.title,
                document.purpose,
                document.eligibility,
                document.utility,
                document.interpretation,
                document.abstract,
            ]
        ),
        specialty=document.specialty,
    )


def _classify_taxonomy_from_text(*, text: str, specialty: str) -> dict[str, list[str]]:
    specialty_text = _taxonomy_text(specialty)
    full_text = _taxonomy_text(text, specialty)

    clinical_domains = _match_taxonomy_labels(full_text, _DOMAIN_KEYWORDS)
    patient_segments = _match_taxonomy_labels(full_text, _PATIENT_SEGMENT_KEYWORDS)
    protocol_buckets = _match_taxonomy_labels(full_text, _PROTOCOL_BUCKET_KEYWORDS)
    calculator_families = _match_taxonomy_labels(full_text, _CALCULATOR_FAMILY_KEYWORDS)

    if "pediatrics" in specialty_text and "pediatric" not in patient_segments:
        patient_segments.append("pediatric")
    if "neonatology" in specialty_text and "neonatal" not in patient_segments:
        patient_segments.append("neonatal")
    if "geriatrics" in specialty_text and "older_adult" not in patient_segments:
        patient_segments.append("older_adult")

    age_matches = re.findall(r"\b(\d{1,3})\s*[- ]?year[- ]old\b", full_text)
    if age_matches:
        age = int(age_matches[0])
        if age < 1 and "neonatal" not in patient_segments:
            patient_segments.append("neonatal")
        elif age < 18 and "pediatric" not in patient_segments:
            patient_segments.append("pediatric")
        elif age >= 65 and "older_adult" not in patient_segments:
            patient_segments.append("older_adult")

    if "emergency medicine" in specialty_text and "emergency_care" not in patient_segments:
        patient_segments.append("emergency_care")
    if any(term in specialty_text for term in ("critical care", "intensive care", "anesthesiology")):
        patient_segments.append("critical_care")
    if "surgery" in specialty_text:
        patient_segments.append("surgical_candidate")
    if "oncology" in specialty_text:
        patient_segments.append("oncology_patient")

    if not calculator_families:
        calculator_families.append("risk_score")

    if not protocol_buckets:
        if "severity_score" in calculator_families:
            protocol_buckets.append("admission_triage")
        elif "diagnostic_rule" in calculator_families:
            protocol_buckets.append("screening_diagnosis")
        elif "treatment_selection" in calculator_families:
            protocol_buckets.append("treatment_selection")
        elif "complication_risk" in calculator_families:
            protocol_buckets.append("monitoring_followup")
        elif "prognostic_model" in calculator_families:
            protocol_buckets.append("prognosis_counseling")

    return {
        "clinical_domains": _dedupe_labels(clinical_domains),
        "patient_segments": _dedupe_labels(patient_segments),
        "protocol_buckets": _dedupe_labels(protocol_buckets),
        "calculator_families": _dedupe_labels(calculator_families),
    }


# Catalog-backed retrievers.
class RiskCalcCatalog:
    def __init__(
        self,
        documents: dict[str, RiskCalcDocument],
        *,
        cache_key: str | None = None,
        source_files: tuple[Path, Path] | None = None,
        dense_cache_dir: Path | None = None,
    ) -> None:
        self._documents = documents
        self.cache_key = str(cache_key or "").strip() or None
        self.source_files = source_files
        self.dense_cache_dir = dense_cache_dir
        self.runtime_cache_key = self.cache_key or f"inmemory:{id(self)}"

    @classmethod
    def from_paths(cls, riskcalcs_path: str | Path, pmid_metadata_path: str | Path) -> "RiskCalcCatalog":
        riskcalcs_file = Path(riskcalcs_path).resolve()
        pmid_file = Path(pmid_metadata_path).resolve()
        riskcalcs_signature = _file_signature(riskcalcs_file)
        pmid_signature = _file_signature(pmid_file)
        cache_slot = (str(riskcalcs_file), str(pmid_file), riskcalcs_signature, pmid_signature)
        with _CACHE_LOCK:
            cached_catalog = _CATALOG_CACHE.get(cache_slot)
        if cached_catalog is not None:
            return cached_catalog

        riskcalcs = json.loads(riskcalcs_file.read_text(encoding="utf-8"))
        pmid2info = json.loads(pmid_file.read_text(encoding="utf-8"))

        documents: dict[str, RiskCalcDocument] = {}
        for pmid, payload in riskcalcs.items():
            meta = dict(pmid2info.get(pmid) or {})
            documents[str(pmid)] = RiskCalcDocument(
                pmid=str(pmid),
                title=str(payload.get("title") or meta.get("t") or ""),
                purpose=str(payload.get("purpose") or ""),
                specialty=str(payload.get("specialty") or ""),
                eligibility=str(payload.get("eligibility") or ""),
                computation=str(payload.get("computation") or ""),
                interpretation=str(payload.get("interpretation") or ""),
                utility=str(payload.get("utility") or ""),
                abstract=str(meta.get("a") or ""),
                retrieval_text=str(payload.get("retrieval_text") or ""),
            )
        catalog = cls(
            documents,
            cache_key=_stable_digest(riskcalcs_signature, pmid_signature),
            source_files=(riskcalcs_file, pmid_file),
            dense_cache_dir=riskcalcs_file.parent / ".cache" / "medcpt",
        )
        with _CACHE_LOCK:
            _CATALOG_CACHE[cache_slot] = catalog
        return catalog

    @staticmethod
    def discover_default_paths(search_root: str | Path | None = None) -> tuple[Path, Path]:
        root = Path(search_root).expanduser().resolve() if search_root else Path(__file__).resolve().parents[2]
        return discover_complete_corpus_pair(root)

    def __len__(self) -> int:
        return len(self._documents)

    def get(self, pmid: str) -> RiskCalcDocument:
        return self._documents[str(pmid)]

    def documents(self) -> list[RiskCalcDocument]:
        return list(self._documents.values())

    def dense_index_cache_paths(
        self,
        *,
        query_model_name: str,
        doc_model_name: str,
    ) -> tuple[Path, Path] | None:
        if self.dense_cache_dir is None or self.cache_key is None:
            return None
        model_key = _stable_digest(query_model_name, doc_model_name)[:16]
        base_dir = self.dense_cache_dir / self.cache_key
        return (
            base_dir / f"{model_key}.faiss",
            base_dir / f"{model_key}.pmids.json",
        )

    def format_calculator_text(self, pmid: str) -> str:
        doc = self.get(pmid)
        sections = [
            ("TITLE", doc.title),
            ("PURPOSE", doc.purpose),
            ("SPECIALTY", doc.specialty),
            ("ELIGIBILITY", doc.eligibility),
            ("COMPUTATION", doc.computation),
            ("INTERPRETATION", doc.interpretation),
            ("UTILITY", doc.utility),
        ]
        return "\n".join(f"{name}\n{value}".strip() for name, value in sections if value.strip())

    def build_candidate_pool(
        self,
        *,
        clinical_text: str,
        limit: int = 240,
    ) -> set[str] | None:
        request_taxonomy = infer_request_taxonomy(clinical_text)
        patient_labels = list(request_taxonomy["patient_segments"])
        family_labels = list(request_taxonomy["calculator_families"])
        protocol_labels = list(request_taxonomy["protocol_buckets"])
        domain_labels = list(request_taxonomy["clinical_domains"])

        if not any([patient_labels, family_labels, protocol_labels, domain_labels]):
            return None

        strong_matches: list[tuple[float, str]] = []
        fallback_matches: list[tuple[float, str]] = []

        for document in self.documents():
            taxonomy = document.taxonomy
            family_overlap_labels = set(family_labels).intersection(taxonomy.get("calculator_families") or [])
            specific_family_overlap = len([label for label in family_overlap_labels if label != "risk_score"])
            generic_family_overlap = 1 if "risk_score" in family_overlap_labels else 0
            protocol_overlap = len(set(protocol_labels).intersection(taxonomy.get("protocol_buckets") or []))
            patient_overlap = len(set(patient_labels).intersection(taxonomy.get("patient_segments") or []))
            domain_overlap = len(set(domain_labels).intersection(taxonomy.get("clinical_domains") or []))

            score = (
                (protocol_overlap * 5.0)
                + (specific_family_overlap * 4.0)
                + (generic_family_overlap * 1.0)
                + (patient_overlap * 2.0)
                + (domain_overlap * 1.5)
            )
            if score <= 0:
                continue

            row = (score, document.pmid)
            if protocol_overlap > 0 or specific_family_overlap > 0:
                strong_matches.append(row)
            fallback_matches.append(row)

        ranked = strong_matches or fallback_matches
        if not ranked:
            return None

        ranked.sort(key=lambda item: (-item[0], item[1]))
        bounded_limit = max(int(limit), 1)
        return {pmid for _, pmid in ranked[:bounded_limit]}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", (text or "").lower())


class KeywordToolRetriever:
    FIELD_WEIGHTS: dict[str, float] = {
        # retrieval_text is a support field; title + abstract remain the
        # highest-precision signals for disambiguating similarly worded tools.
        "title": 3.8,
        "retrieval_text": 1.2,
        "purpose": 0.8,
        "eligibility": 0.4,
        "abstract": 3.6,
    }

    def __init__(self, catalog: RiskCalcCatalog) -> None:
        self.catalog = catalog
        self._search_rows = []
        bm25_documents: list[dict[str, str]] = []
        for doc in catalog.documents():
            self._search_rows.append(
                {
                    "document": doc,
                    "title": doc.title,
                    "retrieval_text": doc.retrieval_text,
                    "purpose": doc.purpose,
                    "eligibility": doc.eligibility,
                    "abstract": doc.abstract,
                }
            )
            bm25_documents.append(
                {
                    "title": doc.title,
                    "retrieval_text": doc.retrieval_text,
                    "purpose": doc.purpose,
                    "eligibility": doc.eligibility,
                    "abstract": doc.abstract,
                }
            )
        self._bm25 = FieldedBM25Index(
            bm25_documents,
            field_weights=self.FIELD_WEIGHTS,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        *,
        candidate_pmids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        filtered_rows = (
            [row for row in self._search_rows if str(row["document"].pmid) in candidate_pmids]
            if candidate_pmids
            else list(self._search_rows)
        )
        if not str(query or "").strip():
            return [self._serialize(row["document"], 0.0) for row in filtered_rows[:top_k]]

        score_by_pmid = {
            str(self._search_rows[index]["document"].pmid): float(score)
            for index, score in enumerate(self._bm25.score(query))
            if index < len(self._search_rows)
        }
        scored_rows = []
        for row in filtered_rows:
            score = float(score_by_pmid.get(str(row["document"].pmid)) or 0.0)
            if score <= 0:
                continue
            scored_rows.append((score, row["document"]))

        scored_rows.sort(key=lambda item: (-item[0], item[1].title.lower(), item[1].pmid))
        if not scored_rows:
            fallback = [self._serialize(row["document"], 0.0) for row in filtered_rows[:top_k]]
            return fallback
        return [self._serialize(doc, score) for score, doc in scored_rows[:top_k]]

    @staticmethod
    def _serialize(document: RiskCalcDocument, score: float) -> dict[str, Any]:
        payload = document.to_brief()
        payload["score"] = score
        return payload


def create_retriever(catalog: RiskCalcCatalog, backend: str = "hybrid"):
    normalized = str(backend or "hybrid").strip().lower()
    if not normalized:
        normalized = "hybrid"
    normalized = {
        "auto": "hybrid",
        "medcpt": "vector",
    }.get(normalized, normalized)
    cache_key = (normalized, catalog.runtime_cache_key)
    with _CACHE_LOCK:
        cached_retriever = _RETRIEVER_CACHE.get(cache_key)
    if cached_retriever is not None:
        return cached_retriever

    if normalized == "keyword":
        retriever = KeywordToolRetriever(catalog)
    elif normalized == "vector":
        retriever = MedCPTRetriever(catalog)
    elif normalized == "hybrid":
        retriever = HybridRetriever(
            catalog,
            keyword_retriever=KeywordToolRetriever(catalog),
        )
    else:
        raise ValueError(
            f"Unsupported retriever backend: {backend!r}. Expected one of: keyword, vector, hybrid."
        )

    with _CACHE_LOCK:
        cached_retriever = _RETRIEVER_CACHE.setdefault(cache_key, retriever)
    return cached_retriever


# Query planning helpers for note-driven retrieval.
_SINGLE_TOKEN_MEDICAL_ANCHORS = {
    "astrocytoma",
    "avastin",
    "bevacizumab",
    "glioma",
    "hypertension",
    "mri",
    "radiation",
}

_SUMMARY_PRIMARY_DISEASE_KEYWORDS = (
    "astrocytoma",
    "glioma",
    "glioblastoma",
    "meningioma",
    "tumor",
    "tumour",
    "neoplasm",
    "malignan",
    "metasta",
    "spinal",
    "conus",
    "cord",
)

_SUMMARY_TREATMENT_KEYWORDS = (
    "radiation",
    "radiotherapy",
    "chemotherapy",
    "temozolomide",
    "bevacizumab",
    "avastin",
    "surgery",
    "resection",
    "steroid",
)

_SUMMARY_COMPLICATION_KEYWORDS = (
    "compression",
    "paralysis",
    "weakness",
    "retention",
    "numbness",
    "pain",
    "progressive",
)


def is_risk_hint_query(query: RetrievalQuery) -> bool:
    stage = str(query.stage or "").strip().lower()
    intent = str(query.intent or "").strip().lower()
    return (
        intent == "risk_hint_seed"
        or stage.startswith("risk_seed_")
        or stage.startswith("risk_hint_")
        or stage == "risk_hint"
    )


def is_case_summary_query(query: RetrievalQuery) -> bool:
    stage = str(query.stage or "").strip().lower()
    intent = str(query.intent or "").strip().lower()
    return (
        intent == "case_summary_dense"
        or stage == "case_summary_dense"
        or stage.startswith("case_summary_")
    )


def is_problem_anchor_query(query: RetrievalQuery) -> bool:
    stage = str(query.stage or "").strip().lower()
    intent = str(query.intent or "").strip().lower()
    return intent == "problem_anchor" or stage.startswith("problem_anchor_")


def classify_query_channel(query: RetrievalQuery) -> str:
    if is_case_summary_query(query):
        return "case_summary"
    if is_problem_anchor_query(query):
        return "problem_anchor"
    if is_risk_hint_query(query):
        return "risk_hint"

    intent = str(query.intent or "").strip().lower()
    stage = str(query.stage or "").strip().lower()
    if intent == "clinical_question" or stage == "coarse_question":
        return "case_summary"
    if intent == "calculator_classification":
        return "calculator_category"
    if intent == "protocol_matching":
        return "protocol_mapping"
    return "general"


def extract_risk_hints_from_queries(queries: list[RetrievalQuery]) -> list[str]:
    hints = [query.text for query in queries if is_risk_hint_query(query) and str(query.text).strip()]
    return [str(hint).strip() for hint in hints if str(hint).strip()]


def build_risk_hint_queries(
    risk_hints: list[str],
    *,
    max_count: int | None = None,
    rationale: str | None = None,
    start_priority: int = 1,
) -> list[RetrievalQuery]:
    limit = len(risk_hints) if max_count is None else max(int(max_count), 0)
    trimmed = [str(hint).strip() for hint in risk_hints if str(hint).strip()][:limit]
    query_rationale = rationale or (
        "Use concise self-contained risk strings for calculator retrieval, mirroring the AgentMD risk triage step."
    )
    return [
        RetrievalQuery(
            stage=f"risk_hint_{index}",
            text=hint,
            intent="risk_hint_seed",
            rationale=query_rationale,
            priority=(start_priority + index - 1),
        )
        for index, hint in enumerate(trimmed, start=1)
    ]


def _normalize_query_text(text: str) -> str:
    cleaned = re.sub(r"</?topic[^>]*>", " ", str(text or ""), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" -,")


def _token_count(text: str) -> int:
    # Count ASCII word-like tokens plus CJK characters so pure Chinese queries
    # are not incorrectly treated as zero-length.
    cleaned = str(text or "")
    ascii_tokens = re.findall(r"[A-Za-z0-9]+", cleaned)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", cleaned)
    return len(ascii_tokens) + len(cjk_chars)


def _trim_leading_context(text: str) -> str:
    cleaned = _normalize_query_text(text)
    patterns = [
        r"^patient (?:is|was) a [^,;]+ with (?:a history of )?",
        r"^patient initially presented with ",
        r"^mri showed ",
        r"^found to be ",
        r"^the tumor is located in ",
    ]
    for pattern in patterns:
        updated = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip(" -,")
        if updated and updated != cleaned:
            return updated
    return cleaned


def _expand_problem_aliases(text: str) -> list[str]:
    normalized = _trim_leading_context(text)
    lowered = normalized.lower()
    aliases: list[str] = []

    if "astrocytoma" in lowered and "glioma" not in lowered:
        aliases.append(f"{normalized} glioma")
    if any(term in lowered for term in ["spinal cord", "spinal", "conus"]) and "spinal tumor" not in lowered:
        aliases.append(f"{normalized} spinal tumor")
    if "avastin" in lowered and "bevacizumab" not in lowered:
        aliases.append(f"{normalized} bevacizumab")
    if "mri" in lowered and "magnetic resonance imaging" not in lowered:
        aliases.append(f"{normalized} magnetic resonance imaging")

    deduped: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        key = alias.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(alias.strip())
    return deduped


def _extract_problem_anchor_texts(problem_list: list[str], *, max_count: int = 8) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def _append(text: str) -> None:
        normalized = _trim_leading_context(text)
        key = normalized.lower()
        if not key or key in seen:
            return
        token_count = _token_count(normalized)
        if token_count < 2 and key not in _SINGLE_TOKEN_MEDICAL_ANCHORS:
            return
        seen.add(key)
        candidates.append(normalized)

    for problem in problem_list:
        normalized_problem = _normalize_query_text(problem)
        if not normalized_problem:
            continue

        fragments = re.split(r",|;|\band\b|\bcomplicated by\b|\bwhere\b", normalized_problem, flags=re.IGNORECASE)
        for fragment in fragments:
            cleaned_fragment = _normalize_query_text(fragment)
            candidate_fragment = _trim_leading_context(cleaned_fragment)
            candidate_token_count = _token_count(candidate_fragment)
            if (
                2 <= candidate_token_count <= 14
                or candidate_fragment.lower() in _SINGLE_TOKEN_MEDICAL_ANCHORS
            ):
                _append(cleaned_fragment)
                for alias in _expand_problem_aliases(cleaned_fragment):
                    _append(alias)

        if _token_count(normalized_problem) <= 14:
            _append(normalized_problem)
            for alias in _expand_problem_aliases(normalized_problem):
                _append(alias)

        if len(candidates) >= max_count:
            break

    return candidates[:max_count]


def _summary_anchor_score(text: str) -> float:
    lowered = str(text or "").lower()
    token_count = _token_count(lowered)
    score = 0.0

    if any(keyword in lowered for keyword in _SUMMARY_PRIMARY_DISEASE_KEYWORDS):
        score += 3.0
    if any(keyword in lowered for keyword in _SUMMARY_TREATMENT_KEYWORDS):
        score += 2.0
    if any(keyword in lowered for keyword in _SUMMARY_COMPLICATION_KEYWORDS):
        score += 1.5
    if 2 <= token_count <= 10:
        score += 0.5
    elif token_count > 16:
        score -= 0.5

    return score


def _select_problem_anchor_queries(problem_list: list[str], *, max_count: int = 8) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()

    for problem in problem_list:
        normalized_problem = _trim_leading_context(_normalize_query_text(problem))
        key = normalized_problem.lower()
        if not key or key in seen:
            continue

        token_count = _token_count(normalized_problem)
        if token_count < 2 and key not in _SINGLE_TOKEN_MEDICAL_ANCHORS:
            continue

        seen.add(key)
        selected.append(normalized_problem)
        if len(selected) >= max_count:
            break

    return selected


def build_case_summary(
    *,
    problem_list: list[str],
    max_anchor_count: int = 5,
) -> str:
    anchors = _extract_problem_anchor_texts(problem_list, max_count=24)
    if not anchors:
        return ""

    ranked = sorted(
        enumerate(anchors),
        key=lambda item: (-_summary_anchor_score(item[1]), item[0]),
    )

    selected_with_index: list[tuple[int, str]] = []
    for index, anchor in ranked:
        normalized = anchor.strip()
        lowered = normalized.lower()
        if any(lowered in existing.lower() or existing.lower() in lowered for _, existing in selected_with_index):
            continue
        selected_with_index.append((index, normalized))
        if len(selected_with_index) >= max(max_anchor_count, 1):
            break

    selected_anchors = [anchor for _, anchor in sorted(selected_with_index, key=lambda item: item[0])]
    summary_parts: list[str] = []
    if selected_anchors:
        summary_parts.append("primary problem: " + "; ".join(selected_anchors[: max(max_anchor_count, 1)]))

    return ". ".join(part for part in summary_parts if str(part).strip()).strip(" .")


def build_patient_note_queries(
    *,
    case_summary: str | None = None,
    risk_hints: list[str],
    problem_list: list[str],
    risk_count: int,
) -> list[RetrievalQuery]:
    queries: list[RetrievalQuery] = []
    next_priority = 1

    if str(case_summary or "").strip():
        queries.append(
            RetrievalQuery(
                stage="case_summary_dense",
                text=str(case_summary).strip(),
                intent="case_summary_dense",
                rationale=(
                    "Use a compact case summary as the primary semantic retrieval query so vector search "
                    "captures the overall disease and management direction before keyword recall expands coverage."
                ),
                priority=next_priority,
            )
        )
        next_priority += 1

    for index, anchor in enumerate(_select_problem_anchor_queries(problem_list, max_count=16), start=1):
        queries.append(
            RetrievalQuery(
                stage=f"problem_anchor_{index}",
                text=anchor,
                intent="problem_anchor",
                rationale=(
                    "Carry forward one normalized local problem segment per query so staged retrieval "
                    "keeps disease, treatment, imaging, and comorbidity evidence separated."
                ),
                priority=next_priority,
            )
        )
        next_priority += 1

    queries.extend(
        build_risk_hint_queries(
            risk_hints,
            max_count=risk_count,
            start_priority=next_priority,
        )
    )

    return queries


def _fallback_risk_hints(clinical_text: str, risk_count: int) -> list[str]:
    cleaned = " ".join(str(clinical_text or "").split())
    if not cleaned:
        return []
    return [cleaned[:160]]


def generate_risk_hints(
    *,
    clinical_text: str,
    risk_count: int,
    chat_client: ChatClient,
    model: str | None = None,
    temperature: float = 0.0,
) -> tuple[list[str], str]:
    system = (
        "You are a helpful assistant doctor. Your task is to generate a list of risks for the given patient. "
        "Your response is for research purpose only and will not be used in clinical practice."
    )
    prompt = (
        "Here is the patient admission note:\n"
        f"{clinical_text}\n\n"
        f"Please generate a list of {risk_count} potential clinical risks that are significant, urgent, and "
        'specific to the patient. Output a json list where each element is a self-contained short risk string '
        'that contains both the risk event and the underlying condition, e.g. "X due to Y". '
        "Please be concise, and each risk should only contain several words."
    )
    answer = chat_client.complete(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        model=model,
        temperature=temperature,
    )
    payload = maybe_load_json(answer)
    if isinstance(payload, list):
        hints = [str(item).strip() for item in payload if str(item).strip()]
    else:
        hints = []
    if not hints:
        hints = _fallback_risk_hints(clinical_text, risk_count)
    return hints[: max(int(risk_count), 1)], answer


# Parameter retrieval over calculator inputs and aliases.
def _dedupe_texts(values: Iterable[Any]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _clean_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _coerce_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return _dedupe_texts([value])
    if isinstance(value, Mapping):
        items: list[Any] = []
        for mapping_value in value.values():
            if isinstance(mapping_value, (list, tuple, set)):
                items.extend(mapping_value)
            else:
                items.append(mapping_value)
        return _dedupe_texts(items)
    if isinstance(value, Iterable):
        return _dedupe_texts(value)
    return _dedupe_texts([value])


_PARAMETER_ACTION_PREFIXES = {
    "analyze",
    "build",
    "calculate",
    "check",
    "classify",
    "compute",
    "construct",
    "determine",
    "estimate",
    "extract",
    "identify",
    "integrate",
    "predict",
    "transform",
    "use",
}

_PARAMETER_PROCEDURAL_TOKENS = {
    "analysis",
    "based",
    "calculate",
    "calculated",
    "classify",
    "combination",
    "conditions",
    "construct",
    "difference",
    "discharge",
    "following",
    "groups",
    "identified",
    "integrate",
    "model",
    "prediction",
    "presentation",
    "prognosis",
    "risk",
    "safe",
    "score",
    "software",
    "suitable",
    "survival",
    "these",
    "those",
    "through",
    "transform",
    "using",
    "value",
    "values",
}

_PARAMETER_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "could",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "otherwise",
    "that",
    "the",
    "their",
    "these",
    "this",
    "those",
    "through",
    "to",
    "upon",
    "using",
    "was",
    "were",
    "with",
    "without",
}

_GENERIC_PARAMETER_TOKENS = {
    "factor",
    "factors",
    "history",
    "measurement",
    "model",
    "patient",
    "patients",
    "prediction",
    "predictor",
    "predictors",
    "prognosis",
    "result",
    "results",
    "risk",
    "score",
    "status",
    "symptom",
    "symptoms",
    "test",
    "tests",
    "value",
    "values",
}


def _extract_parameter_candidate_tokens(value: Any) -> list[str]:
    normalized = _clean_text(value).lower()
    if not normalized:
        return []
    normalized = normalized.replace("-", "_").replace("/", "_").replace(" ", "_")
    return [token for token in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", normalized) if token]


def _looks_like_noisy_parameter_candidate(value: Any) -> bool:
    cleaned = _clean_text(value)
    if not cleaned:
        return True

    if re.fullmatch(r"\d+(?:[_-]\d+)*", cleaned) or re.match(r"^\d", cleaned):
        return True

    # Keep concise Chinese aliases such as "高血压" or "卒中史".
    if re.search(r"[\u4e00-\u9fff]", cleaned):
        return len(cleaned) > 24

    tokens = _extract_parameter_candidate_tokens(cleaned)
    if not tokens:
        return True

    if len(cleaned) >= 96 or len(tokens) >= 14:
        return True

    lowered_tokens = [token.lower() for token in tokens]
    if any(token in {"otherwise", "respectively"} for token in lowered_tokens):
        return True

    if len(lowered_tokens) == 1 and lowered_tokens[0] in _GENERIC_PARAMETER_TOKENS:
        return True

    if len(lowered_tokens) <= 2 and all(token in _GENERIC_PARAMETER_TOKENS for token in lowered_tokens):
        return True

    procedural_count = sum(token in _PARAMETER_PROCEDURAL_TOKENS for token in lowered_tokens)
    stopword_count = sum(token in _PARAMETER_STOPWORDS for token in lowered_tokens)
    generic_count = sum(token in _GENERIC_PARAMETER_TOKENS for token in lowered_tokens)

    if lowered_tokens[0] in _PARAMETER_ACTION_PREFIXES and len(lowered_tokens) >= 4:
        return True

    if len(lowered_tokens) >= 8 and procedural_count >= 3:
        return True

    if len(lowered_tokens) >= 10 and stopword_count >= 4:
        return True

    if len(lowered_tokens) >= 6 and procedural_count >= 2 and stopword_count >= 2:
        return True

    if len(lowered_tokens) >= 7 and generic_count >= 4 and stopword_count >= 2:
        return True

    return False


def _sanitize_parameter_names(values: Iterable[Any]) -> list[str]:
    return _dedupe_texts(
        value
        for value in values
        if not _looks_like_noisy_parameter_candidate(value)
    )


def _sanitize_parameter_aliases(parameter_name: str, aliases: Iterable[Any]) -> list[str]:
    return _dedupe_texts(
        value
        for value in (parameter_name, *list(aliases))
        if not _looks_like_noisy_parameter_candidate(value)
    )


_BM25_RAW_TEXT_MAX_CHARS = 1200


def build_case_query_text(
    *,
    raw_text: Any = "",
    case_summary: Any = None,
    problem_list: Any = None,
    known_facts: Any = None,
    raw_text_max_chars: int = _BM25_RAW_TEXT_MAX_CHARS,
) -> str:
    parts: list[str] = []

    cleaned_raw_text = _clean_text(raw_text)
    cleaned_case_summary = _clean_text(case_summary)
    resolved_raw_text_max_chars = max(int(raw_text_max_chars), 0)

    narrative_text = cleaned_raw_text
    if (
        cleaned_raw_text
        and resolved_raw_text_max_chars > 0
        and len(cleaned_raw_text) > resolved_raw_text_max_chars
        and cleaned_case_summary
    ):
        narrative_text = cleaned_case_summary
    elif not cleaned_raw_text and cleaned_case_summary:
        narrative_text = cleaned_case_summary

    if narrative_text:
        parts.append(narrative_text)

    if cleaned_case_summary and narrative_text != cleaned_case_summary:
        parts.append("case_summary: " + cleaned_case_summary)

    problem_items = _coerce_text_list(problem_list)
    if problem_items:
        parts.append("problem_list: " + " ; ".join(problem_items))

    known_fact_items = _coerce_text_list(known_facts)
    if known_fact_items:
        parts.append("known_facts: " + " ; ".join(known_fact_items))

    return "\n".join(parts).strip()


def build_parameter_document_payload(calc_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    parameter_names = _merge_parameter_name_sources(
        payload.get("parameter_names") or [],
        extract_parameter_names_from_computation(
            str(payload.get("computation") or ""),
            example=str(payload.get("example") or ""),
        ),
    )

    parameter_aliases: dict[str, list[str]] = {}
    alias_bag: list[str] = []
    for parameter_name in parameter_names:
        aliases = _coerce_text_list((payload.get("parameter_aliases") or {}).get(parameter_name))
        merged_aliases = _sanitize_parameter_aliases(parameter_name, aliases)
        parameter_aliases[parameter_name] = merged_aliases
        alias_bag.extend(merged_aliases)

    title = _clean_text(payload.get("title"))
    purpose = _clean_text(payload.get("purpose"))
    eligibility = _clean_text(payload.get("eligibility"))
    specialty = _clean_text(payload.get("specialty"))
    parameter_names_text = " ".join(parameter_names)
    parameter_aliases_text = " ".join(_dedupe_texts(alias_bag))
    parameter_text = parameter_aliases_text or parameter_names_text

    document_text = "\n".join(
        part
        for part in (
            title,
            purpose,
            eligibility,
            specialty,
            parameter_names_text,
            parameter_aliases_text,
            parameter_text,
        )
        if part
    ).strip()

    return {
        "calc_id": str(calc_id).strip(),
        "title": title,
        "purpose": purpose,
        "eligibility": eligibility,
        "specialty": specialty,
        "parameter_names": parameter_names,
        "parameter_aliases": parameter_aliases,
        "parameter_names_text": parameter_names_text,
        "parameter_aliases_text": parameter_aliases_text,
        "parameter_text": parameter_text,
        "document_text": document_text,
    }


class RiskCalcParameterRetrievalTool:
    DEFAULT_FIELD_WEIGHTS: dict[str, float] = {
        # Title/purpose are higher-precision than raw parameter text and should
        # dominate once noisy generated parameter phrases are filtered.
        "title": 2.8,
        "purpose": 1.4,
        "eligibility": 0.8,
        "specialty": 0.4,
        "parameter_names_text": 1.8,
        "parameter_aliases_text": 1.5,
        "parameter_text": 0.9,
        "document_text": 0.35,
    }

    def __init__(
        self,
        *,
        parameter_path: str | Path | None = None,
        field_weights: Mapping[str, float] | None = None,
    ) -> None:
        self.parameter_path = self._resolve_parameter_path(parameter_path)
        payload = json.loads(self.parameter_path.read_text(encoding="utf-8"))
        self.documents = [
            build_parameter_document_payload(calc_id, calc_payload)
            for calc_id, calc_payload in sorted(payload.items(), key=lambda item: str(item[0]))
        ]
        self.field_weights = {
            **self.DEFAULT_FIELD_WEIGHTS,
            **{
                str(name): float(weight)
                for name, weight in dict(field_weights or {}).items()
            },
        }
        self._bm25 = FieldedBM25Index(
            [
                {
                    "title": str(document.get("title") or ""),
                    "purpose": str(document.get("purpose") or ""),
                    "eligibility": str(document.get("eligibility") or ""),
                    "specialty": str(document.get("specialty") or ""),
                    "parameter_names_text": str(document.get("parameter_names_text") or ""),
                    "parameter_aliases_text": str(document.get("parameter_aliases_text") or ""),
                    "parameter_text": str(document.get("parameter_text") or ""),
                    "document_text": str(document.get("document_text") or ""),
                }
                for document in self.documents
            ],
            field_weights=self.field_weights,
        )

    @staticmethod
    def _resolve_parameter_path(parameter_path: str | Path | None) -> Path:
        if parameter_path is not None:
            return Path(parameter_path).resolve()
        project_root = Path(__file__).resolve().parents[2]
        candidates = [
            project_root / "data" / "data" / "riskcalcs_parameter.json",
            project_root / "data" / "riskcalcs_parameter.json",
            project_root / "数据" / "riskcalcs_parameter.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def retrieve(self, query: str, *, top_k: int = 10) -> list[dict[str, Any]]:
        cleaned_query = _clean_text(query)
        if not self.documents:
            return []

        top_limit = max(int(top_k), 1)
        bm25_scores = self._bm25.score(cleaned_query)
        query_tokens = set(tokenize_bm25_text(cleaned_query))
        normalized_query = cleaned_query.lower()

        rows: list[dict[str, Any]] = []
        for index, document in enumerate(self.documents):
            bm25_score = float(bm25_scores[index]) if index < len(bm25_scores) else 0.0
            match_info = self._collect_matches(document, normalized_query, query_tokens)
            matched_parameter_names = match_info["matched_parameter_names"]
            matched_aliases = match_info["matched_aliases"]
            parameter_alias_hit_count = int(match_info["parameter_alias_hit_count"])
            parameter_name_hit_count = int(match_info["parameter_name_hit_count"])
            rerank_score = (
                (parameter_alias_hit_count * 1.0)
                + (parameter_name_hit_count * 0.35)
                + (len(matched_aliases) * 0.05)
            )
            rows.append(
                {
                    "calc_id": document["calc_id"],
                    "title": document["title"],
                    "score": bm25_score + rerank_score,
                    "bm25_score": bm25_score,
                    "rerank_score": rerank_score,
                    "matched_parameter_names": matched_parameter_names,
                    "matched_aliases": matched_aliases,
                    "parameter_names": list(document["parameter_names"]),
                }
            )

        rows.sort(
            key=lambda row: (
                -float(row["score"]),
                -float(row["bm25_score"]),
                -len(list(row.get("matched_parameter_names") or [])),
                str(row.get("title") or "").lower(),
                str(row.get("calc_id") or ""),
            )
        )
        return rows[:top_limit]

    def retrieve_from_case_fields(
        self,
        *,
        raw_text: Any = "",
        case_summary: Any = None,
        problem_list: Any = None,
        known_facts: Any = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        query_text = build_case_query_text(
            raw_text=raw_text,
            case_summary=case_summary,
            problem_list=problem_list,
            known_facts=known_facts,
        )
        rows = self.retrieve(query_text, top_k=top_k)
        return [{**row, "query_text": query_text} for row in rows]

    def retrieve_from_structured_case(
        self,
        structured_case: Mapping[str, Any],
        *,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        case_payload = dict(structured_case or {})
        if isinstance(case_payload.get("structured_case"), Mapping):
            case_payload = dict(case_payload["structured_case"])
        return self.retrieve_from_case_fields(
            raw_text=case_payload.get("raw_text") or case_payload.get("raw_request") or "",
            case_summary=case_payload.get("case_summary"),
            problem_list=case_payload.get("problem_list"),
            known_facts=case_payload.get("known_facts"),
            top_k=top_k,
        )

    def _collect_matches(
        self,
        document: Mapping[str, Any],
        normalized_query: str,
        query_tokens: set[str],
    ) -> dict[str, Any]:
        matched_parameter_names: list[str] = []
        matched_aliases: list[str] = []
        parameter_alias_hit_count = 0
        parameter_name_hit_count = 0

        parameter_aliases = dict(document.get("parameter_aliases") or {})
        for parameter_name in list(document.get("parameter_names") or []):
            aliases = list(parameter_aliases.get(parameter_name) or [])
            name_hit = self._matches_text(parameter_name, normalized_query, query_tokens)
            alias_hits = [alias for alias in aliases if self._matches_text(alias, normalized_query, query_tokens)]
            if not name_hit and not alias_hits:
                continue

            matched_parameter_names.append(str(parameter_name))
            matched_aliases.extend(alias_hits)
            if name_hit:
                parameter_name_hit_count += 1
            if alias_hits:
                parameter_alias_hit_count += 1

        return {
            "matched_parameter_names": _dedupe_texts(matched_parameter_names),
            "matched_aliases": _dedupe_texts(matched_aliases),
            "parameter_alias_hit_count": parameter_alias_hit_count,
            "parameter_name_hit_count": parameter_name_hit_count,
        }

    @staticmethod
    def _matches_text(candidate: str, normalized_query: str, query_tokens: set[str]) -> bool:
        cleaned_candidate = _clean_text(candidate).lower()
        if not cleaned_candidate:
            return False
        candidate_tokens = set(tokenize_bm25_text(cleaned_candidate))
        if not candidate_tokens:
            return False

        # Single-token ASCII aliases are too error-prone for substring matching:
        # "male" would match "female", "ic" would match "clinic", and "nd"
        # would match arbitrary words. Require an exact token hit instead.
        if len(candidate_tokens) == 1 and re.fullmatch(r"[a-z0-9]+", cleaned_candidate):
            return cleaned_candidate in query_tokens

        if cleaned_candidate in normalized_query:
            return True
        return candidate_tokens.issubset(query_tokens)


def create_parameter_retrieval_tool(
    *,
    parameter_path: str | Path | None = None,
    field_weights: Mapping[str, float] | None = None,
) -> RiskCalcParameterRetrievalTool:
    return RiskCalcParameterRetrievalTool(
        parameter_path=parameter_path,
        field_weights=field_weights,
    )


# Retrieval tool orchestration and cache helpers.
def _document_field(document: Any, name: str) -> Any:
    if isinstance(document, dict):
        return document.get(name)
    return getattr(document, name, None)


def _clean_text(value: Any) -> str:
    return _normalize_whitespace(value)


def _document_to_brief(document: Any) -> dict[str, Any]:
    if hasattr(document, "to_brief"):
        payload = document.to_brief()
        if isinstance(payload, dict):
            return dict(payload)
    return {
        "pmid": str(_document_field(document, "pmid") or ""),
        "title": _clean_text(_document_field(document, "title")),
        "purpose": _clean_text(_document_field(document, "purpose")),
        "eligibility": _clean_text(_document_field(document, "eligibility")),
        "taxonomy": dict(_document_field(document, "taxonomy") or {}),
    }


def _normalize_backend_name(backend: str | None, *, default: str = "hybrid") -> str:
    normalized = str(backend or default).strip().lower()
    if not normalized:
        normalized = default
    return {
        "auto": "hybrid",
        "keyword": "parameter",
        "bm25": "parameter",
        "medcpt": "vector",
    }.get(normalized, normalized)


def _file_signature(path: str | Path | None) -> str:
    if path is None:
        return "<missing>"
    file_path = Path(path).resolve()
    try:
        stat = file_path.stat()
    except OSError:
        return str(file_path)
    return f"{file_path}|{stat.st_size}|{stat.st_mtime_ns}"


def _get_cache_key_lock(namespace: str, cache_key: Any) -> threading.Lock:
    normalized_key = (str(namespace), str(cache_key))
    with _CACHE_LOCK:
        lock = _CACHE_KEY_LOCKS.get(normalized_key)
        if lock is None:
            lock = threading.Lock()
            _CACHE_KEY_LOCKS[normalized_key] = lock
    return lock


def _normalize_field_weight_items(
    field_weights: Mapping[str, float] | None,
) -> tuple[tuple[str, float], ...]:
    return tuple(
        sorted(
            (str(name), float(weight))
            for name, weight in dict(field_weights or {}).items()
        )
    )


def _catalog_runtime_cache_key(catalog: Any) -> str:
    runtime_cache_key = str(getattr(catalog, "runtime_cache_key", "") or "").strip()
    if runtime_cache_key:
        return runtime_cache_key
    return f"inmemory:{id(catalog)}"


def _parameter_retriever_cache_key(
    *,
    parameter_path: str | Path | None,
    field_weights: Mapping[str, float] | None,
) -> tuple[str, tuple[tuple[str, float], ...]]:
    resolved_path = Path(parameter_path).resolve() if parameter_path is not None else None
    return (
        _file_signature(resolved_path),
        _normalize_field_weight_items(field_weights),
    )


def _build_vector_retriever_cache_token(
    *,
    catalog: Any,
    vector_retriever: Any | None,
) -> str:
    if vector_retriever is None:
        return f"inline:{_catalog_runtime_cache_key(catalog)}"
    return f"custom:{id(vector_retriever)}"


def _get_or_create_parameter_retriever(
    *,
    parameter_path: str | Path | None,
    field_weights: Mapping[str, float] | None,
) -> RiskCalcParameterRetrievalTool | None:
    if parameter_path is None:
        return None
    cache_key = _parameter_retriever_cache_key(
        parameter_path=parameter_path,
        field_weights=field_weights,
    )
    with _CACHE_LOCK:
        cached = _PARAMETER_RETRIEVER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    with _get_cache_key_lock("parameter_retriever", cache_key):
        with _CACHE_LOCK:
            cached = _PARAMETER_RETRIEVER_CACHE.get(cache_key)
        if cached is not None:
            return cached
        try:
            retriever = RiskCalcParameterRetrievalTool(
                parameter_path=parameter_path,
                field_weights=field_weights,
            )
        except FileNotFoundError:
            return None
        with _CACHE_LOCK:
            cached = _PARAMETER_RETRIEVER_CACHE.setdefault(cache_key, retriever)
        return cached


def _resolve_catalog_parameter_payload_path(
    catalog: Any,
    parameter_path: str | Path | None,
) -> Path | None:
    if parameter_path is not None:
        return Path(parameter_path).resolve()

    source_files = getattr(catalog, "source_files", None)
    if not source_files:
        return None

    try:
        riskcalcs_path = Path(source_files[0]).resolve()
    except Exception:
        return None

    candidates = [
        riskcalcs_path.with_name("riskcalcs_parameter.json"),
        riskcalcs_path.parent / "riskcalcs_parameter.json",
        riskcalcs_path.parent.parent / "data" / "riskcalcs_parameter.json",
        riskcalcs_path.parent.parent / "data" / "data" / "riskcalcs_parameter.json",
        riskcalcs_path.parent.parent / "数据" / "riskcalcs_parameter.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _get_or_create_inline_vector_retriever(catalog: Any) -> Any | None:
    cache_key = _catalog_runtime_cache_key(catalog)
    with _CACHE_LOCK:
        cached = _INLINE_MEDCPT_RETRIEVER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    with _get_cache_key_lock("inline_medcpt_retriever", cache_key):
        with _CACHE_LOCK:
            cached = _INLINE_MEDCPT_RETRIEVER_CACHE.get(cache_key)
        if cached is not None:
            return cached
        try:
            retriever = _InlineMedCPTRetriever(catalog)
        except Exception:
            return None
        with _CACHE_LOCK:
            cached = _INLINE_MEDCPT_RETRIEVER_CACHE.setdefault(cache_key, retriever)
        return cached


def _load_riskcalcs_payload_index(catalog: Any) -> dict[str, dict[str, Any]]:
    source_files = getattr(catalog, "source_files", None)
    if not source_files:
        return {}
    try:
        riskcalcs_path = Path(source_files[0]).resolve()
    except Exception:
        return {}
    if not riskcalcs_path.exists():
        return {}

    cache_key = _file_signature(riskcalcs_path)
    with _CACHE_LOCK:
        cached = _RISKCALCS_PAYLOAD_CACHE.get(cache_key)
    if cached is not None:
        return cached

    with _get_cache_key_lock("riskcalcs_payload", cache_key):
        with _CACHE_LOCK:
            cached = _RISKCALCS_PAYLOAD_CACHE.get(cache_key)
        if cached is not None:
            return cached

        try:
            payload = json.loads(riskcalcs_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        normalized = {
            str(calc_id): dict(calc_payload or {})
            for calc_id, calc_payload in dict(payload or {}).items()
        }
        with _CACHE_LOCK:
            cached = _RISKCALCS_PAYLOAD_CACHE.setdefault(cache_key, normalized)
        return cached


def _resolve_department_payload_root(catalog: Any) -> Path | None:
    explicit_root = getattr(catalog, "department_payload_root", None)
    if explicit_root:
        root_path = Path(explicit_root).resolve()
        if root_path.exists():
            return root_path

    source_files = getattr(catalog, "source_files", None)
    if not source_files:
        return None

    try:
        riskcalcs_path = Path(source_files[0]).resolve()
    except Exception:
        return None

    candidate_roots = [
        riskcalcs_path.parent / "计算器科室",
        riskcalcs_path.parent.parent / "数据" / "计算器科室",
        riskcalcs_path.parent.parent / "data" / "计算器科室",
    ]
    for department_root in candidate_roots:
        if department_root.exists():
            return department_root
    return None


def _load_department_pmids(department_file: Path) -> set[str]:
    cache_key = _file_signature(department_file)
    with _CACHE_LOCK:
        cached = _DEPARTMENT_PMID_CACHE.get(cache_key)
    if cached is not None:
        return set(cached)

    with _get_cache_key_lock("department_pmids", cache_key):
        with _CACHE_LOCK:
            cached = _DEPARTMENT_PMID_CACHE.get(cache_key)
        if cached is not None:
            return set(cached)

        try:
            payload = json.loads(department_file.read_text(encoding="utf-8"))
        except Exception:
            pmids: set[str] = set()
        else:
            pmids = {
                str(calc_id).strip()
                for calc_id in dict(payload or {}).keys()
                if str(calc_id).strip()
            }
        with _CACHE_LOCK:
            cached = _DEPARTMENT_PMID_CACHE.setdefault(cache_key, set(pmids))
        return set(cached)


class _InlineMedCPTRetriever:
    QUERY_MODEL_NAME = "ncbi/MedCPT-Query-Encoder"
    DOC_MODEL_NAME = "ncbi/MedCPT-Article-Encoder"

    def __init__(self, catalog: Any) -> None:
        try:
            import faiss  # type: ignore
            import torch  # type: ignore
            from transformers import AutoModel, AutoTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on local environment
            raise RuntimeError("MedCPT retrieval requires faiss, torch, and transformers.") from exc

        self.catalog = catalog
        self._faiss = faiss
        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._query_encoder, self._doc_encoder, self._tokenizer = self._load_shared_resources(
            AutoModel=AutoModel,
            AutoTokenizer=AutoTokenizer,
        )
        self._pmids = [str(_document_field(document, "pmid") or "") for document in catalog.documents()]
        self._pmid_to_index = {str(pmid): index for index, pmid in enumerate(self._pmids)}
        self._inference_lock = threading.RLock()
        self._index = self._load_or_build_index()

    def _load_shared_resources(self, *, AutoModel: Any, AutoTokenizer: Any) -> tuple[Any, Any, Any]:
        cache_key = (self._device, self.QUERY_MODEL_NAME, self.DOC_MODEL_NAME)
        with _CACHE_LOCK:
            cached = _MEDCPT_RESOURCE_CACHE.get(cache_key)
        if cached is not None:
            return cached

        with _get_cache_key_lock("medcpt_resources", cache_key):
            with _CACHE_LOCK:
                cached = _MEDCPT_RESOURCE_CACHE.get(cache_key)
            if cached is not None:
                return cached

            query_encoder = AutoModel.from_pretrained(self.QUERY_MODEL_NAME).to(self._device)
            doc_encoder = AutoModel.from_pretrained(self.DOC_MODEL_NAME).to(self._device)
            tokenizer = AutoTokenizer.from_pretrained(self.QUERY_MODEL_NAME)
            if hasattr(query_encoder, "eval"):
                query_encoder.eval()
            if hasattr(doc_encoder, "eval"):
                doc_encoder.eval()

            with _CACHE_LOCK:
                cached = _MEDCPT_RESOURCE_CACHE.setdefault(cache_key, (query_encoder, doc_encoder, tokenizer))
            return cached

    def _load_or_build_index(self):
        cache_paths = None
        dense_index_cache_paths = getattr(self.catalog, "dense_index_cache_paths", None)
        if callable(dense_index_cache_paths):
            try:
                cache_paths = dense_index_cache_paths(
                    query_model_name=self.QUERY_MODEL_NAME,
                    doc_model_name=self.DOC_MODEL_NAME,
                )
            except TypeError:
                cache_paths = None

        if cache_paths is not None:
            index_path, pmids_path = cache_paths
            try:
                if index_path.exists() and pmids_path.exists():
                    cached_pmids = json.loads(pmids_path.read_text(encoding="utf-8"))
                    if list(cached_pmids) == self._pmids:
                        return self._faiss.read_index(str(index_path))
            except Exception:
                pass

        index = self._build_index()
        if cache_paths is not None:
            index_path, pmids_path = cache_paths
            try:
                index_path.parent.mkdir(parents=True, exist_ok=True)
                self._faiss.write_index(index, str(index_path))
                pmids_path.write_text(json.dumps(self._pmids, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass
        return index

    def _build_index(self):  # pragma: no cover - depends on local environment
        tool_texts = []
        for document in self.catalog.documents():
            title = _clean_text(_document_field(document, "title"))
            retrieval_text = _clean_text(_document_field(document, "retrieval_text"))
            abstract = _clean_text(_document_field(document, "abstract"))
            purpose = _clean_text(_document_field(document, "purpose"))
            tool_texts.append([title, retrieval_text or abstract or purpose])

        embeddings = []
        with self._torch.no_grad():
            for start in range(0, len(tool_texts), 16):
                encoded = self._tokenizer(
                    tool_texts[start : start + 16],
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=512,
                )
                encoded.to(self._device)
                model_output = self._doc_encoder(**encoded)
                embeddings.extend(model_output.last_hidden_state[:, 0, :].detach().cpu())

        matrix = self._torch.stack(embeddings).numpy()
        index = self._faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return index

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        *,
        candidate_pmids: set[str] | None = None,
    ) -> list[dict[str, Any]]:  # pragma: no cover
        with self._inference_lock:
            with self._torch.no_grad():
                encoded = self._tokenizer(
                    [query],
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=512,
                )
                encoded.to(self._device)
                model_output = self._query_encoder(**encoded)
                query_embedding = model_output.last_hidden_state[:, 0, :].detach().cpu().numpy()

        results = []
        scored_pmids = _retrieve_dense_scored_pmids(
            index=self._index,
            query_embedding=query_embedding,
            pmids=self._pmids,
            pmid_to_index=getattr(
                self,
                "_pmid_to_index",
                {str(pmid): index for index, pmid in enumerate(self._pmids)},
            ),
            candidate_pmids=candidate_pmids,
            top_k=top_k,
        )
        for pmid, score in scored_pmids:
            document = self.catalog.get(pmid)
            payload = _document_to_brief(document)
            payload["score"] = float(score)
            results.append(payload)
        return results


class RiskCalcRetrievalTool:
    def __init__(
        self,
        catalog: Any,
        *,
        parameter_retriever: RiskCalcParameterRetrievalTool | None = None,
        vector_retriever: Any | None = None,
        backend: str = "hybrid",
    ) -> None:
        self.catalog = catalog
        self.catalog_keyword_retriever = KeywordToolRetriever(catalog)
        self.parameter_retriever = parameter_retriever
        self.vector_retriever = vector_retriever
        self._riskcalcs_payload_index = _load_riskcalcs_payload_index(catalog)
        self.default_backend = self.resolve_backend(backend)
        self.structured_tools = StructuredRetrievalTool(
            catalog,
            bm25_retriever=parameter_retriever,
            vector_retriever=vector_retriever,
            query_builder=build_case_query_text,
            default_backend=self.default_backend,
            id_field="pmid",
        )

    @property
    def available_backends(self) -> tuple[str, ...]:
        if self.vector_retriever is None:
            return ("parameter",)
        return ("parameter", "vector", "hybrid")

    def resolve_backend(self, backend: str | None = None) -> str:
        normalized = _normalize_backend_name(backend, default="hybrid")
        if normalized not in {"parameter", "vector", "hybrid"}:
            normalized = "parameter"
        if normalized in {"vector", "hybrid"} and self.vector_retriever is None:
            return "parameter"
        return normalized

    def retrieve_from_case_fields(
        self,
        *,
        raw_text: Any = "",
        case_summary: Any = None,
        problem_list: Any = None,
        known_facts: Any = None,
        department_tags: Any = None,
        top_k: int = 10,
        candidate_pmids: set[str] | None = None,
        backend: str | None = None,
        include_scores: bool = False,
    ) -> dict[str, Any]:
        query_text = build_case_query_text(
            raw_text=raw_text,
            case_summary=case_summary,
            problem_list=problem_list,
            known_facts=known_facts,
        )
        department_candidate_pmids = self._resolve_department_candidate_pmids(department_tags)
        scoped_candidate_pmids = self._resolve_candidate_pmids(
            candidate_pmids=candidate_pmids,
            department_tags=department_tags,
        )
        bundle = self._build_retrieval_bundle(
            query_text=query_text,
            top_k=top_k,
            candidate_pmids=scoped_candidate_pmids,
            backend=backend,
        )
        fallback_triggered = False
        if department_candidate_pmids is not None and not list(bundle.get("candidate_ranking") or []):
            fallback_triggered = True
            fallback_candidate_pmids = set(candidate_pmids) if candidate_pmids is not None else None
            bundle = self._build_retrieval_bundle(
                query_text=query_text,
                top_k=top_k,
                candidate_pmids=fallback_candidate_pmids,
                backend=backend,
            )
        bundle["department_tags"] = [
            str(tag).strip()
            for tag in list(department_tags or [])
            if str(tag).strip()
        ]
        bundle["department_candidate_pmids"] = (
            sorted(department_candidate_pmids)
            if department_candidate_pmids is not None
            else None
        )
        bundle["fallback_to_full_catalog"] = fallback_triggered
        if include_scores:
            return bundle
        return self._build_public_retrieval_bundle(bundle)

    def retrieve_coarse_from_case_fields(
        self,
        *,
        raw_text: Any = "",
        case_summary: Any = None,
        problem_list: Any = None,
        known_facts: Any = None,
        department_tags: Any = None,
        top_k: int = _DEFAULT_COARSE_RETRIEVAL_TOP_K,
        candidate_pmids: set[str] | None = None,
        backend: str | None = None,
    ) -> dict[str, Any]:
        query_text = build_case_query_text(
            raw_text=raw_text,
            case_summary=case_summary,
            problem_list=problem_list,
            known_facts=known_facts,
        )
        department_candidate_pmids = self._resolve_department_candidate_pmids(department_tags)
        scoped_candidate_pmids = self._resolve_candidate_pmids(
            candidate_pmids=candidate_pmids,
            department_tags=department_tags,
        )
        bundle = self._build_coarse_retrieval_bundle(
            query_text=query_text,
            top_k=top_k,
            candidate_pmids=scoped_candidate_pmids,
            backend=backend,
        )
        fallback_triggered = False
        if department_candidate_pmids is not None and not list(bundle.get("candidate_ranking") or []):
            fallback_triggered = True
            fallback_candidate_pmids = set(candidate_pmids) if candidate_pmids is not None else None
            bundle = self._build_coarse_retrieval_bundle(
                query_text=query_text,
                top_k=top_k,
                candidate_pmids=fallback_candidate_pmids,
                backend=backend,
            )

        bundle["department_tags"] = [
            str(tag).strip()
            for tag in list(department_tags or [])
            if str(tag).strip()
        ]
        bundle["department_candidate_pmids"] = (
            sorted(department_candidate_pmids)
            if department_candidate_pmids is not None
            else None
        )
        bundle["fallback_to_full_catalog"] = fallback_triggered
        return self._build_public_coarse_retrieval_bundle(bundle)

    @tool(
        name="riskcalc_coarse_retriever",
        description=(
            "Coarsely recall calculator candidates from structured_case with a configurable recall size. "
            "Return only PMID and title so the agent can decide which pool to pass into the second-stage matcher."
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
            "candidate_pmids": "set[str] | None",
        },
        state_fields={
            "structured_case": (
                "structured_case",
                "structured_case_json",
                "clinical_tool_job.structured_case",
            ),
            "top_k": (
                "top_k",
                "clinical_tool_job.top_k",
            ),
            "department_tags": (
                "department_tags",
                "structured_case.department_tags",
                "clinical_tool_job.structured_case.department_tags",
            ),
        },
    )
    def retrieve_coarse_from_structured_case(
        self,
        structured_case: Mapping[str, Any],
        *,
        top_k: int = _DEFAULT_COARSE_RETRIEVAL_TOP_K,
        department_tags: Any = None,
        candidate_pmids: set[str] | None = None,
    ) -> dict[str, Any]:
        case_payload = dict(structured_case or {})
        if isinstance(case_payload.get("structured_case"), Mapping):
            case_payload = dict(case_payload["structured_case"])
        resolved_department_tags = department_tags
        if resolved_department_tags is None:
            resolved_department_tags = case_payload.get("department_tags")
        return self.retrieve_coarse_from_case_fields(
            raw_text=case_payload.get("raw_text") or case_payload.get("raw_request") or "",
            case_summary=case_payload.get("case_summary"),
            problem_list=case_payload.get("problem_list"),
            known_facts=case_payload.get("known_facts"),
            department_tags=resolved_department_tags,
            top_k=top_k,
            candidate_pmids=candidate_pmids,
        )

    @tool(
        name="riskcalc_parameter_retriever",
        description=(
            "Retrieve calculator candidates from structured_case and return a compact JSON bundle "
            "mapped from the calculator database with PMID plus key calculator fields."
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
            "candidate_pmids": "set[str] | None",
            "include_scores": "bool",
        },
        state_fields={
            "structured_case": (
                "structured_case",
                "structured_case_json",
                "clinical_tool_job.structured_case",
            ),
            "top_k": (
                "top_k",
                "clinical_tool_job.top_k",
            ),
            "department_tags": (
                "department_tags",
                "structured_case.department_tags",
                "clinical_tool_job.structured_case.department_tags",
            ),
        },
    )
    def retrieve_from_structured_case(
        self,
        structured_case: Mapping[str, Any],
        *,
        top_k: int = 10,
        department_tags: Any = None,
        candidate_pmids: set[str] | None = None,
        include_scores: bool = False,
    ) -> dict[str, Any]:
        case_payload = dict(structured_case or {})
        if isinstance(case_payload.get("structured_case"), Mapping):
            case_payload = dict(case_payload["structured_case"])
        resolved_department_tags = department_tags
        if resolved_department_tags is None:
            resolved_department_tags = case_payload.get("department_tags")
        return self.retrieve_from_case_fields(
            raw_text=case_payload.get("raw_text") or case_payload.get("raw_request") or "",
            case_summary=case_payload.get("case_summary"),
            problem_list=case_payload.get("problem_list"),
            known_facts=case_payload.get("known_facts"),
            department_tags=resolved_department_tags,
            top_k=top_k,
            candidate_pmids=candidate_pmids,
            include_scores=include_scores,
        )

    def _resolve_candidate_pmids(
        self,
        *,
        candidate_pmids: set[str] | None,
        department_tags: Any,
    ) -> set[str] | None:
        department_filtered_pmids = self._resolve_department_candidate_pmids(department_tags)
        if department_filtered_pmids is None:
            return candidate_pmids
        if candidate_pmids is None:
            return set(department_filtered_pmids)
        return set(candidate_pmids).intersection(department_filtered_pmids)

    def _resolve_department_candidate_pmids(self, department_tags: Any) -> set[str] | None:
        normalized_tags = [
            str(tag).strip()
            for tag in list(department_tags or [])
            if str(tag).strip()
        ]
        if not normalized_tags:
            return None

        department_root = _resolve_department_payload_root(self.catalog)
        if department_root is None:
            return None

        pmids: set[str] = set()
        found_department_file = False
        for department_tag in normalized_tags:
            department_file = department_root / department_tag / "riskcalcs.json"
            if not department_file.exists():
                continue
            found_department_file = True
            pmids.update(_load_department_pmids(department_file))

        if not found_department_file:
            return None
        return pmids

    def _build_retrieval_bundle(
        self,
        *,
        query_text: str,
        top_k: int,
        candidate_pmids: set[str] | None,
        backend: str | None = None,
    ) -> dict[str, Any]:
        resolved_backend = self.resolve_backend(backend or self.default_backend)
        requested_top_k = max(int(top_k), 1)
        branch_top_k = (
            _resolve_hybrid_branch_top_k(requested_top_k)
            if resolved_backend == "hybrid"
            else requested_top_k
        )

        bm25_rows = self._retrieve_parameter_rows(
            query_text=query_text,
            top_k=branch_top_k,
            candidate_pmids=candidate_pmids,
        )
        vector_rows = self._retrieve_vector_rows(
            query_text=query_text,
            top_k=branch_top_k if resolved_backend == "hybrid" else max(requested_top_k, 5),
            candidate_pmids=candidate_pmids,
        )
        retrieved_tools = self._build_candidate_union(
            bm25_rows=bm25_rows,
            vector_rows=vector_rows,
            top_k=requested_top_k,
            backend=resolved_backend,
        )
        bm25_top5 = self._build_context_top_hits(bm25_rows, limit=5)
        vector_top5 = self._build_context_top_hits(vector_rows, limit=5)
        bm25_raw_top5 = self._build_full_context_hits(bm25_rows, limit=5, channel="bm25")
        vector_raw_top5 = self._build_full_context_hits(vector_rows, limit=5, channel="vector")
        return {
            "query_text": query_text,
            "backend_used": resolved_backend,
            "available_backends": list(self.available_backends),
            "hybrid_branch_top_k": branch_top_k if resolved_backend == "hybrid" else None,
            "bm25_top5": bm25_top5,
            "vector_top5": vector_top5,
            "bm25_raw_top5": bm25_raw_top5,
            "vector_raw_top5": vector_raw_top5,
            "model_context": {
                "bm25_top5": dict(bm25_top5),
                "vector_top5": dict(vector_top5),
            },
            "bm25_candidates": list(bm25_rows),
            "vector_candidates": list(vector_rows),
            "retrieved_tools": list(retrieved_tools),
            "candidate_ranking": list(retrieved_tools),
        }

    def _build_coarse_retrieval_bundle(
        self,
        *,
        query_text: str,
        top_k: int,
        candidate_pmids: set[str] | None,
        backend: str | None = None,
    ) -> dict[str, Any]:
        resolved_backend = self.resolve_backend(backend or self.default_backend)
        requested_top_k = max(int(top_k), 1)
        branch_top_k = (
            _resolve_hybrid_branch_top_k(requested_top_k)
            if resolved_backend == "hybrid"
            else requested_top_k
        )

        # Coarse recall should stay broad and use the legacy full-catalog
        # keyword retriever instead of parameter-only matching.
        bm25_rows = self._retrieve_catalog_keyword_rows(
            query_text=query_text,
            top_k=branch_top_k,
            candidate_pmids=candidate_pmids,
        )
        vector_rows = self._retrieve_vector_rows(
            query_text=query_text,
            top_k=branch_top_k,
            candidate_pmids=candidate_pmids,
        )
        retrieved_tools = self._build_candidate_union(
            bm25_rows=bm25_rows,
            vector_rows=vector_rows,
            top_k=requested_top_k,
            backend=resolved_backend,
        )
        return {
            "query_text": query_text,
            "backend_used": resolved_backend,
            "available_backends": list(self.available_backends),
            "hybrid_branch_top_k": branch_top_k if resolved_backend == "hybrid" else None,
            "bm25_candidate_pmids": [str(row.get("pmid") or "").strip() for row in bm25_rows if str(row.get("pmid") or "").strip()],
            "vector_candidate_pmids": [str(row.get("pmid") or "").strip() for row in vector_rows if str(row.get("pmid") or "").strip()],
            "retrieved_tools": list(retrieved_tools),
            "candidate_ranking": list(retrieved_tools),
        }

    @staticmethod
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

    def _resolve_public_parameter_names(self, pmid: str, row: Mapping[str, Any]) -> list[str]:
        raw_payload = dict(self._riskcalcs_payload_index.get(pmid) or {})
        return _merge_parameter_name_sources(
            row.get("parameter_names") or [],
            extract_parameter_names_from_computation(
                str(raw_payload.get("computation") or ""),
                example=str(raw_payload.get("example") or ""),
            ),
        )

    def _build_public_mapped_candidate(
        self,
        row: Mapping[str, Any],
        *,
        rank: int | None = None,
        recommended_channels: Mapping[str, list[str]],
    ) -> dict[str, Any]:
        pmid = str(row.get("pmid") or "").strip()
        calculator_payload = self._build_full_calculator_payload(pmid)
        payload: dict[str, Any] = {
            "pmid": pmid,
            "title": _clean_text(calculator_payload.get("title") or row.get("title")),
            "purpose": _clean_text(calculator_payload.get("purpose") or row.get("purpose")),
            "specialty": _clean_text(calculator_payload.get("specialty")),
            "eligibility": _clean_text(calculator_payload.get("eligibility") or row.get("eligibility")),
            "parameter_names": self._resolve_public_parameter_names(pmid, row),
            "matched_parameter_names": list(row.get("matched_parameter_names") or []),
            "matched_aliases": list(row.get("matched_aliases") or []),
            "recommended": list(recommended_channels.get(pmid) or []),
        }
        if rank is not None:
            payload["rank"] = rank
        return payload

    @staticmethod
    def _build_public_full_context_hits(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        public_hits: list[dict[str, Any]] = []
        for row in list(rows or []):
            public_hits.append(
                {
                    "rank": int(row.get("rank") or 0),
                    "channel": str(row.get("channel") or ""),
                    "pmid": str(row.get("pmid") or ""),
                    "matched_parameter_names": list(row.get("matched_parameter_names") or []),
                    "matched_aliases": list(row.get("matched_aliases") or []),
                    "parameter_names": list(row.get("parameter_names") or []),
                    "match_sources": sorted(set(list(row.get("match_sources") or []))),
                    "query_text": str(row.get("query_text") or ""),
                    "calculator_payload": dict(row.get("calculator_payload") or {}),
                }
            )
        return public_hits

    def _build_public_retrieval_bundle(self, bundle: Mapping[str, Any]) -> dict[str, Any]:
        raw_bundle = dict(bundle or {})
        bm25_top5 = dict(raw_bundle.get("bm25_top5") or {})
        vector_top5 = dict(raw_bundle.get("vector_top5") or {})
        bm25_raw_top5 = [dict(row) for row in list(raw_bundle.get("bm25_raw_top5") or [])]
        vector_raw_top5 = [dict(row) for row in list(raw_bundle.get("vector_raw_top5") or [])]
        recommended_channels = self._build_recommended_channels(
            bm25_raw_top5=bm25_raw_top5,
            vector_raw_top5=vector_raw_top5,
        )

        retrieved_tools = [
            self._build_public_mapped_candidate(
                dict(row),
                recommended_channels=recommended_channels,
            )
            for row in list(raw_bundle.get("retrieved_tools") or raw_bundle.get("candidate_ranking") or [])
        ]

        return {
            "query_text": str(raw_bundle.get("query_text") or ""),
            "backend_used": raw_bundle.get("backend_used"),
            "available_backends": list(raw_bundle.get("available_backends") or []),
            "hybrid_branch_top_k": raw_bundle.get("hybrid_branch_top_k"),
            "department_tags": list(raw_bundle.get("department_tags") or []),
            "fallback_to_full_catalog": bool(raw_bundle.get("fallback_to_full_catalog")),
            "bm25_top5": bm25_top5,
            "vector_top5": vector_top5,
            "bm25_raw_top5": self._build_public_full_context_hits(bm25_raw_top5),
            "vector_raw_top5": self._build_public_full_context_hits(vector_raw_top5),
            "model_context": {
                "bm25_top5": bm25_top5,
                "vector_top5": vector_top5,
            },
            "retrieved_tools": retrieved_tools,
            "candidate_ranking": retrieved_tools,
            "recommended_pmids": list(recommended_channels.keys()),
        }

    def _build_public_coarse_candidate(self, row: Mapping[str, Any]) -> dict[str, str]:
        pmid = str(row.get("pmid") or "").strip()
        title = ""
        if pmid:
            title = _clean_text(self._build_full_calculator_payload(pmid).get("title"))
        if not title:
            title = _clean_text(row.get("title"))
        return {
            "pmid": pmid,
            "title": title,
        }

    def _build_public_coarse_retrieval_bundle(self, bundle: Mapping[str, Any]) -> dict[str, Any]:
        raw_bundle = dict(bundle or {})
        candidate_rows = [
            dict(row)
            for row in list(raw_bundle.get("candidate_ranking") or raw_bundle.get("retrieved_tools") or [])
        ]
        candidates = [self._build_public_coarse_candidate(row) for row in candidate_rows]
        return {
            "query_text": str(raw_bundle.get("query_text") or ""),
            "backend_used": raw_bundle.get("backend_used"),
            "available_backends": list(raw_bundle.get("available_backends") or []),
            "hybrid_branch_top_k": raw_bundle.get("hybrid_branch_top_k"),
            "department_tags": list(raw_bundle.get("department_tags") or []),
            "fallback_to_full_catalog": bool(raw_bundle.get("fallback_to_full_catalog")),
            "bm25_candidate_pmids": [
                str(pmid).strip()
                for pmid in list(raw_bundle.get("bm25_candidate_pmids") or [])
                if str(pmid).strip()
            ],
            "vector_candidate_pmids": [
                str(pmid).strip()
                for pmid in list(raw_bundle.get("vector_candidate_pmids") or [])
                if str(pmid).strip()
            ],
            "retrieved_tools": candidates,
            "candidate_ranking": candidates,
            "candidate_pmids": [candidate["pmid"] for candidate in candidates if candidate.get("pmid")],
        }

    def _retrieve_parameter_rows(
        self,
        *,
        query_text: str,
        top_k: int,
        candidate_pmids: set[str] | None,
    ) -> list[dict[str, Any]]:
        if self.parameter_retriever is None:
            return self._retrieve_catalog_keyword_rows(
                query_text=query_text,
                top_k=top_k,
                candidate_pmids=candidate_pmids,
            )
        rows = self.parameter_retriever.retrieve(query_text, top_k=top_k)
        serialized_rows: list[dict[str, Any]] = []
        for row in list(rows):
            calc_id = str(row.get("calc_id") or row.get("pmid") or "").strip()
            if not calc_id:
                continue
            if candidate_pmids is not None and calc_id not in candidate_pmids:
                continue
            serialized = self._serialize_catalog_row(calc_id, row)
            serialized["query_text"] = query_text
            serialized["match_sources"] = sorted(
                {
                    *list(serialized.get("match_sources") or []),
                    "parameter",
                }
            )
            serialized_rows.append(serialized)
        return serialized_rows[:top_k]

    def _retrieve_catalog_keyword_rows(
        self,
        *,
        query_text: str,
        top_k: int,
        candidate_pmids: set[str] | None,
    ) -> list[dict[str, Any]]:
        rows = self.catalog_keyword_retriever.retrieve(
            query_text,
            top_k=top_k,
            candidate_pmids=candidate_pmids,
        )
        serialized_rows: list[dict[str, Any]] = []
        for row in list(rows):
            calc_id = str(row.get("calc_id") or row.get("pmid") or "").strip()
            if not calc_id:
                continue
            serialized = self._serialize_catalog_row(calc_id, row)
            serialized["bm25_score"] = float(row.get("score") or 0.0)
            serialized["score"] = float(serialized["bm25_score"])
            serialized["query_text"] = query_text
            serialized["match_sources"] = sorted(
                {
                    *list(serialized.get("match_sources") or []),
                    "parameter",
                }
            )
            serialized_rows.append(serialized)
        return serialized_rows[:top_k]

    def _retrieve_vector_rows(
        self,
        *,
        query_text: str,
        top_k: int,
        candidate_pmids: set[str] | None,
    ) -> list[dict[str, Any]]:
        if self.vector_retriever is None:
            return []
        try:
            rows = self.vector_retriever.retrieve(
                query_text,
                top_k=top_k,
                candidate_pmids=candidate_pmids,
            )
        except TypeError:
            rows = self.vector_retriever.retrieve(query_text, top_k=top_k)
            if candidate_pmids is not None:
                rows = [row for row in list(rows) if str(row.get("pmid") or "").strip() in candidate_pmids]

        serialized_rows: list[dict[str, Any]] = []
        for row in list(rows):
            calc_id = str(row.get("calc_id") or row.get("pmid") or "").strip()
            if not calc_id:
                continue
            serialized = self._serialize_catalog_row(calc_id, row)
            serialized["vector_score"] = float(row.get("score") or 0.0)
            serialized["score"] = float(serialized["vector_score"])
            serialized["query_text"] = query_text
            serialized["match_sources"] = sorted(
                {
                    *list(serialized.get("match_sources") or []),
                    "vector",
                }
            )
            serialized_rows.append(serialized)
        return serialized_rows[:top_k]

    def _serialize_catalog_row(self, calc_id: str, row: Mapping[str, Any]) -> dict[str, Any]:
        document = None
        if self.catalog is not None:
            try:
                document = self.catalog.get(calc_id)
            except Exception:
                document = None

        payload = (
            _document_to_brief(document)
            if document is not None
            else {
                "pmid": calc_id,
                "title": _clean_text(row.get("title")),
                "purpose": _clean_text(row.get("purpose")),
                "eligibility": _clean_text(row.get("eligibility")),
                "taxonomy": {},
            }
        )
        raw_payload = dict(self._riskcalcs_payload_index.get(calc_id) or {})
        payload["pmid"] = calc_id
        payload["title"] = _clean_text(raw_payload.get("title") or payload.get("title") or row.get("title"))
        payload["purpose"] = _clean_text(raw_payload.get("purpose") or payload.get("purpose") or row.get("purpose"))
        payload["specialty"] = _clean_text(
            raw_payload.get("specialty") or payload.get("specialty") or row.get("specialty")
        )
        payload["eligibility"] = _clean_text(
            raw_payload.get("eligibility") or payload.get("eligibility") or row.get("eligibility")
        )
        payload["example"] = _clean_text(
            raw_payload.get("example") or _document_field(document, "example") or row.get("example")
        )
        payload["score"] = float(row.get("score") or 0.0)
        if "bm25_score" in row:
            payload["bm25_score"] = float(row.get("bm25_score") or 0.0)
        if "rerank_score" in row:
            payload["rerank_score"] = float(row.get("rerank_score") or 0.0)
        if "vector_score" in row:
            payload["vector_score"] = float(row.get("vector_score") or 0.0)
        payload["matched_parameter_names"] = list(row.get("matched_parameter_names") or [])
        payload["matched_aliases"] = list(row.get("matched_aliases") or [])
        payload["parameter_names"] = list(row.get("parameter_names") or [])
        if "query_text" in row:
            payload["query_text"] = str(row.get("query_text") or "")
        payload["match_sources"] = sorted(set(list(row.get("match_sources") or [])))
        return payload

    @staticmethod
    def _normalize_scores(rows: list[dict[str, Any]], *, score_key: str = "score") -> dict[str, float]:
        raw_scores = {
            str(row.get("pmid") or ""): float(row.get(score_key) or 0.0)
            for row in rows
            if str(row.get("pmid") or "").strip()
        }
        if not raw_scores:
            return {}
        values = list(raw_scores.values())
        max_score = max(values)
        min_score = min(values)
        if math.isclose(max_score, min_score):
            if max_score <= 0.0:
                return {pmid: 0.0 for pmid in raw_scores}
            return {pmid: 1.0 for pmid in raw_scores}
        denominator = max_score - min_score
        return {
            pmid: (score - min_score) / denominator
            for pmid, score in raw_scores.items()
        }

    def _build_candidate_union(
        self,
        *,
        bm25_rows: list[dict[str, Any]],
        vector_rows: list[dict[str, Any]],
        top_k: int,
        backend: str,
    ) -> list[dict[str, Any]]:
        max_candidates = max(int(top_k), 1)
        ordered: list[dict[str, Any]] = []
        by_pmid: dict[str, dict[str, Any]] = {}

        def _merge_rows(rows: list[dict[str, Any]], *, source_name: str) -> None:
            for row in list(rows):
                pmid = str(row.get("pmid") or "").strip()
                if not pmid:
                    continue

                candidate = by_pmid.get(pmid)
                if candidate is None:
                    if len(ordered) >= max_candidates:
                        continue
                    candidate = dict(row)
                    candidate["source_channels"] = []
                    candidate["source_entries"] = {}
                    by_pmid[pmid] = candidate
                    ordered.append(candidate)

                for field_name in (
                    "title",
                    "purpose",
                    "eligibility",
                    "example",
                    "taxonomy",
                    "specialty",
                ):
                    if not candidate.get(field_name):
                        candidate[field_name] = row.get(field_name)
                if not candidate.get("parameter_names"):
                    candidate["parameter_names"] = list(row.get("parameter_names") or [])
                if not candidate.get("calculator_payload") and row.get("calculator_payload"):
                    candidate["calculator_payload"] = dict(row.get("calculator_payload") or {})

                source_channels = list(candidate.get("source_channels") or [])
                if source_name not in source_channels:
                    candidate["source_channels"] = source_channels + [source_name]

                source_entries = dict(candidate.get("source_entries") or {})
                source_entries[source_name] = dict(row)
                candidate["source_entries"] = source_entries

                match_sources = set(list(candidate.get("match_sources") or []))
                match_sources.update(list(row.get("match_sources") or []))
                if source_name == "bm25":
                    match_sources.add("parameter")
                elif source_name == "vector":
                    match_sources.add("vector")
                candidate["match_sources"] = sorted(match_sources)

        if backend == "vector":
            source_batches = [("vector", list(vector_rows)[:max_candidates])]
        elif backend in {"keyword", "parameter"}:
            source_batches = [("bm25", list(bm25_rows)[:max_candidates])]
        else:
            hybrid_branch_top_k = _resolve_hybrid_branch_top_k(max_candidates)
            source_batches = [
                ("bm25", list(bm25_rows)[:hybrid_branch_top_k]),
                ("vector", list(vector_rows)[:hybrid_branch_top_k]),
            ]

        for source_name, rows in source_batches:
            _merge_rows(rows, source_name=source_name)

        return ordered[:max_candidates]

    @staticmethod
    def _build_context_top_hits(rows: list[dict[str, Any]], *, limit: int) -> dict[str, dict[str, str]]:
        context_payload: dict[str, dict[str, str]] = {}
        for row in list(rows)[: max(int(limit), 0)]:
            pmid = str(row.get("pmid") or "").strip()
            if not pmid:
                continue
            context_payload[pmid] = {
                "title": _clean_text(row.get("title")),
                "purpose": _clean_text(row.get("purpose")),
                "example": _clean_text(row.get("example")),
            }
        return context_payload

    def _build_full_context_hits(
        self,
        rows: list[dict[str, Any]],
        *,
        limit: int,
        channel: str,
    ) -> list[dict[str, Any]]:
        full_hits: list[dict[str, Any]] = []
        for rank, row in enumerate(list(rows)[: max(int(limit), 0)], start=1):
            pmid = str(row.get("pmid") or "").strip()
            if not pmid:
                continue
            full_hits.append(
                {
                    "rank": rank,
                    "channel": channel,
                    "pmid": pmid,
                    "score": float(row.get("score") or 0.0),
                    "bm25_score": float(row.get("bm25_score") or 0.0) if "bm25_score" in row else None,
                    "vector_score": (
                        float(row.get("vector_score") or row.get("score") or 0.0)
                        if "vector_score" in row or channel == "vector"
                        else None
                    ),
                    "matched_parameter_names": list(row.get("matched_parameter_names") or []),
                    "matched_aliases": list(row.get("matched_aliases") or []),
                    "parameter_names": list(row.get("parameter_names") or []),
                    "match_sources": sorted(set(list(row.get("match_sources") or []))),
                    "query_text": str(row.get("query_text") or ""),
                    "calculator_payload": self._build_full_calculator_payload(pmid),
                }
            )
        return full_hits

    def _build_full_calculator_payload(self, calc_id: str) -> dict[str, Any]:
        normalized_id = str(calc_id or "").strip()
        if not normalized_id:
            return {}

        raw_payload = dict(self._riskcalcs_payload_index.get(normalized_id) or {})
        document = None
        if self.catalog is not None:
            try:
                document = self.catalog.get(normalized_id)
            except Exception:
                document = None

        if document is not None and hasattr(document, "to_dict"):
            try:
                document_payload = dict(document.to_dict() or {})
            except Exception:
                document_payload = {}
        else:
            document_payload = {}

        payload: dict[str, Any] = {
            **document_payload,
            **raw_payload,
        }
        payload["pmid"] = normalized_id

        if document is not None:
            payload.setdefault("title", _clean_text(_document_field(document, "title")))
            payload.setdefault("purpose", _clean_text(_document_field(document, "purpose")))
            payload.setdefault("specialty", _clean_text(_document_field(document, "specialty")))
            payload.setdefault("eligibility", _clean_text(_document_field(document, "eligibility")))
            payload.setdefault("computation", _clean_text(_document_field(document, "computation")))
            payload.setdefault("interpretation", _clean_text(_document_field(document, "interpretation")))
            payload.setdefault("utility", _clean_text(_document_field(document, "utility")))
            payload.setdefault("abstract", _clean_text(_document_field(document, "abstract")))
            payload.setdefault("retrieval_text", _clean_text(_document_field(document, "retrieval_text")))
            payload.setdefault("taxonomy", dict(_document_field(document, "taxonomy") or {}))

        return payload


def create_retrieval_tool(
    catalog: Any,
    backend: str = "parameter",
    *,
    parameter_path: str | Path | None = None,
    parameter_field_weights: Mapping[str, float] | None = None,
    vector_retriever: Any | None = None,
) -> RiskCalcRetrievalTool:
    normalized_backend = _normalize_backend_name(backend, default="parameter")
    resolved_parameter_path = _resolve_catalog_parameter_payload_path(
        catalog,
        parameter_path,
    )
    parameter_cache_key = _parameter_retriever_cache_key(
        parameter_path=resolved_parameter_path,
        field_weights=parameter_field_weights,
    )
    vector_cache_token = _build_vector_retriever_cache_token(
        catalog=catalog,
        vector_retriever=vector_retriever,
    )
    tool_cache_key = (
        _catalog_runtime_cache_key(catalog),
        normalized_backend,
        parameter_cache_key,
        vector_cache_token,
    )

    with _CACHE_LOCK:
        cached = _RETRIEVAL_TOOL_CACHE.get(tool_cache_key)
    if cached is not None:
        return cached

    with _get_cache_key_lock("retrieval_tool", tool_cache_key):
        with _CACHE_LOCK:
            cached = _RETRIEVAL_TOOL_CACHE.get(tool_cache_key)
        if cached is not None:
            return cached

        parameter_retriever = _get_or_create_parameter_retriever(
            parameter_path=resolved_parameter_path,
            field_weights=parameter_field_weights,
        )
        if vector_retriever is None and normalized_backend in {"vector", "hybrid"}:
            vector_retriever = _get_or_create_inline_vector_retriever(catalog)

        tool_instance = RiskCalcRetrievalTool(
            catalog,
            parameter_retriever=parameter_retriever,
            vector_retriever=vector_retriever,
            backend=backend,
        )
        with _CACHE_LOCK:
            cached = _RETRIEVAL_TOOL_CACHE.setdefault(tool_cache_key, tool_instance)
        return cached
