from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET


DEFAULT_CHUNK_CHAR_LIMIT = 1400
DEFAULT_INPUT_PATHS = (
    Path("/home/yuanzy/MadAI/数据/totrials/corpus_2021_2022"),
    Path("/home/yuanzy/MadAI/数据/totrials/corpus_2023"),
)
DEFAULT_OUTPUT_DIR = Path("/home/yuanzy/MadAI/outputs/trial_vector_kb")
_CORPUS_PRIORITY = {
    "totrials_2021_2022": 1,
    "totrials_2023": 2,
}
_CHUNK_RANK_WEIGHTS = {
    "overview": 1.0,
    "description": 0.75,
    "eligibility_inclusion": 1.2,
    "eligibility_exclusion": 1.2,
    "outcomes": 0.9,
    "arms_interventions": 0.85,
}


@dataclass(slots=True)
class TrialXmlSource:
    xml_bytes: bytes
    source_corpus: str
    source_archive: str
    source_member_path: str
    input_path: str


def _normalize_whitespace(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_block_text(value: Any) -> str:
    raw_text = str(value or "").replace("\r", "\n")
    lines = [_normalize_whitespace(line) for line in raw_text.splitlines()]
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def _normalize_text_list(values: Iterable[Any]) -> list[str]:
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


def _find_text(root: ET.Element, path: str) -> str:
    element = root.find(path)
    if element is None:
        return ""
    return _normalize_whitespace("".join(element.itertext()))


def _find_block_text(root: ET.Element, path: str) -> str:
    element = root.find(path)
    if element is None:
        return ""
    return _normalize_block_text("".join(element.itertext()))


def _find_all_texts(root: ET.Element, path: str) -> list[str]:
    return _normalize_text_list("".join(node.itertext()) for node in root.findall(path))


def _find_attr_text(root: ET.Element, path: str, attr_name: str) -> str:
    element = root.find(path)
    if element is None:
        return ""
    return _normalize_whitespace(element.attrib.get(attr_name, ""))


def _safe_filename_fragment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip()) or "default"


def _parse_snapshot_datetime(value: str) -> datetime | None:
    text = _normalize_whitespace(value)
    if not text:
        return None
    match = re.search(r"([A-Za-z]+ \d{1,2}, \d{4})", text)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%B %d, %Y")
    except ValueError:
        return None


def _parse_partial_date(value: str) -> datetime | None:
    text = _normalize_whitespace(value)
    if not text or text.upper() == "N/A":
        return None
    for fmt in ("%B %d, %Y", "%B %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _record_sort_key(record: dict[str, Any]) -> tuple[int, float, float, str, str]:
    snapshot = _parse_snapshot_datetime(str(record.get("source_snapshot_date") or ""))
    last_update = _parse_partial_date(str(record.get("last_update_posted") or ""))
    return (
        int(_CORPUS_PRIORITY.get(str(record.get("source_corpus") or ""), 0)),
        snapshot.timestamp() if snapshot is not None else 0.0,
        last_update.timestamp() if last_update is not None else 0.0,
        str(record.get("source_archive") or ""),
        str(record.get("source_member_path") or ""),
    )


def _normalize_status(value: str) -> str:
    return _normalize_whitespace(value)


def _parse_age_to_years(value: str) -> float | None:
    text = _normalize_whitespace(value)
    if not text or text.upper() == "N/A":
        return None
    match = re.match(r"(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>[A-Za-z]+)", text)
    if not match:
        return None
    amount = float(match.group("number"))
    unit = match.group("unit").lower()
    factors = {
        "year": 1.0,
        "years": 1.0,
        "month": 1.0 / 12.0,
        "months": 1.0 / 12.0,
        "week": 1.0 / 52.0,
        "weeks": 1.0 / 52.0,
        "day": 1.0 / 365.0,
        "days": 1.0 / 365.0,
        "hour": 1.0 / (365.0 * 24.0),
        "hours": 1.0 / (365.0 * 24.0),
    }
    factor = factors.get(unit)
    if factor is None:
        return None
    return round(amount * factor, 4)


def _split_eligibility_sections(text: str) -> tuple[str, str]:
    normalized = _normalize_block_text(text)
    if not normalized:
        return "", ""

    inclusion_match = re.search(r"\bInclusion Criteria:?\b", normalized, flags=re.IGNORECASE)
    exclusion_match = re.search(r"\bExclusion Criteria:?\b", normalized, flags=re.IGNORECASE)

    if inclusion_match and exclusion_match:
        if inclusion_match.start() <= exclusion_match.start():
            inclusion = normalized[inclusion_match.end() : exclusion_match.start()].strip(" -:\n")
            exclusion = normalized[exclusion_match.end() :].strip(" -:\n")
            return (_normalize_block_text(inclusion), _normalize_block_text(exclusion))
        exclusion = normalized[exclusion_match.end() : inclusion_match.start()].strip(" -:\n")
        inclusion = normalized[inclusion_match.end() :].strip(" -:\n")
        return (_normalize_block_text(inclusion), _normalize_block_text(exclusion))

    if inclusion_match:
        inclusion = normalized[inclusion_match.end() :].strip(" -:\n")
        return (_normalize_block_text(inclusion), "")

    if exclusion_match:
        exclusion = normalized[exclusion_match.end() :].strip(" -:\n")
        return ("", _normalize_block_text(exclusion))

    return (normalized, "")


def _paragraphs(text: str) -> list[str]:
    return [paragraph.strip() for paragraph in _normalize_block_text(text).split("\n\n") if paragraph.strip()]


def _pack_paragraphs(paragraphs: list[str], *, max_chars: int) -> list[str]:
    if not paragraphs:
        return []
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for paragraph in paragraphs:
        paragraph_length = len(paragraph)
        projected = paragraph_length if not current else current_length + 2 + paragraph_length
        if current and projected > max_chars:
            chunks.append("\n\n".join(current))
            current = [paragraph]
            current_length = paragraph_length
            continue
        current.append(paragraph)
        current_length = projected if current_length else paragraph_length
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _estimate_tokens(text: str) -> int:
    return max(1, int(math.ceil(len(str(text or "")) / 4.0)))


def _build_overview_text(record: dict[str, Any]) -> str:
    lines: list[str] = []
    if record.get("display_title"):
        lines.append(f"title: {record['display_title']}")
    if record.get("acronym"):
        lines.append(f"acronym: {record['acronym']}")
    profile = [
        item
        for item in (
            record.get("study_type"),
            record.get("phase"),
            record.get("primary_purpose"),
            record.get("overall_status"),
        )
        if _normalize_whitespace(item)
    ]
    if profile:
        lines.append("study profile: " + "; ".join(str(item) for item in profile))
    if record.get("conditions"):
        lines.append("conditions: " + ", ".join(record["conditions"]))
    if record.get("condition_mesh_terms"):
        lines.append("condition mesh terms: " + ", ".join(record["condition_mesh_terms"]))
    if record.get("interventions"):
        lines.append("interventions: " + ", ".join(record["interventions"]))
    if record.get("intervention_mesh_terms"):
        lines.append("intervention mesh terms: " + ", ".join(record["intervention_mesh_terms"]))
    if record.get("brief_summary"):
        lines.append(f"summary: {record['brief_summary']}")
    return "\n".join(lines).strip()


def _chunk_context_lines(record: dict[str, Any], *, chunk_type: str) -> list[str]:
    lines = [f"title: {record.get('display_title') or record.get('nct_id') or ''}"]
    if record.get("nct_id"):
        lines.append(f"nct_id: {record['nct_id']}")
    lines.append(f"chunk type: {chunk_type}")
    if record.get("overall_status"):
        lines.append(f"overall status: {record['overall_status']}")
    if record.get("study_type") or record.get("phase") or record.get("primary_purpose"):
        profile = [
            item
            for item in (
                record.get("study_type"),
                record.get("phase"),
                record.get("primary_purpose"),
            )
            if _normalize_whitespace(item)
        ]
        if profile:
            lines.append("study profile: " + "; ".join(str(item) for item in profile))
    if record.get("conditions"):
        lines.append("conditions: " + ", ".join(record["conditions"]))
    if record.get("interventions"):
        lines.append("interventions: " + ", ".join(record["interventions"]))
    return [line for line in lines if _normalize_whitespace(line)]


def _build_contextual_chunk_text(
    record: dict[str, Any],
    *,
    chunk_type: str,
    label: str,
    body: str,
) -> str:
    lines = _chunk_context_lines(record, chunk_type=chunk_type)
    body_text = _normalize_block_text(body)
    if body_text:
        lines.append(f"{label}: {body_text}")
    return "\n".join(line for line in lines if line.strip()).strip()


def _iter_outcome_rows(record: dict[str, Any], prefix: str) -> list[str]:
    measures = list(record.get(f"{prefix}_outcome_measures") or [])
    descriptions = list(record.get(f"{prefix}_outcome_descriptions") or [])
    time_frames = list(record.get(f"{prefix}_outcome_time_frames") or [])
    rows: list[str] = []
    total = max(len(measures), len(descriptions), len(time_frames))
    for index in range(total):
        pieces: list[str] = []
        if index < len(measures) and measures[index]:
            pieces.append(f"measure={measures[index]}")
        if index < len(descriptions) and descriptions[index]:
            pieces.append(f"description={descriptions[index]}")
        if index < len(time_frames) and time_frames[index]:
            pieces.append(f"time_frame={time_frames[index]}")
        if pieces:
            rows.append("; ".join(pieces))
    return rows


def _build_outcomes_text(record: dict[str, Any]) -> str:
    lines: list[str] = []
    primary_rows = _iter_outcome_rows(record, "primary")
    secondary_rows = _iter_outcome_rows(record, "secondary")
    if primary_rows:
        lines.append("Primary outcomes:")
        lines.extend(f"- {row}" for row in primary_rows)
    if secondary_rows:
        lines.append("Secondary outcomes:")
        lines.extend(f"- {row}" for row in secondary_rows)
    if record.get("reference_citations"):
        lines.append("References:")
        lines.extend(f"- {citation}" for citation in record["reference_citations"])
    return "\n".join(lines).strip()


def _build_arms_interventions_text(record: dict[str, Any]) -> str:
    lines: list[str] = []
    interventions = list(record.get("interventions") or [])
    intervention_types = list(record.get("intervention_types") or [])
    intervention_descriptions = list(record.get("intervention_descriptions") or [])
    intervention_rows: list[str] = []
    total = max(len(interventions), len(intervention_types), len(intervention_descriptions))
    for index in range(total):
        pieces: list[str] = []
        if index < len(interventions) and interventions[index]:
            pieces.append(f"name={interventions[index]}")
        if index < len(intervention_types) and intervention_types[index]:
            pieces.append(f"type={intervention_types[index]}")
        if index < len(intervention_descriptions) and intervention_descriptions[index]:
            pieces.append(f"description={intervention_descriptions[index]}")
        if pieces:
            intervention_rows.append("; ".join(pieces))
    if intervention_rows:
        lines.append("Interventions:")
        lines.extend(f"- {row}" for row in intervention_rows)

    arm_labels = list(record.get("arm_group_labels") or [])
    arm_types = list(record.get("arm_group_types") or [])
    arm_descriptions = list(record.get("arm_group_descriptions") or [])
    arm_rows: list[str] = []
    total = max(len(arm_labels), len(arm_types), len(arm_descriptions))
    for index in range(total):
        pieces = []
        if index < len(arm_labels) and arm_labels[index]:
            pieces.append(f"label={arm_labels[index]}")
        if index < len(arm_types) and arm_types[index]:
            pieces.append(f"type={arm_types[index]}")
        if index < len(arm_descriptions) and arm_descriptions[index]:
            pieces.append(f"description={arm_descriptions[index]}")
        if pieces:
            arm_rows.append("; ".join(pieces))
    if arm_rows:
        lines.append("Arm groups:")
        lines.extend(f"- {row}" for row in arm_rows)

    return "\n".join(lines).strip()


def _detect_source_corpus(path: Path) -> str:
    normalized_parts = {part.lower() for part in path.parts}
    if "corpus_2023" in normalized_parts:
        return "totrials_2023"
    if "corpus_2021_2022" in normalized_parts:
        return "totrials_2021_2022"
    return _safe_filename_fragment(path.parent.name or path.stem)


def _build_trial_record_from_root(
    root: ET.Element,
    *,
    xml_bytes: bytes,
    source_corpus: str,
    source_archive: str,
    source_member_path: str,
) -> dict[str, Any]:
    source_snapshot_date = _find_text(root, "./required_header/download_date")
    overall_status = _find_text(root, "./overall_status")
    eligibility_text = _find_block_text(root, "./eligibility/criteria/textblock")
    eligibility_inclusion_text, eligibility_exclusion_text = _split_eligibility_sections(eligibility_text)
    conditions = _find_all_texts(root, "./condition")
    condition_mesh_terms = _find_all_texts(root, "./condition_browse/mesh_term")
    keywords = _find_all_texts(root, "./keyword")
    interventions = _find_all_texts(root, "./intervention/intervention_name")
    intervention_mesh_terms = _find_all_texts(root, "./intervention_browse/mesh_term")
    brief_title = _find_text(root, "./brief_title")
    official_title = _find_text(root, "./official_title")
    nct_id = _find_text(root, "./id_info/nct_id") or Path(source_member_path).stem
    display_title = brief_title or official_title or nct_id

    record = {
        "nct_id": nct_id,
        "source_url": _find_text(root, "./required_header/url"),
        "source_corpus": source_corpus,
        "source_archive": source_archive,
        "source_member_path": source_member_path,
        "source_snapshot_date": source_snapshot_date,
        "xml_sha256": hashlib.sha256(xml_bytes).hexdigest(),
        "brief_title": brief_title,
        "official_title": official_title,
        "acronym": _find_text(root, "./acronym"),
        "brief_summary": _find_block_text(root, "./brief_summary/textblock"),
        "detailed_description": _find_block_text(root, "./detailed_description/textblock"),
        "overall_status": overall_status,
        "study_type": _find_text(root, "./study_type"),
        "phase": _find_text(root, "./phase"),
        "primary_purpose": _find_text(root, "./study_design_info/primary_purpose"),
        "intervention_model": _find_text(root, "./study_design_info/intervention_model"),
        "allocation": _find_text(root, "./study_design_info/allocation"),
        "masking": _find_text(root, "./study_design_info/masking"),
        "enrollment": _find_text(root, "./enrollment"),
        "enrollment_type": _find_attr_text(root, "./enrollment", "type"),
        "conditions": conditions,
        "condition_mesh_terms": condition_mesh_terms,
        "keywords": keywords,
        "interventions": interventions,
        "intervention_types": _find_all_texts(root, "./intervention/intervention_type"),
        "intervention_descriptions": _find_all_texts(root, "./intervention/description"),
        "intervention_mesh_terms": intervention_mesh_terms,
        "arm_group_labels": _find_all_texts(root, "./arm_group/arm_group_label"),
        "arm_group_types": _find_all_texts(root, "./arm_group/arm_group_type"),
        "arm_group_descriptions": _find_all_texts(root, "./arm_group/description"),
        "eligibility_text": eligibility_text,
        "gender": _find_text(root, "./eligibility/gender"),
        "minimum_age": _find_text(root, "./eligibility/minimum_age"),
        "maximum_age": _find_text(root, "./eligibility/maximum_age"),
        "healthy_volunteers": _find_text(root, "./eligibility/healthy_volunteers"),
        "primary_outcome_measures": _find_all_texts(root, "./primary_outcome/measure"),
        "primary_outcome_descriptions": _find_all_texts(root, "./primary_outcome/description"),
        "primary_outcome_time_frames": _find_all_texts(root, "./primary_outcome/time_frame"),
        "secondary_outcome_measures": _find_all_texts(root, "./secondary_outcome/measure"),
        "secondary_outcome_descriptions": _find_all_texts(root, "./secondary_outcome/description"),
        "secondary_outcome_time_frames": _find_all_texts(root, "./secondary_outcome/time_frame"),
        "reference_pmids": _find_all_texts(root, "./reference/PMID"),
        "reference_citations": _find_all_texts(root, "./reference/citation"),
        "start_date": _find_text(root, "./start_date"),
        "completion_date": _find_text(root, "./completion_date"),
        "primary_completion_date": _find_text(root, "./primary_completion_date"),
        "study_first_submitted": _find_text(root, "./study_first_submitted"),
        "study_first_posted": _find_text(root, "./study_first_posted"),
        "last_update_submitted": _find_text(root, "./last_update_submitted"),
        "last_update_posted": _find_text(root, "./last_update_posted"),
        "countries": _find_all_texts(root, "./location_countries/country"),
        "facility_cities": _find_all_texts(root, "./location/facility/address/city"),
        "facility_states": _find_all_texts(root, "./location/facility/address/state"),
        "facility_countries": _find_all_texts(root, "./location/facility/address/country"),
        "display_title": display_title,
        "normalized_status": _normalize_status(overall_status),
        "condition_terms": _normalize_text_list([*conditions, *condition_mesh_terms, *keywords]),
        "intervention_terms": _normalize_text_list([*interventions, *intervention_mesh_terms]),
        "title_text": " ".join(item for item in (display_title, _find_text(root, "./acronym")) if item).strip(),
        "eligibility_inclusion_text": eligibility_inclusion_text,
        "eligibility_exclusion_text": eligibility_exclusion_text,
        "has_results_references": bool(_find_all_texts(root, "./reference/citation") or _find_all_texts(root, "./reference/PMID")),
        "age_floor_years": _parse_age_to_years(_find_text(root, "./eligibility/minimum_age")),
        "age_ceiling_years": _parse_age_to_years(_find_text(root, "./eligibility/maximum_age")),
    }
    record["overview_text"] = _build_overview_text(record)
    return record


def build_trial_record_from_xml_bytes(
    xml_bytes: bytes,
    *,
    source_corpus: str,
    source_archive: str,
    source_member_path: str,
) -> dict[str, Any]:
    root = ET.fromstring(xml_bytes)
    return _build_trial_record_from_root(
        root,
        xml_bytes=xml_bytes,
        source_corpus=source_corpus,
        source_archive=source_archive,
        source_member_path=source_member_path,
    )


def _append_chunk(
    chunks: list[dict[str, Any]],
    sequence_by_type: Counter[str],
    *,
    record: dict[str, Any],
    chunk_type: str,
    text: str,
    source_fields: list[str],
) -> None:
    normalized_text = _normalize_block_text(text)
    if not normalized_text:
        return
    sequence = int(sequence_by_type[chunk_type])
    sequence_by_type[chunk_type] += 1
    chunks.append(
        {
            "chunk_id": f"{record['nct_id']}::{chunk_type}::{sequence}",
            "nct_id": record["nct_id"],
            "chunk_type": chunk_type,
            "sequence": sequence,
            "text": normalized_text,
            "embedding_text": normalized_text,
            "source_fields": _normalize_text_list(source_fields),
            "token_estimate": _estimate_tokens(normalized_text),
            "rank_weight": float(_CHUNK_RANK_WEIGHTS.get(chunk_type, 1.0)),
        }
    )


def build_trial_chunks(
    record: dict[str, Any],
    *,
    chunk_char_limit: int = DEFAULT_CHUNK_CHAR_LIMIT,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    sequence_by_type: Counter[str] = Counter()

    overview_text = _normalize_block_text(record.get("overview_text"))
    if overview_text:
        _append_chunk(
            chunks,
            sequence_by_type,
            record=record,
            chunk_type="overview",
            text=_build_contextual_chunk_text(record, chunk_type="overview", label="overview", body=overview_text),
            source_fields=[
                "display_title",
                "acronym",
                "study_type",
                "phase",
                "primary_purpose",
                "overall_status",
                "conditions",
                "condition_mesh_terms",
                "interventions",
                "brief_summary",
            ],
        )

    description_paragraphs = _paragraphs(str(record.get("detailed_description") or ""))
    for chunk_body in _pack_paragraphs(description_paragraphs, max_chars=chunk_char_limit):
        _append_chunk(
            chunks,
            sequence_by_type,
            record=record,
            chunk_type="description",
            text=_build_contextual_chunk_text(
                record,
                chunk_type="description",
                label="detailed description",
                body=chunk_body,
            ),
            source_fields=["detailed_description"],
        )

    eligibility_sections = (
        ("eligibility_inclusion", str(record.get("eligibility_inclusion_text") or "")),
        ("eligibility_exclusion", str(record.get("eligibility_exclusion_text") or "")),
    )
    for chunk_type, body in eligibility_sections:
        section_paragraphs = _paragraphs(body)
        for chunk_body in _pack_paragraphs(section_paragraphs, max_chars=chunk_char_limit):
            label = "inclusion criteria" if chunk_type == "eligibility_inclusion" else "exclusion criteria"
            source_fields = (
                ["eligibility_text", "eligibility_inclusion_text"]
                if chunk_type == "eligibility_inclusion"
                else ["eligibility_text", "eligibility_exclusion_text"]
            )
            _append_chunk(
                chunks,
                sequence_by_type,
                record=record,
                chunk_type=chunk_type,
                text=_build_contextual_chunk_text(record, chunk_type=chunk_type, label=label, body=chunk_body),
                source_fields=source_fields,
            )

    outcomes_text = _build_outcomes_text(record)
    if outcomes_text:
        for chunk_body in _pack_paragraphs(_paragraphs(outcomes_text), max_chars=chunk_char_limit):
            _append_chunk(
                chunks,
                sequence_by_type,
                record=record,
                chunk_type="outcomes",
                text=_build_contextual_chunk_text(
                    record,
                    chunk_type="outcomes",
                    label="outcomes and references",
                    body=chunk_body,
                ),
                source_fields=[
                    "primary_outcome_measures",
                    "primary_outcome_descriptions",
                    "primary_outcome_time_frames",
                    "secondary_outcome_measures",
                    "secondary_outcome_descriptions",
                    "secondary_outcome_time_frames",
                    "reference_citations",
                    "reference_pmids",
                ],
            )

    arms_interventions_text = _build_arms_interventions_text(record)
    if arms_interventions_text:
        for chunk_body in _pack_paragraphs(_paragraphs(arms_interventions_text), max_chars=chunk_char_limit):
            _append_chunk(
                chunks,
                sequence_by_type,
                record=record,
                chunk_type="arms_interventions",
                text=_build_contextual_chunk_text(
                    record,
                    chunk_type="arms_interventions",
                    label="arms and interventions",
                    body=chunk_body,
                ),
                source_fields=[
                    "interventions",
                    "intervention_types",
                    "intervention_descriptions",
                    "intervention_mesh_terms",
                    "arm_group_labels",
                    "arm_group_types",
                    "arm_group_descriptions",
                ],
            )

    return chunks


def _iter_zip_sources(path: Path) -> Iterator[TrialXmlSource]:
    source_corpus = _detect_source_corpus(path)
    with zipfile.ZipFile(path) as archive:
        for member_name in sorted(archive.namelist()):
            if not member_name.lower().endswith(".xml"):
                continue
            yield TrialXmlSource(
                xml_bytes=archive.read(member_name),
                source_corpus=source_corpus,
                source_archive=path.name,
                source_member_path=member_name,
                input_path=str(path),
            )


def _iter_directory_sources(path: Path) -> Iterator[TrialXmlSource]:
    for zip_path in sorted(item for item in path.rglob("*.zip") if item.is_file()):
        yield from _iter_zip_sources(zip_path)

    source_corpus = _detect_source_corpus(path)
    for xml_path in sorted(item for item in path.rglob("*.xml") if item.is_file()):
        relative_path = str(xml_path.relative_to(path)) if xml_path.is_relative_to(path) else xml_path.name
        yield TrialXmlSource(
            xml_bytes=xml_path.read_bytes(),
            source_corpus=source_corpus,
            source_archive="",
            source_member_path=relative_path,
            input_path=str(xml_path),
        )


def iter_trial_xml_sources(input_paths: Sequence[str | Path]) -> Iterator[TrialXmlSource]:
    for raw_path in list(input_paths or []):
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            continue
        if path.is_file():
            if path.suffix.lower() == ".zip":
                yield from _iter_zip_sources(path)
            elif path.suffix.lower() == ".xml":
                yield TrialXmlSource(
                    xml_bytes=path.read_bytes(),
                    source_corpus=_detect_source_corpus(path),
                    source_archive="",
                    source_member_path=path.name,
                    input_path=str(path),
                )
            continue
        if path.is_dir():
            yield from _iter_directory_sources(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def build_trial_vector_kb(
    *,
    input_paths: Sequence[str | Path] | None = None,
    output_dir: str | Path | None = None,
    limit: int = 0,
    chunk_char_limit: int = DEFAULT_CHUNK_CHAR_LIMIT,
) -> dict[str, Any]:
    resolved_inputs = [Path(path).expanduser().resolve() for path in list(input_paths or DEFAULT_INPUT_PATHS)]
    resolved_output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    selected_records: dict[str, dict[str, Any]] = {}
    source_documents = 0
    parse_errors: list[dict[str, str]] = []
    duplicate_candidates = 0

    for source in iter_trial_xml_sources(resolved_inputs):
        source_documents += 1
        if limit > 0 and source_documents > int(limit):
            break
        try:
            record = build_trial_record_from_xml_bytes(
                source.xml_bytes,
                source_corpus=source.source_corpus,
                source_archive=source.source_archive,
                source_member_path=source.source_member_path,
            )
        except Exception as exc:
            parse_errors.append(
                {
                    "input_path": source.input_path,
                    "source_member_path": source.source_member_path,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        nct_id = _normalize_whitespace(record.get("nct_id"))
        if not nct_id:
            parse_errors.append(
                {
                    "input_path": source.input_path,
                    "source_member_path": source.source_member_path,
                    "error": "Missing nct_id",
                }
            )
            continue

        existing = selected_records.get(nct_id)
        if existing is None:
            selected_records[nct_id] = record
            continue

        duplicate_candidates += 1
        if _record_sort_key(record) >= _record_sort_key(existing):
            selected_records[nct_id] = record

    trial_records = [selected_records[nct_id] for nct_id in sorted(selected_records)]
    trial_chunks: list[dict[str, Any]] = []
    chunk_type_counter: Counter[str] = Counter()
    source_corpus_counter: Counter[str] = Counter()
    status_counter: Counter[str] = Counter()

    for record in trial_records:
        source_corpus_counter[str(record.get("source_corpus") or "")] += 1
        status_counter[str(record.get("normalized_status") or "")] += 1
        chunks = build_trial_chunks(record, chunk_char_limit=chunk_char_limit)
        for chunk in chunks:
            chunk_type_counter[str(chunk.get("chunk_type") or "")] += 1
        trial_chunks.extend(chunks)

    trial_record_path = resolved_output_dir / "trial_record.jsonl"
    trial_chunk_path = resolved_output_dir / "trial_chunk.jsonl"
    manifest_path = resolved_output_dir / "manifest.json"

    write_jsonl(trial_record_path, trial_records)
    write_jsonl(trial_chunk_path, trial_chunks)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_paths": [str(path) for path in resolved_inputs],
        "output_dir": str(resolved_output_dir),
        "trial_record_filename": trial_record_path.name,
        "trial_chunk_filename": trial_chunk_path.name,
        "source_document_count": source_documents if limit <= 0 else min(source_documents, int(limit)),
        "trial_record_count": len(trial_records),
        "trial_chunk_count": len(trial_chunks),
        "chunk_type_distribution": dict(sorted(chunk_type_counter.items())),
        "selected_source_corpus_distribution": dict(sorted(source_corpus_counter.items())),
        "normalized_status_distribution": dict(sorted(status_counter.items())),
        "duplicate_candidate_count": duplicate_candidates,
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors[:50],
        "chunk_char_limit": int(chunk_char_limit),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


__all__ = [
    "DEFAULT_CHUNK_CHAR_LIMIT",
    "DEFAULT_INPUT_PATHS",
    "DEFAULT_OUTPUT_DIR",
    "TrialXmlSource",
    "build_trial_chunks",
    "build_trial_record_from_xml_bytes",
    "build_trial_vector_kb",
    "iter_trial_xml_sources",
    "write_jsonl",
]
