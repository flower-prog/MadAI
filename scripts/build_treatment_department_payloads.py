#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET


OPEN_TRIAL_STATUSES = {
    "Recruiting",
    "Not yet recruiting",
    "Enrolling by invitation",
}
EVIDENCE_TRIAL_STATUSES = {
    "Active, not recruiting",
    "Completed",
}
ABANDONED_TRIAL_STATUSES = {
    "Terminated",
    "Withdrawn",
    "Suspended",
    "No longer available",
    "Temporarily not available",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build department-classified treatment trial payloads and align each trial with "
            "MedAI treatment recommendation statuses."
        )
    )
    parser.add_argument(
        "--package-dir",
        default="数据/治疗方案",
        help="Path to the treatment package directory.",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    text = str(value or "")
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


def unique_non_empty(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = clean_text(value)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def find_text(root: ET.Element, path: str) -> str:
    node = root.find(path)
    if node is None:
        return ""
    return clean_text("".join(node.itertext()))


def find_all_texts(root: ET.Element, path: str) -> list[str]:
    return unique_non_empty(["".join(node.itertext()) for node in root.findall(path)])


def parse_index_tsv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        header_line = handle.readline().rstrip("\n")
        if not header_line:
            return rows
        headers = header_line.split("\t")
        for line in handle:
            values = line.rstrip("\n").split("\t")
            row = dict(zip(headers, values))
            secondary_departments = [
                item.strip()
                for item in str(row.get("secondary_departments", "")).split(",")
                if item.strip()
            ]
            rows.append(
                {
                    "sample_index": int(str(row.get("sample_index", "0") or "0")),
                    "nct_id": str(row.get("nct_id", "")).strip(),
                    "primary_department": str(row.get("primary_department", "")).strip(),
                    "secondary_departments": secondary_departments,
                    "copied_xml_relpath": str(row.get("copied_xml_relpath", "")).strip(),
                    "source_xml_path": str(row.get("source_xml_path", "")).strip(),
                }
            )
    return rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_tsv(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(headers) + "\n")
        for row in rows:
            handle.write(
                "\t".join(str(row.get(header, "")) for header in headers) + "\n"
            )


def parse_trial_xml(xml_path: Path) -> dict[str, Any]:
    root = ET.parse(xml_path).getroot()
    return {
        "nct_id": find_text(root, "./id_info/nct_id") or xml_path.stem,
        "brief_title": find_text(root, "./brief_title"),
        "official_title": find_text(root, "./official_title"),
        "conditions": find_all_texts(root, "./condition"),
        "mesh_terms": find_all_texts(root, "./condition_browse/mesh_term"),
        "keywords": find_all_texts(root, "./keyword"),
        "interventions": find_all_texts(root, "./intervention/intervention_name"),
        "brief_summary": find_text(root, "./brief_summary/textblock"),
        "eligibility_text": find_text(root, "./eligibility/criteria/textblock"),
        "overall_status": find_text(root, "./overall_status"),
        "study_type": find_text(root, "./study_type"),
        "phase": find_text(root, "./phase"),
        "primary_purpose": find_text(root, "./study_design_info/primary_purpose"),
    }


def map_trial_status(overall_status: str) -> dict[str, Any]:
    normalized = clean_text(overall_status)
    strategy = "trial_match"
    enrollment_open = normalized in OPEN_TRIAL_STATUSES

    if normalized in OPEN_TRIAL_STATUSES:
        return {
            "strategy": strategy,
            "status": "trial_matched",
            "enrollment_open": True,
            "status_reason": (
                "The trial is actively open or preparing to open enrollment, so MedAI can surface it as a "
                "direct trial candidate."
            ),
            "actions": [
                "Review detailed inclusion and exclusion criteria against the current case.",
                "Confirm site-level enrollment availability before surfacing this as a live trial option.",
            ],
        }

    if normalized in EVIDENCE_TRIAL_STATUSES:
        return {
            "strategy": strategy,
            "status": "trial_matched",
            "enrollment_open": False,
            "status_reason": (
                "The trial is not currently open for enrollment, but it still provides a concrete trial-backed "
                "treatment path or protocol evidence that MedAI can treat as a matched trial reference."
            ),
            "actions": [
                "Use the trial as protocol or evidence support instead of assuming active enrollment.",
                "If a current trial option is needed, search for an active successor or related study before surfacing.",
            ],
        }

    if normalized in ABANDONED_TRIAL_STATUSES:
        return {
            "strategy": strategy,
            "status": "abandoned",
            "enrollment_open": False,
            "status_reason": (
                "The trial lifecycle indicates that it is halted or unavailable, so MedAI should not treat it as a "
                "current treatment-trial option."
            ),
            "actions": [
                "Do not surface this trial as a direct option for the current patient.",
                "Keep it only as historical context if the protocol remains clinically informative.",
            ],
        }

    if normalized == "Unknown status":
        return {
            "strategy": strategy,
            "status": "manual_review",
            "enrollment_open": False,
            "status_reason": (
                "The trial lifecycle is unclear, so MedAI should require manual review before using it in "
                "protocol recommendations."
            ),
            "actions": [
                "Review the full XML and the current ClinicalTrials.gov record manually before use.",
                "Do not promote this trial to a direct recommendation until the lifecycle status is confirmed.",
            ],
        }

    return {
        "strategy": strategy,
        "status": "manual_review",
        "enrollment_open": enrollment_open,
        "status_reason": (
            "The trial status does not map cleanly to a stable MedAI recommendation state, so it should remain "
            "under manual review."
        ),
        "actions": [
            "Review the trial lifecycle manually before using it in MedAI.",
            "Document any case-specific rationale before surfacing this trial to clinicians.",
        ],
    }


def build_department_payload_entry(
    *,
    index_row: dict[str, Any],
    trial_meta: dict[str, Any],
    status_meta: dict[str, Any],
    department_tag: str,
    department_role: str,
) -> dict[str, Any]:
    display_title = (
        trial_meta.get("brief_title")
        or trial_meta.get("official_title")
        or index_row["nct_id"]
    )
    return {
        "name": display_title,
        "strategy": status_meta["strategy"],
        "source": "clinicaltrials.gov",
        "status": status_meta["status"],
        "rationale": (
            f"ClinicalTrials.gov overall_status={trial_meta['overall_status'] or 'missing'}; "
            f"aligned to MedAI status={status_meta['status']} for department tag {department_tag}."
        ),
        "linked_calculators": [],
        "linked_trials": [index_row["nct_id"]],
        "actions": list(status_meta["actions"]),
        "nct_id": index_row["nct_id"],
        "department_tag": department_tag,
        "department_role": department_role,
        "department_tags": [index_row["primary_department"], *index_row["secondary_departments"]],
        "primary_department": index_row["primary_department"],
        "secondary_departments": list(index_row["secondary_departments"]),
        "overall_status": trial_meta["overall_status"],
        "status_reason": status_meta["status_reason"],
        "enrollment_open": bool(status_meta["enrollment_open"]),
        "brief_title": trial_meta["brief_title"],
        "official_title": trial_meta["official_title"],
        "conditions": list(trial_meta["conditions"]),
        "mesh_terms": list(trial_meta["mesh_terms"]),
        "keywords": list(trial_meta["keywords"]),
        "interventions": list(trial_meta["interventions"]),
        "brief_summary": trial_meta["brief_summary"],
        "eligibility_text": trial_meta["eligibility_text"],
        "study_type": trial_meta["study_type"],
        "phase": trial_meta["phase"],
        "primary_purpose": trial_meta["primary_purpose"],
        "copied_xml_relpath": index_row["copied_xml_relpath"],
        "source_xml_path": index_row["source_xml_path"],
    }


def update_readme(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# MedAI Smoke Package",
                "",
                "- xml/ contains the copied raw ClinicalTrials.gov XML files.",
                "- medai_smoke.jsonl now includes department tags plus MedAI-aligned trial strategy/status fields.",
                "- medai_index.tsv is the one-row-per-trial index.",
                "- department_payload_index.tsv expands one trial into one or more department-tag payload entries.",
                "- <department>/treatment_trials.json mirrors the calculator-style department split under MedAI.",
                "",
                "MedAI status alignment:",
                "",
                "1. Open enrollment trial statuses map to status=trial_matched.",
                "2. Completed or active-not-recruiting trials stay status=trial_matched as trial-backed evidence, but actions warn that enrollment is not assumed.",
                "3. Unknown status maps to status=manual_review.",
                "4. Terminated, withdrawn, suspended, and similarly unavailable trials map to status=abandoned.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def build_package(package_dir: Path) -> dict[str, Any]:
    index_path = package_dir / "medai_index.tsv"
    jsonl_path = package_dir / "medai_smoke.jsonl"
    manifest_path = package_dir / "manifest.json"
    readme_path = package_dir / "README.md"

    if not index_path.exists():
        raise SystemExit(f"Index file does not exist: {index_path}")

    index_rows = parse_index_tsv(index_path)
    existing_jsonl_rows = read_jsonl(jsonl_path)
    existing_jsonl_by_nct = {
        str(row.get("nct_id", "")).strip(): row
        for row in existing_jsonl_rows
        if str(row.get("nct_id", "")).strip()
    }

    enriched_jsonl_rows: list[dict[str, Any]] = []
    enriched_index_rows: list[dict[str, Any]] = []
    department_payload_index_rows: list[dict[str, Any]] = []
    department_payloads: dict[str, dict[str, Any]] = defaultdict(dict)
    overall_status_counter: Counter[str] = Counter()
    medai_status_counter: Counter[str] = Counter()
    department_payload_counter: Counter[str] = Counter()
    department_status_counter: dict[str, Counter[str]] = defaultdict(Counter)

    for index_row in index_rows:
        copied_xml_path = package_dir / index_row["copied_xml_relpath"]
        if not copied_xml_path.exists():
            raise SystemExit(f"Missing copied XML file: {copied_xml_path}")

        trial_meta = parse_trial_xml(copied_xml_path)
        status_meta = map_trial_status(str(trial_meta.get("overall_status", "")))
        department_tags = unique_non_empty(
            [index_row["primary_department"], *index_row["secondary_departments"]]
        )

        overall_status_counter[trial_meta["overall_status"] or ""] += 1
        medai_status_counter[status_meta["status"]] += 1

        enriched_index_rows.append(
            {
                "sample_index": index_row["sample_index"],
                "nct_id": index_row["nct_id"],
                "primary_department": index_row["primary_department"],
                "secondary_departments": ",".join(index_row["secondary_departments"]),
                "department_tags": ",".join(department_tags),
                "overall_status": trial_meta["overall_status"],
                "medai_strategy": status_meta["strategy"],
                "medai_status": status_meta["status"],
                "copied_xml_relpath": index_row["copied_xml_relpath"],
                "source_xml_path": index_row["source_xml_path"],
            }
        )

        existing_row = dict(existing_jsonl_by_nct.get(index_row["nct_id"], {}))
        if not existing_row:
            existing_row = {
                "sample_index": index_row["sample_index"],
                "nct_id": index_row["nct_id"],
                "primary_department": index_row["primary_department"],
                "secondary_departments": list(index_row["secondary_departments"]),
                "copied_xml_relpath": index_row["copied_xml_relpath"],
                "source_xml_path": index_row["source_xml_path"],
                "xml_content": copied_xml_path.read_text(encoding="utf-8", errors="replace"),
                "xml_size_bytes": copied_xml_path.stat().st_size,
            }

        existing_row["department_tags"] = department_tags
        existing_row["overall_status"] = trial_meta["overall_status"]
        existing_row["medai_strategy"] = status_meta["strategy"]
        existing_row["medai_status"] = status_meta["status"]
        existing_row["medai_status_reason"] = status_meta["status_reason"]
        existing_row["enrollment_open"] = bool(status_meta["enrollment_open"])
        existing_row["brief_title"] = trial_meta["brief_title"]
        existing_row["official_title"] = trial_meta["official_title"]
        existing_row["conditions"] = list(trial_meta["conditions"])
        existing_row["mesh_terms"] = list(trial_meta["mesh_terms"])
        existing_row["keywords"] = list(trial_meta["keywords"])
        existing_row["interventions"] = list(trial_meta["interventions"])
        existing_row["brief_summary"] = trial_meta["brief_summary"]
        existing_row["eligibility_text"] = trial_meta["eligibility_text"]
        existing_row["study_type"] = trial_meta["study_type"]
        existing_row["phase"] = trial_meta["phase"]
        existing_row["primary_purpose"] = trial_meta["primary_purpose"]
        enriched_jsonl_rows.append(existing_row)

        for department_tag in department_tags:
            department_role = (
                "primary"
                if department_tag == index_row["primary_department"]
                else "secondary"
            )
            payload_entry = build_department_payload_entry(
                index_row=index_row,
                trial_meta=trial_meta,
                status_meta=status_meta,
                department_tag=department_tag,
                department_role=department_role,
            )
            department_payloads[department_tag][index_row["nct_id"]] = payload_entry
            department_payload_counter[department_tag] += 1
            department_status_counter[department_tag][status_meta["status"]] += 1
            department_payload_index_rows.append(
                {
                    "department_tag": department_tag,
                    "department_role": department_role,
                    "nct_id": index_row["nct_id"],
                    "medai_status": status_meta["status"],
                    "overall_status": trial_meta["overall_status"],
                    "payload_relpath": f"{department_tag}/treatment_trials.json",
                    "copied_xml_relpath": index_row["copied_xml_relpath"],
                }
            )

    enriched_jsonl_rows.sort(
        key=lambda row: (
            int(str(row.get("sample_index", "0") or "0")),
            str(row.get("nct_id", "")),
        )
    )
    enriched_index_rows.sort(
        key=lambda row: (
            int(str(row.get("sample_index", "0") or "0")),
            str(row.get("nct_id", "")),
        )
    )
    department_payload_index_rows.sort(
        key=lambda row: (
            str(row["department_tag"]),
            str(row["department_role"]),
            str(row["nct_id"]),
        )
    )

    write_jsonl(jsonl_path, enriched_jsonl_rows)
    write_tsv(
        index_path,
        [
            "sample_index",
            "nct_id",
            "primary_department",
            "secondary_departments",
            "department_tags",
            "overall_status",
            "medai_strategy",
            "medai_status",
            "copied_xml_relpath",
            "source_xml_path",
        ],
        enriched_index_rows,
    )
    write_tsv(
        package_dir / "department_payload_index.tsv",
        [
            "department_tag",
            "department_role",
            "nct_id",
            "medai_status",
            "overall_status",
            "payload_relpath",
            "copied_xml_relpath",
        ],
        department_payload_index_rows,
    )

    for department_tag, payload in department_payloads.items():
        department_dir = package_dir / department_tag
        department_dir.mkdir(parents=True, exist_ok=True)
        (department_dir / "treatment_trials.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["sample_size"] = len(enriched_index_rows)
    manifest["overall_status_distribution"] = dict(sorted(overall_status_counter.items()))
    manifest["medai_status_distribution"] = dict(sorted(medai_status_counter.items()))
    manifest["department_payload_distribution"] = dict(sorted(department_payload_counter.items()))
    manifest["department_status_distribution"] = {
        department: dict(sorted(counter.items()))
        for department, counter in sorted(department_status_counter.items())
    }
    manifest["department_payload_filename"] = "treatment_trials.json"
    manifest["medai_status_mapping"] = {
        "open_or_preopen": {
            "trial_statuses": sorted(OPEN_TRIAL_STATUSES),
            "medai_status": "trial_matched",
        },
        "evidence_ready": {
            "trial_statuses": sorted(EVIDENCE_TRIAL_STATUSES),
            "medai_status": "trial_matched",
        },
        "abandoned": {
            "trial_statuses": sorted(ABANDONED_TRIAL_STATUSES),
            "medai_status": "abandoned",
        },
        "manual_review": {
            "trial_statuses": ["Unknown status"],
            "medai_status": "manual_review",
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    update_readme(readme_path)

    return {
        "package_dir": str(package_dir),
        "sample_size": len(enriched_index_rows),
        "overall_status_distribution": dict(sorted(overall_status_counter.items())),
        "medai_status_distribution": dict(sorted(medai_status_counter.items())),
        "department_payload_distribution": dict(sorted(department_payload_counter.items())),
    }


def main() -> int:
    args = parse_args()
    package_dir = Path(args.package_dir).resolve()
    if not package_dir.exists():
        raise SystemExit(f"Package directory does not exist: {package_dir}")
    result = build_package(package_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
