from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR in sys.path:
    sys.path.remove(PROJECT_ROOT_STR)
sys.path.insert(0, PROJECT_ROOT_STR)


from agent.config.env import load_dotenv_if_present  # noqa: E402
from agent.protocol.pipeline import assess_trial_eligibility_candidates  # noqa: E402
from agent.retrieval.trial_chunks import resolve_trial_vector_output_root  # noqa: E402
from agent.tools.trial_vector_retrieval_tools import (  # noqa: E402
    create_trial_chunk_retrieval_tool,
)


DEFAULT_CASE_FILE = PROJECT_ROOT / "数据" / "虚拟病例_trial_af78.txt"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "outputs" / "trial_retrieval_probe.json"
DEFAULT_PART1_KB_ROOT = PROJECT_ROOT / "vector_stores" / "trials" / "part1_caseprobe" / "kb"
DEFAULT_PART1_QDRANT_PATH = PROJECT_ROOT / "vector_stores" / "trials" / "part1_caseprobe" / "qdrant_embedded"
DEFAULT_PART1_COLLECTION = "trial_chunks_medcpt_part1_caseprobe"
DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _resolve_optional_path(raw_path: str | None) -> str | None:
    text = _normalize_text(raw_path)
    if not text:
        return None
    return str(Path(text).expanduser().resolve())


def _read_case_text(args: argparse.Namespace) -> tuple[str, str]:
    if _normalize_text(args.case_text):
        return str(args.case_text), "argument:--case-text"

    case_file = Path(args.case_file).expanduser().resolve()
    if not case_file.exists():
        raise FileNotFoundError(f"case file not found: {case_file}")
    return case_file.read_text(encoding="utf-8"), str(case_file)


def _load_structured_case(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if _normalize_text(args.structured_case_json):
        return json.loads(str(args.structured_case_json)), "argument:--structured-case-json"

    if _normalize_text(args.structured_case_file):
        path = Path(args.structured_case_file).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"structured case file not found: {path}")
        return json.loads(path.read_text(encoding="utf-8")), str(path)

    case_text, case_source = _read_case_text(args)
    return (
        {
            "raw_text": case_text,
            "case_summary": case_text[:1000],
            "problem_list": [],
            "known_facts": [],
            "structured_inputs": {},
        },
        case_source,
    )


def _brief_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "nct_id": _normalize_text(candidate.get("nct_id")),
        "title": _normalize_text(candidate.get("title") or candidate.get("display_title")),
        "score": float(candidate.get("score") or 0.0),
        "overall_status": _normalize_text(candidate.get("overall_status")),
        "status": _normalize_text(candidate.get("status")),
        "enrollment_open": bool(candidate.get("enrollment_open")),
        "conditions": list(candidate.get("conditions") or [])[:5],
        "interventions": list(candidate.get("interventions") or [])[:5],
        "matched_condition_terms": list(candidate.get("matched_condition_terms") or []),
        "matched_intervention_terms": list(candidate.get("matched_intervention_terms") or []),
        "eligibility_conflicts": list(candidate.get("eligibility_conflicts") or []),
        "eligibility_signals": list(candidate.get("eligibility_signals") or []),
        "matched_chunks": list(candidate.get("matched_chunks") or [])[:3],
    }


def _print_summary(result: dict[str, Any], *, candidate_limit: int = 8) -> None:
    trial_bundle = dict(result.get("trial_retrieval_bundle") or {})
    query_profile = dict(trial_bundle.get("query_profile") or {})
    candidates = list(trial_bundle.get("candidate_ranking") or [])
    eligibility_bundle = dict(result.get("eligibility_assessment_bundle") or {})

    print("Case source:", result.get("case_source", ""))
    print("Trial output root:", result.get("trial_config", {}).get("trial_output_root", ""))
    print("Trial vector store:", result.get("trial_config", {}).get("trial_vector_store", ""))
    print("Backend used:", trial_bundle.get("backend_used", ""))
    print("Query text:")
    print(_normalize_text(trial_bundle.get("query_text")))
    print("Focus terms:", ", ".join(list(query_profile.get("focus_terms") or [])[:20]))
    print("Trial condition terms:", ", ".join(list(query_profile.get("trial_condition_terms") or [])))
    print("Trial intervention terms:", ", ".join(list(query_profile.get("trial_intervention_terms") or [])))
    print("Trial intent terms:", ", ".join(list(query_profile.get("trial_intent_terms") or [])))
    print("Coarse candidate count:", len(list(trial_bundle.get("coarse_candidate_ids") or [])))
    print("Fine candidate count:", len(candidates))
    print()

    print(f"Top {min(candidate_limit, len(candidates))} candidates:")
    for index, candidate in enumerate(candidates[: max(int(candidate_limit), 1)], start=1):
        brief = _brief_candidate(dict(candidate))
        print(
            f"{index}. {brief['nct_id']} | score={brief['score']:.3f} | "
            f"{brief['overall_status'] or brief['status']} | {brief['title']}"
        )
        if brief["conditions"]:
            print("   conditions:", "; ".join(str(item) for item in brief["conditions"]))
        if brief["interventions"]:
            print("   interventions:", "; ".join(str(item) for item in brief["interventions"]))
        if brief["matched_condition_terms"] or brief["matched_intervention_terms"]:
            print(
                "   matched:",
                "; ".join(
                    [
                        *[f"condition={item}" for item in brief["matched_condition_terms"]],
                        *[f"intervention={item}" for item in brief["matched_intervention_terms"]],
                    ]
                ),
            )
        if brief["eligibility_conflicts"]:
            print("   conflicts:", "; ".join(str(item) for item in brief["eligibility_conflicts"][:3]))
        if brief["eligibility_signals"]:
            print("   signals:", "; ".join(str(item) for item in brief["eligibility_signals"][:3]))

    assessed_trials = list(eligibility_bundle.get("assessed_trials") or [])
    if assessed_trials:
        print()
        print(f"Eligibility assessed trials: {len(assessed_trials)}")
        for trial in assessed_trials:
            print(
                f"- {trial.get('nct_id', '')} | {trial.get('aggregate_status', '')} | "
                f"{trial.get('aggregate_reason', '')}"
            )
            blocking = list(trial.get("blocking_criteria") or [])
            unknown = list(trial.get("unknown_criteria") or [])
            print(f"  criteria={len(list(trial.get('criteria') or []))} blocking={len(blocking)} unknown={len(unknown)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe MedAI trial retrieval without running the full workflow."
    )
    parser.add_argument("--case-text", default="", help="Raw case text to use directly.")
    parser.add_argument(
        "--case-file",
        default=str(DEFAULT_CASE_FILE),
        help="Raw case text file. Ignored when --case-text or structured-case input is provided.",
    )
    parser.add_argument("--structured-case-json", default="", help="Structured case JSON object.")
    parser.add_argument("--structured-case-file", default="", help="Path to a structured case JSON file.")
    parser.add_argument(
        "--trial-output-root",
        default=str(DEFAULT_PART1_KB_ROOT if DEFAULT_PART1_KB_ROOT.exists() else resolve_trial_vector_output_root()),
        help="Trial chunk KB root containing trial_record.jsonl and trial_chunk.jsonl.",
    )
    parser.add_argument("--backend", choices=("bm25", "keyword", "vector", "hybrid"), default="hybrid")
    parser.add_argument("--top-k", type=int, default=10, help="Final trial-level candidates to keep.")
    parser.add_argument("--coarse-top-k", type=int, default=100, help="Trial IDs kept after coarse recall.")
    parser.add_argument("--chunk-top-k", type=int, default=80, help="Chunks retrieved during fine recall.")
    parser.add_argument("--eligibility-limit", type=int, default=3, help="Top trials to run criteria assessment on.")
    parser.add_argument("--trial-vector-store", choices=("auto", "faiss", "qdrant", "qdrant_local"), default="qdrant")
    parser.add_argument(
        "--trial-qdrant-collection",
        default=DEFAULT_PART1_COLLECTION,
        help="Qdrant collection name.",
    )
    parser.add_argument(
        "--trial-qdrant-url",
        default=DEFAULT_QDRANT_URL,
        help="Qdrant server URL. Overrides embedded path if set.",
    )
    parser.add_argument(
        "--trial-qdrant-path",
        default="" if DEFAULT_QDRANT_URL else str(DEFAULT_PART1_QDRANT_PATH) if DEFAULT_PART1_QDRANT_PATH.exists() else "",
        help="Embedded Qdrant path. Leave empty to use URL/env/defaults.",
    )
    parser.add_argument("--trial-qdrant-api-key", default="", help="Optional Qdrant API key.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_FILE), help="Full result JSON path.")
    parser.add_argument("--summary-limit", type=int, default=8, help="How many ranked candidates to print.")
    parser.add_argument("--show-json", action="store_true", help="Print the full JSON result.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv_if_present(PROJECT_ROOT / ".env")

    structured_case, case_source = _load_structured_case(args)
    trial_output_root = str(resolve_trial_vector_output_root(args.trial_output_root or None))
    trial_qdrant_path = _resolve_optional_path(args.trial_qdrant_path)
    trial_qdrant_url = _normalize_text(args.trial_qdrant_url) or None
    if trial_qdrant_url:
        trial_qdrant_path = None

    trial_retriever = create_trial_chunk_retrieval_tool(
        output_root=trial_output_root,
        backend=args.backend,
        vector_store=args.trial_vector_store,
        qdrant_collection_name=args.trial_qdrant_collection or None,
        qdrant_url=trial_qdrant_url,
        qdrant_api_key=args.trial_qdrant_api_key or None,
        qdrant_path=trial_qdrant_path,
    )

    trial_bundle = trial_retriever.retrieve_from_structured_case(
        structured_case,
        top_k=max(int(args.top_k), 1),
        coarse_top_k=max(int(args.coarse_top_k), 1),
        chunk_top_k=max(int(args.chunk_top_k), 1),
        backend=args.backend,
    )
    eligibility_bundle = assess_trial_eligibility_candidates(
        structured_case=structured_case,
        calculation_results=[],
        calculator_matches=[],
        calculator_evidence_bundle=None,
        trial_bundle=trial_bundle,
        trial_retriever=trial_retriever,
        limit=max(int(args.eligibility_limit), 0),
    )

    result = {
        "case_source": case_source,
        "structured_case": structured_case,
        "trial_config": {
            "trial_output_root": trial_output_root,
            "trial_vector_store": args.trial_vector_store,
            "trial_qdrant_collection": args.trial_qdrant_collection or "",
            "trial_qdrant_url": trial_qdrant_url or "",
            "trial_qdrant_path": trial_qdrant_path or "",
            "backend": args.backend,
            "top_k": max(int(args.top_k), 1),
            "coarse_top_k": max(int(args.coarse_top_k), 1),
            "chunk_top_k": max(int(args.chunk_top_k), 1),
            "eligibility_limit": max(int(args.eligibility_limit), 0),
        },
        "trial_retrieval_bundle": trial_bundle,
        "eligibility_assessment_bundle": eligibility_bundle,
    }

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.show_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_summary(result, candidate_limit=max(int(args.summary_limit), 1))
    print("Result JSON:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
