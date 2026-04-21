from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR in sys.path:
    sys.path.remove(PROJECT_ROOT_STR)
sys.path.insert(0, PROJECT_ROOT_STR)


from agent.config.env import load_dotenv_if_present
from agent.retrieval.trial_chunks import resolve_trial_vector_output_root
from agent.tools.trial_vector_retrieval_tools import create_trial_chunk_retrieval_tool
from agent.workflow import run_workflow
from scripts.try_single_case_workflow import (
    build_summary,
    default_corpus_paths,
    resolve_case_text,
)


DEFAULT_CASE_FILE = PROJECT_ROOT / "数据" / "病例.txt"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "outputs" / "single_case_e2e_result.json"
DEFAULT_SNAPSHOT_DIR = PROJECT_ROOT / "outputs" / "snapshots_e2e"


def _resolve_existing_file(raw_path: str | None, *, label: str) -> str | None:
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    return str(path.resolve())


def _resolve_optional_path(raw_path: str | None) -> str | None:
    text = str(raw_path or "").strip()
    if not text:
        return None
    return str(Path(text).expanduser().resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a fully wired MedAI single-case end-to-end workflow.",
    )
    parser.add_argument(
        "--case-text",
        help="Clinical case text passed directly on the command line.",
    )
    parser.add_argument(
        "--case-file",
        default=str(DEFAULT_CASE_FILE),
        help="Path to the clinical case text file. Used when --case-text is not provided.",
    )
    parser.add_argument(
        "--mode",
        choices=("patient_note", "question"),
        default="patient_note",
        help="Workflow mode.",
    )
    parser.add_argument(
        "--retriever-backend",
        choices=("bm25", "vector", "hybrid"),
        default="hybrid",
        help="Workflow retrieval backend.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Retrieval recall top-k. Defaults to MEDAI_RETRIEVAL_TOP_K or MEDAI_TOP_K.",
    )
    parser.add_argument(
        "--max-selected-tools",
        type=int,
        default=5,
        help="Maximum number of calculators to keep in patient_note mode.",
    )
    parser.add_argument(
        "--forced-tool-pmid",
        default="",
        help="Optional oracle PMID to force during execution.",
    )
    parser.add_argument(
        "--riskcalcs-path",
        help="Optional override for riskcalcs.json. By default a complete local corpus pair is discovered.",
    )
    parser.add_argument(
        "--pmid-metadata-path",
        help="Optional override for pmid2info.json. By default a complete local corpus pair is discovered.",
    )
    parser.add_argument(
        "--trial-output-root",
        default="",
        help="Optional trial chunk KB root. Defaults to the repository's resolved trial vector KB root.",
    )
    parser.add_argument(
        "--trial-vector-store",
        choices=("auto", "faiss", "qdrant", "qdrant_local"),
        default="auto",
        help="Explicit trial vector store for the injected trial retriever.",
    )
    parser.add_argument(
        "--trial-qdrant-collection",
        default="",
        help="Optional Qdrant collection name for the injected trial retriever.",
    )
    parser.add_argument(
        "--trial-qdrant-url",
        default="",
        help="Optional Qdrant server URL for the injected trial retriever.",
    )
    parser.add_argument(
        "--trial-qdrant-api-key",
        default="",
        help="Optional Qdrant API key for remote trial retrieval.",
    )
    parser.add_argument(
        "--trial-qdrant-path",
        default="",
        help="Optional embedded Qdrant path for the injected trial retriever.",
    )
    parser.add_argument(
        "--skip-trial-retriever-injection",
        action="store_true",
        help="Let the workflow resolve trial retrieval on its own instead of injecting a prebuilt retriever.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_FILE),
        help="Path to save the full workflow result JSON.",
    )
    parser.add_argument(
        "--snapshot-dir",
        default=str(DEFAULT_SNAPSHOT_DIR),
        help="Directory for workflow snapshots.",
    )
    parser.add_argument(
        "--show-json",
        action="store_true",
        help="Print the full workflow JSON instead of only the summary.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for the workflow run.",
    )
    return parser


def _resolve_corpus_pair(args: argparse.Namespace) -> tuple[str | None, str | None]:
    default_riskcalcs_path, default_pmid_metadata_path = default_corpus_paths(PROJECT_ROOT)
    riskcalcs_path = _resolve_existing_file(args.riskcalcs_path, label="RiskCalcs") or default_riskcalcs_path
    pmid_metadata_path = (
        _resolve_existing_file(args.pmid_metadata_path, label="PMID metadata") or default_pmid_metadata_path
    )
    return riskcalcs_path, pmid_metadata_path


def _build_trial_tool_registry(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, str]]:
    if args.skip_trial_retriever_injection:
        return {}, {}

    trial_output_root = str(resolve_trial_vector_output_root(args.trial_output_root or None))
    trial_qdrant_path = _resolve_optional_path(args.trial_qdrant_path)
    retriever = create_trial_chunk_retrieval_tool(
        output_root=trial_output_root,
        backend=args.retriever_backend,
        vector_store=args.trial_vector_store,
        qdrant_collection_name=args.trial_qdrant_collection or None,
        qdrant_url=args.trial_qdrant_url or None,
        qdrant_api_key=args.trial_qdrant_api_key or None,
        qdrant_path=trial_qdrant_path,
    )
    config_summary = {
        "trial_output_root": trial_output_root,
        "trial_vector_store": args.trial_vector_store,
        "trial_qdrant_collection": args.trial_qdrant_collection or "",
        "trial_qdrant_url": args.trial_qdrant_url or "",
        "trial_qdrant_path": trial_qdrant_path or "",
    }
    return {"trial_retriever": retriever}, config_summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv_if_present(PROJECT_ROOT / ".env")

    riskcalcs_path, pmid_metadata_path = _resolve_corpus_pair(args)
    case_text, case_source = resolve_case_text(args)
    snapshot_dir = _resolve_optional_path(args.snapshot_dir)
    tool_registry, trial_config = _build_trial_tool_registry(args)

    result = run_workflow(
        case_text=case_text,
        mode=args.mode,
        riskcalcs_path=riskcalcs_path,
        pmid_metadata_path=pmid_metadata_path,
        retriever_backend=args.retriever_backend,
        top_k=args.top_k,
        max_selected_tools=max(int(args.max_selected_tools), 1),
        forced_tool_pmid=str(args.forced_tool_pmid or "").strip() or None,
        debug=args.debug,
        snapshot_dir=snapshot_dir,
        tool_registry=tool_registry or None,
    )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Case source:", case_source)
    print("RiskCalcs path:", riskcalcs_path or "")
    print("PMID metadata path:", pmid_metadata_path or "")
    print("Retriever backend:", args.retriever_backend)
    if trial_config:
        print("Injected trial retriever:", json.dumps(trial_config, ensure_ascii=False))
    else:
        print("Injected trial retriever: disabled")
    print("Result JSON:", output_path)
    if snapshot_dir:
        print("Snapshot dir:", snapshot_dir)

    if args.show_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(build_summary(result), ensure_ascii=False, indent=2))

    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
