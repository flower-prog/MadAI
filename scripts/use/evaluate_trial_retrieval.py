from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)


from agent.graph.nodes import protocol_node  # noqa: E402
from agent.protocol.evaluation import load_eval_cases, run_protocol_trial_eval  # noqa: E402


DEFAULT_CASES_PATH = PROJECT_ROOT / "data" / "eval" / "trial_cases.example.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "eval" / "trial_retrieval_eval.json"
DEFAULT_TRACE_DIR = PROJECT_ROOT / "outputs" / "eval" / "trial_retrieval_traces"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate MedAI protocol trial retrieval and eligibility traces over JSONL cases."
    )
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH), help="Path to trial eval cases JSONL.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Path for summary metrics JSON.")
    parser.add_argument("--trace-dir", default=str(DEFAULT_TRACE_DIR), help="Directory for per-case trace JSON files.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of cases to run.")
    parser.add_argument("--backend", default="hybrid", choices=["keyword", "vector", "hybrid"], help="Trial retriever backend.")
    parser.add_argument("--top-k", type=int, default=20, help="Fine retrieval top-k for the eval state.")
    parser.add_argument("--fail-fast", action="store_true", help="Raise on the first protocol exception.")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = load_eval_cases(args.cases, limit=args.limit)
    summary = run_protocol_trial_eval(
        cases,
        output_path=args.output,
        trace_dir=args.trace_dir,
        protocol_runner=protocol_node,
        backend=args.backend,
        top_k=args.top_k,
        fail_fast=args.fail_fast,
    )
    print(f"Trial retrieval eval completed: {summary['completed_case_count']}/{summary['case_count']} cases")
    print(f"Summary JSON: {Path(args.output).resolve()}")
    print(f"Trace dir: {Path(args.trace_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
