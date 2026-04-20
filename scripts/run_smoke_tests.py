from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR in sys.path:
    sys.path.remove(PROJECT_ROOT_STR)
sys.path.insert(0, PROJECT_ROOT_STR)


SMOKE_SUITES: dict[str, list[str]] = {
    "core": [
        "tests.test_tool_spec_export",
        "tests.test_graph_snapshots",
        "tests.test_workflow",
        "tests.test_try_single_case_workflow",
    ],
    "retrieval": [
        "tests.test_retrieval_tools_minimal",
        "tests.test_clinical_tool_agent_parameter_retrieval",
    ],
    "trials": [
        "tests.test_trial_retrieval_tools",
        "tests.test_trial_vector_retrieval_tools",
        "tests.test_trial_vector_kb",
    ],
}
SMOKE_SUITES["all"] = (
    SMOKE_SUITES["core"] + SMOKE_SUITES["retrieval"] + SMOKE_SUITES["trials"]
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a fast MedAI smoke-test subset to catch regressions before longer evaluations."
    )
    parser.add_argument(
        "--suite",
        choices=tuple(SMOKE_SUITES.keys()),
        default="all",
        help="Named smoke suite to run. Default: all.",
    )
    parser.add_argument(
        "--module",
        action="append",
        default=[],
        help="Additional unittest module to append. Can be passed multiple times.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch unittest subprocesses. Default: current interpreter.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Pass -v to unittest and print full output for passing modules too.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failing smoke-test module.",
    )
    parser.add_argument(
        "--json-output",
        help="Optional path to write a JSON summary of the smoke-test results.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print available suites and modules, then exit.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _unique_modules(modules: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for module_name in modules:
        normalized = str(module_name).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def resolve_modules(args: argparse.Namespace) -> list[str]:
    modules = list(SMOKE_SUITES[args.suite])
    modules.extend(args.module or [])
    return _unique_modules(modules)


def list_suites() -> int:
    print("Available smoke suites:")
    for suite_name, modules in SMOKE_SUITES.items():
        print(f"- {suite_name}")
        for module_name in modules:
            print(f"  - {module_name}")
    return 0


def run_module(module_name: str, *, python_executable: str, verbose: bool) -> dict[str, Any]:
    command = [python_executable, "-m", "unittest"]
    if verbose:
        command.append("-v")
    command.append(module_name)

    started_at = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    duration_seconds = time.perf_counter() - started_at

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    combined_output = "\n".join(part for part in (stdout.strip(), stderr.strip()) if part.strip())

    return {
        "module": module_name,
        "command": command,
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "duration_seconds": round(duration_seconds, 3),
        "stdout": stdout,
        "stderr": stderr,
        "combined_output": combined_output,
    }


def print_result(result: dict[str, Any], *, verbose: bool) -> None:
    status = "PASS" if result["passed"] else "FAIL"
    print(f"[{status}] {result['module']} ({result['duration_seconds']:.3f}s)")
    if verbose or not result["passed"]:
        output = str(result.get("combined_output") or "").strip()
        if output:
            print(output)
            print()


def write_json_summary(path_str: str, payload: dict[str, Any]) -> Path:
    output_path = Path(path_str).expanduser()
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        return list_suites()

    modules = resolve_modules(args)
    if not modules:
        print("No smoke-test modules resolved.")
        return 1

    print(f"Running smoke suite '{args.suite}' with {len(modules)} module(s).")
    results: list[dict[str, Any]] = []

    for module_name in modules:
        result = run_module(
            module_name,
            python_executable=str(args.python),
            verbose=bool(args.verbose),
        )
        results.append(result)
        print_result(result, verbose=bool(args.verbose))
        if args.fail_fast and not result["passed"]:
            break

    failed_modules = [result["module"] for result in results if not result["passed"]]
    summary = {
        "suite": args.suite,
        "python": str(args.python),
        "module_count": len(results),
        "failed_count": len(failed_modules),
        "failed_modules": failed_modules,
        "results": results,
    }

    if args.json_output:
        output_path = write_json_summary(args.json_output, summary)
        print(f"JSON summary: {output_path}")

    if failed_modules:
        print("Smoke suite failed.")
        for module_name in failed_modules:
            print(f"- {module_name}")
        return 1

    print("Smoke suite passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
