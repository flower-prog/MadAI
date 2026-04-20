from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
import unittest
import uuid
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import scripts.try_single_case_workflow as try_single_case_workflow_module
from scripts.use.try_single_case_workflow import build_summary, default_corpus_paths, normalize_case_text


class TrySingleCaseWorkflowTests(unittest.TestCase):
    def test_script_supports_direct_execution(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "use" / "try_single_case_workflow.py"), "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertIn("Run the MedAI workflow against a single demo clinical case.", completed.stdout)

    def test_parser_defaults_snapshot_dir_to_outputs_snapshots(self) -> None:
        args = try_single_case_workflow_module.parse_args([])

        self.assertEqual(
            Path(args.snapshot_dir),
            try_single_case_workflow_module.DEFAULT_SNAPSHOT_DIR,
        )

    def test_normalize_case_text_prefers_structured_sections_from_fragment(self) -> None:
        raw_text = (
            '病史摘要患者诊断胼胝体梗死后失用症伴失语34 d。'
            '","resolved_detail_abstract_en":null,'
            '"resolved_detail_abstract_zh_sections":['
            '{"label":"病史摘要","text":"患者诊断胼胝体梗死后失用症伴失语34 d。"},'
            '{"label":"治疗方法","text":"康复训练治疗和经颅直流电刺激（tDCS）定位治疗。"}'
            "]"
        )

        normalized = normalize_case_text(raw_text)

        self.assertIn("病史摘要：患者诊断胼胝体梗死后失用症伴失语34 d。", normalized)
        self.assertIn("治疗方法：康复训练治疗和经颅直流电刺激（tDCS）定位治疗。", normalized)
        self.assertNotIn("resolved_detail_abstract_en", normalized)

    def test_build_summary_collects_key_fields(self) -> None:
        result = {
            "status": "completed",
            "review_passed": True,
            "clinical_answer": [{"name": "rehab_plan"}],
            "errors": [],
            "final_output": {
                "clinical_tool_agent": {
                    "selected_tool": {"pmid": "123"},
                    "executions": [{"pmid": "123", "status": "completed"}],
                }
            },
        }

        summary = build_summary(result)

        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["review_passed"], True)
        self.assertEqual(summary["clinical_answer"], [{"name": "rehab_plan"}])
        self.assertEqual(summary["selected_tool"], {"pmid": "123"})
        self.assertEqual(summary["executions"], [{"pmid": "123", "status": "completed"}])

    def test_default_corpus_paths_prefers_local_riskcalcs_and_discovers_missing_metadata(self) -> None:
        sandbox_root = PROJECT_ROOT / ".tmp_test_artifacts" / f"corpus-paths-{uuid.uuid4().hex}"
        try:
            medai_root = sandbox_root / "MedAI"
            local_data_dir = medai_root / "数据"
            local_data_dir.mkdir(parents=True)

            local_riskcalcs_path = local_data_dir / "riskcalcs.json"
            local_riskcalcs_path.write_text("{}", encoding="utf-8")

            sibling_root = medai_root.parent / "Clinical-Tool-Learning" / "riskqa_evaluation"
            sibling_tools_dir = sibling_root / "tools"
            sibling_dataset_dir = sibling_root / "dataset"
            sibling_tools_dir.mkdir(parents=True)
            sibling_dataset_dir.mkdir(parents=True)

            (sibling_tools_dir / "riskcalcs.json").write_text("{}", encoding="utf-8")
            sibling_pmid_metadata_path = sibling_dataset_dir / "pmid2info.json"
            sibling_pmid_metadata_path.write_text("{}", encoding="utf-8")

            riskcalcs_path, pmid_metadata_path = default_corpus_paths(medai_root)

            self.assertEqual(Path(riskcalcs_path), local_riskcalcs_path.resolve())
            self.assertEqual(Path(pmid_metadata_path), sibling_pmid_metadata_path.resolve())
        finally:
            shutil.rmtree(sandbox_root, ignore_errors=True)

    def test_main_passes_snapshot_dir_into_run_workflow(self) -> None:
        sandbox_root = PROJECT_ROOT / ".tmp_test_artifacts" / f"single-case-main-{uuid.uuid4().hex}"
        sandbox_root.mkdir(parents=True, exist_ok=True)
        output_path = sandbox_root / "result.json"
        snapshot_dir = sandbox_root / "snapshots"

        args = argparse.Namespace(
            case_text="患者房颤伴TIA病史，评估卒中风险。",
            case_file=str(PROJECT_ROOT / "数据" / "病例.txt"),
            mode="patient_note",
            top_k=8,
            riskcalcs_path=None,
            pmid_metadata_path=None,
            output=str(output_path),
            show_json=False,
            debug=False,
            snapshot_dir=str(snapshot_dir),
        )
        fake_result = {
            "status": "completed",
            "review_passed": True,
            "clinical_answer": [{"name": "demo"}],
            "errors": [],
            "final_output": {},
        }

        try:
            with patch.object(try_single_case_workflow_module, "parse_args", return_value=args), patch.object(
                try_single_case_workflow_module, "load_dotenv_if_present"
            ), patch.object(
                try_single_case_workflow_module,
                "default_corpus_paths",
                return_value=("/tmp/riskcalcs.json", "/tmp/pmid2info.json"),
            ), patch.object(
                try_single_case_workflow_module,
                "resolve_case_text",
                return_value=("患者房颤伴TIA病史，评估卒中风险。", "inline"),
            ), patch.object(
                try_single_case_workflow_module,
                "run_workflow",
                return_value=fake_result,
            ) as run_workflow_mock:
                exit_code = try_single_case_workflow_module.main()

            self.assertEqual(exit_code, 0)
            run_workflow_mock.assert_called_once()
            self.assertEqual(
                run_workflow_mock.call_args.kwargs["snapshot_dir"],
                str(snapshot_dir.resolve()),
            )
            self.assertTrue(output_path.exists())
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8"))["status"],
                "completed",
            )
        finally:
            shutil.rmtree(sandbox_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
