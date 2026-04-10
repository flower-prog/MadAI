from __future__ import annotations

import sys
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.use.evaluate import build_report, evaluate_records
from scripts.use.riskqa_support import format_record_block, parse_record_blocks
from scripts.use.run_riskqa_parallel import build_result_record


class RiskQATextRecordTests(unittest.TestCase):
    def test_format_and_parse_record_blocks_round_trip(self) -> None:
        block = format_record_block(
            {
                "task_id": "riskqa-00000",
                "dataset_index": 0,
                "predicted_choice": "B",
                "answer_text": "Answer: B.",
            }
        )

        records = parse_record_blocks(block)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["task_id"], "riskqa-00000")
        self.assertEqual(records[0]["predicted_choice"], "B")


class RiskQARunnerTests(unittest.TestCase):
    def test_build_result_record_extracts_choice_from_workflow_result(self) -> None:
        record = build_result_record(
            task_id="riskqa-00003",
            dataset_index=3,
            entry={
                "question": "Which option is correct?",
                "choices": {"A": "Low", "B": "High"},
                "answer": "B",
                "pmid": "12345",
            },
            workflow_result={
                "status": "completed",
                "review_passed": True,
                "errors": [],
                "final_output": {
                    "clinical_tool_agent": {
                        "selected_tool": {"pmid": "12345"},
                        "execution": {
                            "final_text": "The correct option is High.\nAnswer: B.",
                            "messages": [
                                {
                                    "role": "assistant",
                                    "content": "The correct option is High.\nAnswer: B.",
                                }
                            ],
                        },
                    }
                },
            },
            elapsed_seconds=1.25,
        )

        self.assertEqual(record["predicted_choice"], "B")
        self.assertEqual(record["gold_answer"], "B")
        self.assertTrue(record["correct"])
        self.assertEqual(record["status"], "completed")


class RiskQAEvaluateTests(unittest.TestCase):
    def test_evaluate_records_compares_predicted_choices_against_dataset_answers(self) -> None:
        summary = evaluate_records(
            [
                {"task_id": "riskqa-00000", "dataset_index": 0, "predicted_choice": "A", "status": "completed"},
                {"task_id": "riskqa-00001", "dataset_index": 1, "predicted_choice": "C", "status": "completed"},
            ],
            [
                {"question": "Q1", "answer": "A"},
                {"question": "Q2", "answer": "B"},
            ],
        )

        self.assertEqual(summary["total_records"], 2)
        self.assertEqual(summary["correct_count"], 1)
        self.assertEqual(summary["wrong_count"], 1)
        self.assertAlmostEqual(summary["accuracy"], 0.5)

        report = build_report(summary, wrong_limit=5)
        self.assertIn("Accuracy: 50.00%", report)
        self.assertIn("riskqa-00001", report)


if __name__ == "__main__":
    unittest.main()
