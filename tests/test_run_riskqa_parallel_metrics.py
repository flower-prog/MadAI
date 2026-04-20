from __future__ import annotations

import sys
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.run_riskqa_parallel import build_evaluation_summary, default_evaluation_report_path


class RiskQABatchMetricsTests(unittest.TestCase):
    def test_build_evaluation_summary_reports_accuracy(self) -> None:
        summary, report_text = build_evaluation_summary(
            [
                {"task_id": "riskqa-00000", "dataset_index": 0, "predicted_choice": "A", "status": "completed"},
                {"task_id": "riskqa-00001", "dataset_index": 1, "predicted_choice": "C", "status": "completed"},
            ],
            [
                {"question": "Q1", "answer": "A"},
                {"question": "Q2", "answer": "B"},
            ],
            wrong_limit=5,
        )

        self.assertEqual(summary["total_records"], 2)
        self.assertEqual(summary["comparable_count"], 2)
        self.assertEqual(summary["correct_count"], 1)
        self.assertEqual(summary["wrong_count"], 1)
        self.assertAlmostEqual(summary["accuracy"], 0.5)
        self.assertIn("Accuracy: 50.00%", report_text)
        self.assertIn("riskqa-00001", report_text)

    def test_default_evaluation_report_path_uses_output_stem(self) -> None:
        report_path = default_evaluation_report_path(
            PROJECT_ROOT / "outputs" / "riskqa" / "riskqa_answers.txt"
        )

        self.assertEqual(report_path.name, "riskqa_answers.evaluation.txt")


if __name__ == "__main__":
    unittest.main()
