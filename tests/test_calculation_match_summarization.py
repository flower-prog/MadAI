from __future__ import annotations

import unittest

from agent.graph.nodes import _build_calculation_tasks_from_matches, _summarize_calculator_matches
from agent.graph.types import CalculatorMatch


class CalculationMatchSummarizationTests(unittest.TestCase):
    def test_summarize_calculator_matches_uses_execution_first_gate_fields(self) -> None:
        result = {
            "mode": "patient_note",
            "retrieved_tools": [
                {"pmid": "1", "title": "CHADS2 Stroke Risk Calculator"},
                {"pmid": "2", "title": "FIB-4 Index for Liver Fibrosis"},
            ],
            "selected_tool": {"pmid": "1", "title": "CHADS2 Stroke Risk Calculator"},
            "selection_decisions": [
                {
                    "pmid": "1",
                    "title": "CHADS2 Stroke Risk Calculator",
                    "gate_status": "passed",
                    "execution_status": "completed",
                    "rationale": "Calculator produced a concrete score.",
                },
                {
                    "pmid": "2",
                    "title": "FIB-4 Index for Liver Fibrosis",
                    "gate_status": "failed_missing_inputs",
                    "execution_status": "missing_inputs",
                    "missing_inputs": ["ast", "alt", "platelet_count"],
                    "rationale": "Required liver parameters are missing.",
                },
            ],
            "executions": [
                {
                    "pmid": "1",
                    "title": "CHADS2 Stroke Risk Calculator",
                    "status": "completed",
                    "result": 3,
                    "final_text": "CHADS2 score is 3.",
                }
            ],
        }

        matches = _summarize_calculator_matches(result, "risk_score")
        by_pmid = {match.pmid: match for match in matches}

        self.assertEqual(by_pmid["1"].applicability, "selected")
        self.assertEqual(by_pmid["1"].execution_status, "completed")
        self.assertEqual(by_pmid["1"].value, 3)
        self.assertEqual(by_pmid["2"].applicability, "data_missing")
        self.assertEqual(by_pmid["2"].execution_status, "missing_inputs")
        self.assertEqual(by_pmid["2"].missing_inputs, ["ast", "alt", "platelet_count"])

    def test_build_calculation_tasks_from_matches_propagates_values_and_skips_missing_inputs(self) -> None:
        matches = [
            CalculatorMatch(
                pmid="1",
                title="CHADS2 Stroke Risk Calculator",
                category="risk_score",
                applicability="selected",
                execution_status="completed",
                value=3,
            ),
            CalculatorMatch(
                pmid="2",
                title="FIB-4 Index for Liver Fibrosis",
                category="risk_score",
                applicability="data_missing",
                execution_status="missing_inputs",
                missing_inputs=["ast", "alt"],
            ),
        ]

        tasks, results = _build_calculation_tasks_from_matches(matches, "risk_score")

        self.assertEqual(tasks[0].decision, "direct")
        self.assertEqual(results[0].status, "completed")
        self.assertEqual(results[0].value, 3)
        self.assertEqual(tasks[1].decision, "skip")
        self.assertEqual(results[1].status, "skipped")
        self.assertEqual(results[1].missing_inputs, ["ast", "alt"])


if __name__ == "__main__":
    unittest.main()
