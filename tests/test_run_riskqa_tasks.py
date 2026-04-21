from __future__ import annotations

import sys
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.use.riskqa_support import (
    build_question_query,
    extract_choice_from_text,
    select_entries,
)


class RiskQASupportTests(unittest.TestCase):
    def test_build_question_query_appends_choices_and_answer_instruction(self) -> None:
        query = build_question_query(
            {
                "question": "What is the estimated stroke rate?",
                "choices": {
                    "A": "1.9",
                    "B": "2.8",
                    "C": "4.0",
                },
            }
        )

        self.assertIn("Question: What is the estimated stroke rate?", query)
        self.assertIn("A. 1.9", query)
        self.assertIn("B. 2.8", query)
        self.assertIn("C. 4.0", query)
        self.assertIn("Answer: <LETTER>", query)

    def test_extract_choice_from_text_prefers_explicit_letter(self) -> None:
        choice = extract_choice_from_text(
            "Computed score is 6. Therefore the best matching option is E.\nAnswer: E",
            {"A": "1.9", "B": "2.8", "C": "4.0", "D": "5.9", "E": "8.5"},
        )

        self.assertEqual(choice, "E")

    def test_extract_choice_from_text_falls_back_to_choice_value_match(self) -> None:
        choice = extract_choice_from_text(
            "The patient has a high risk and the estimated stroke rate is 8.5 per 100 patient-years.",
            {"A": "1.9", "B": "2.8", "C": "4.0", "D": "5.9", "E": "8.5"},
        )

        self.assertIsNone(choice)

    def test_select_entries_applies_slice_and_limit(self) -> None:
        selected = select_entries(
            [{"question": "Q0"}, {"question": "Q1"}, {"question": "Q2"}, {"question": "Q3"}],
            start_index=1,
            end_index=4,
            limit=2,
        )

        self.assertEqual(selected, [(1, {"question": "Q1"}), (2, {"question": "Q2"})])


if __name__ == "__main__":
    unittest.main()
