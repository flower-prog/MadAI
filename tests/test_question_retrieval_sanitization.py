from __future__ import annotations

import unittest

from agent.graph.nodes import _sanitize_question_retrieval_text


class QuestionRetrievalSanitizationTests(unittest.TestCase):
    def test_sanitize_question_retrieval_text_drops_choices_and_answer_instructions(self) -> None:
        cleaned = _sanitize_question_retrieval_text(
            "\n".join(
                [
                    "A 65-year-old patient with hypertension and diabetes has no prior stroke or CHF.",
                    "What is the estimated stroke rate per 100 patient-years without antithrombotic therapy?",
                    "",
                    "Choices:",
                    "A. 1.9",
                    "B. 2.8",
                    "C. 4.0",
                    "",
                    "Please compute the answer using the appropriate calculator and choose the best option.",
                    "End your final answer with the format: Answer: <LETTER>.",
                ]
            )
        )

        self.assertIn("A 65-year-old patient with hypertension and diabetes", cleaned)
        self.assertIn("What is the estimated stroke rate", cleaned)
        self.assertNotIn("Choices:", cleaned)
        self.assertNotIn("A. 1.9", cleaned)
        self.assertNotIn("Please compute the answer", cleaned)
        self.assertNotIn("Answer: <LETTER>", cleaned)


if __name__ == "__main__":
    unittest.main()
