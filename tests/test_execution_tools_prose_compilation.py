from __future__ import annotations

import unittest
from dataclasses import dataclass

from agent.tools import RiskCalcExecutor, RiskCalcRegistry


_PROSE_ONLY_COMPUTATION = (
    "The risk score is computed based on twelve characteristics. "
    "Each characteristic is assigned a score of 1 if the condition is met, and 0 otherwise. "
    "The characteristics are:\n\n"
    "1. Age > 40\n"
    "2. Male\n"
    "3. Non-white ethnicity\n"
    "4. Oxygen saturations < 93%\n"
    "5. Radiological severity score > 3\n"
    "6. Neutrophil count > 8.0 x 10^9/L\n"
    "7. CRP > 40 mg/L\n"
    "8. Albumin < 34 g/L\n"
    "9. Creatinine > 100 umol/L\n"
    "10. Diabetes mellitus\n"
    "11. Hypertension\n"
    "12. Chronic lung disease\n\n"
    "The total risk score is the sum of the scores for each characteristic.\n"
)


@dataclass
class _FakeDocument:
    pmid: str
    title: str
    computation: str
    example: str = ""
    purpose: str = "Estimate short-term deterioration risk."
    eligibility: str = "Adults admitted with the target disease."
    interpretation: str = ""
    utility: str = ""


class _FakeCatalog:
    def __init__(self, documents: list[_FakeDocument]) -> None:
        self._documents = list(documents)

    def documents(self) -> list[_FakeDocument]:
        return list(self._documents)


class ExecutionToolsProseCompilationTests(unittest.TestCase):
    def test_registry_registers_prose_only_additive_rule_calculator(self) -> None:
        catalog = _FakeCatalog(
            [
                _FakeDocument(
                    pmid="covid-12",
                    title="Twelve Factor Deterioration Score",
                    computation=_PROSE_ONLY_COMPUTATION,
                )
            ]
        )

        registry = RiskCalcRegistry.from_catalog_with_defaults(catalog)

        self.assertTrue(registry.has("covid-12"))
        registration = registry.get("covid-12")
        self.assertIn("age", registration.parameter_names)
        self.assertIn("sex", registration.parameter_names)
        self.assertIn("non_white_ethnicity", registration.parameter_names)
        self.assertIn("oxygen_saturation", registration.parameter_names)
        self.assertIn("radiological_severity_score", registration.parameter_names)
        self.assertIn("neutrophil_count", registration.parameter_names)
        self.assertIn("crp", registration.parameter_names)
        self.assertIn("albumin", registration.parameter_names)
        self.assertIn("creatinine", registration.parameter_names)
        self.assertIn("diabetes", registration.parameter_names)
        self.assertIn("hypertension", registration.parameter_names)
        self.assertIn("chronic_lung_disease", registration.parameter_names)
        self.assertIn("def ", registration.code)

    def test_executor_runs_prose_only_additive_rule_calculator(self) -> None:
        catalog = _FakeCatalog(
            [
                _FakeDocument(
                    pmid="covid-12",
                    title="Twelve Factor Deterioration Score",
                    computation=_PROSE_ONLY_COMPUTATION,
                )
            ]
        )
        registry = RiskCalcRegistry.from_catalog_with_defaults(catalog)
        executor = RiskCalcExecutor(registry)

        execution = executor.run(
            "covid-12",
            {
                "age": 45,
                "sex": "male",
                "non_white_ethnicity": True,
                "oxygen_saturation": 92,
                "radiological_severity_score": 4,
                "neutrophil_count": 7.5,
                "crp": 45,
                "albumin": 33,
                "creatinine": 105,
                "diabetes": True,
                "hypertension": False,
                "chronic_lung_disease": True,
            },
        )

        self.assertEqual(execution["status"], "completed")
        self.assertEqual(execution["result"], 10)
        self.assertEqual(execution["inputs"]["sex"], "male")
        self.assertEqual(execution["inputs"]["hypertension"], False)


if __name__ == "__main__":
    unittest.main()
