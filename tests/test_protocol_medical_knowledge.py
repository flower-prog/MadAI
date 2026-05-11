from __future__ import annotations

import sys
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agent.protocol.medical_knowledge import retrieve_medical_knowledge_for_protocol


class ProtocolMedicalKnowledgeTests(unittest.TestCase):
    def test_retrieval_module_returns_stable_empty_bundle_without_retriever(self) -> None:
        bundle = retrieve_medical_knowledge_for_protocol(
            structured_case={
                "problem_list": ["metastatic melanoma"],
            },
            trial_retrieval_bundle={
                "candidate_ranking": [
                    {
                        "nct_id": "NCTMEL001",
                        "conditions": ["Melanoma"],
                        "interventions": ["Pembrolizumab"],
                    }
                ]
            },
            eligibility_assessment_bundle={
                "assessed_trials": [
                    {
                        "criteria": [
                            {
                                "label": "unknown",
                                "raw_text": "Active autoimmune disease requiring systemic therapy",
                            }
                        ]
                    }
                ]
            },
            calculator_evidence_bundle={
                "risk_evidence_items": [
                    {
                        "calculator": "ECOG performance status",
                        "category": "performance_status",
                    }
                ]
            },
        )

        self.assertEqual(bundle["schema_version"], 1)
        self.assertEqual(bundle["status"], "not_configured")
        self.assertEqual(bundle["backend_used"], "none")
        self.assertTrue(bundle["queries"])
        self.assertEqual(bundle["retrieved_items"], [])
        self.assertTrue(bundle["knowledge_gaps"])
        self.assertTrue(
            any("Active autoimmune disease" in item["query"] for item in bundle["queries"])
        )


if __name__ == "__main__":
    unittest.main()
