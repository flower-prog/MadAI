from __future__ import annotations

import sys
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agent.protocol.patient_card import build_patient_evidence_card
from agent.protocol.trial_card import build_trial_card, build_trial_card_text


class ProtocolCardTests(unittest.TestCase):
    def test_patient_evidence_card_uses_only_case_and_intent_facts(self) -> None:
        card = build_patient_evidence_card(
            {
                "raw_text": (
                    "48 M with HTN hyperlipidemia, bicuspid aortic valve, tobacco abuse, "
                    "critical aortic stenosis, EF was 25%. Cath showed no flow-limiting CAD."
                ),
                "case_summary": "48 M with critical aortic stenosis evaluated for valve replacement.",
                "problem_list": ["critical severe aortic stenosis", "EF 25%"],
                "known_facts": ["no angiographically apparent flow-limiting coronary artery disease"],
                "missing_information": [],
            },
            trial_search_intent={
                "primary_conditions": [
                    {"canonical": "Aortic Valve Stenosis"},
                    {"canonical": "Bicuspid Aortic Valve"},
                    {"canonical": "Left Ventricular Dysfunction"},
                ],
                "interventions_of_interest": [{"canonical": "Aortic Valve Replacement"}],
                "patient_constraints": {
                    "age": 48,
                    "sex": "Male",
                    "key_measurements": {"LVEF": "25%"},
                },
            },
        )

        self.assertEqual(card["age"], 48)
        self.assertEqual(card["sex"], "Male")
        self.assertIn("Aortic Valve Stenosis", card["positive_facts"])
        self.assertIn("Aortic Valve Replacement", card["positive_facts"])
        self.assertIn("LVEF 25%", card["positive_facts"])
        self.assertIn(
            "no angiographically apparent flow-limiting coronary artery disease",
            card["negative_facts"],
        )
        self.assertEqual(card["unknown_facts"], [])

    def test_trial_card_preserves_eligibility_and_matched_chunks(self) -> None:
        card = build_trial_card(
            {
                "nct_id": "NCTTEST001",
                "display_title": "TAVR in Severe Aortic Stenosis",
                "overall_status": "Recruiting",
                "study_type": "Interventional",
                "conditions": ["Aortic Valve Stenosis"],
                "interventions": ["TAVR"],
                "brief_summary": "TAVR trial.",
                "eligibility_inclusion_text": "Adults with symptomatic severe aortic stenosis.",
                "eligibility_exclusion_text": "Active endocarditis.",
                "gender": "All",
                "minimum_age": "18 Years",
                "maximum_age": "N/A",
            },
            candidate={
                "nct_id": "NCTTEST001",
                "score": 0.8,
                "matched_chunks": [
                    {
                        "chunk_id": "NCTTEST001::overview::0",
                        "chunk_type": "overview",
                        "score": 0.9,
                        "text": "overview text",
                    }
                ],
            },
        )

        self.assertEqual(card["nct_id"], "NCTTEST001")
        self.assertEqual(card["title"], "TAVR in Severe Aortic Stenosis")
        self.assertEqual(card["eligibility"]["inclusion_text"], "Adults with symptomatic severe aortic stenosis.")
        self.assertEqual(card["matched_chunks"][0]["chunk_type"], "overview")
        text = build_trial_card_text(card)
        self.assertIn("TAVR in Severe Aortic Stenosis", text)
        self.assertIn("inclusion criteria: Adults with symptomatic severe aortic stenosis.", text)


if __name__ == "__main__":
    unittest.main()
