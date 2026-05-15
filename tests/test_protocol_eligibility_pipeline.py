from __future__ import annotations

import sys
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agent.protocol.criteria_parser import parse_trial_criteria
from agent.protocol.pipeline import assess_trial_eligibility_candidates


class _FakeEligibilityChatClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = []

    def complete(self, messages, *, model=None, temperature=0.0) -> str:
        self.calls.append({"messages": messages, "model": model, "temperature": temperature})
        return self.response


class ProtocolEligibilityPipelineTests(unittest.TestCase):
    def test_parser_splits_inclusion_and_exclusion_criteria_with_stable_ids(self) -> None:
        criteria = parse_trial_criteria(
            {
                "nct_id": "NCTTEST001",
                "eligibility_inclusion_text": "Inclusion Criteria:\n- Age >= 18 years\n- ECOG performance status 0-1",
                "eligibility_exclusion_text": "Exclusion Criteria:\n1. Active brain metastases\n2. Pregnancy",
            }
        )

        self.assertEqual([item.criterion_id for item in criteria], [
            "NCTTEST001::inclusion::001",
            "NCTTEST001::inclusion::002",
            "NCTTEST001::exclusion::001",
            "NCTTEST001::exclusion::002",
        ])
        self.assertEqual(criteria[0].condition, "age")
        self.assertEqual(criteria[0].operator, ">=")
        self.assertEqual(criteria[1].condition, "ECOG")
        self.assertEqual(criteria[2].condition, "CNS metastases")
        self.assertEqual(criteria[3].condition, "pregnancy")

    def test_pipeline_outputs_criterion_labels_and_missing_questions(self) -> None:
        bundle = assess_trial_eligibility_candidates(
            structured_case={
                "raw_text": "62-year-old male with melanoma, ECOG 1, and no brain metastases.",
                "case_summary": "Melanoma patient.",
                "problem_list": ["melanoma"],
            },
            calculation_results=[],
            calculator_matches=[],
            trial_bundle={
                "candidate_ranking": [
                    {
                        "nct_id": "NCTTEST001",
                        "title": "Melanoma Trial",
                        "overall_status": "Recruiting",
                        "enrollment_open": True,
                        "eligibility_inclusion_text": "- Age >= 18 years\n- ECOG performance status 0-1",
                        "eligibility_exclusion_text": "- Active brain metastases\n- Pregnancy",
                    }
                ]
            },
            limit=3,
        )

        self.assertEqual(bundle["schema_version"], 1)
        self.assertEqual(bundle["assessed_trial_count"], 1)
        trial = bundle["assessed_trials"][0]
        self.assertEqual(trial["nct_id"], "NCTTEST001")
        labels = {item["criterion_id"]: item["label"] for item in trial["criteria"]}
        self.assertEqual(labels["NCTTEST001::inclusion::001"], "met")
        self.assertEqual(labels["NCTTEST001::inclusion::002"], "met")
        self.assertEqual(labels["NCTTEST001::exclusion::001"], "not_met")
        self.assertEqual(labels["NCTTEST001::exclusion::002"], "unknown")
        self.assertEqual(trial["aggregate_status"], "needs_data")
        self.assertIn("NCTTEST001::exclusion::002", trial["unknown_criteria"])
        self.assertTrue(trial["missing_questions"])

    def test_parser_does_not_force_unsplit_eligibility_text_into_inclusion(self) -> None:
        criteria = parse_trial_criteria(
            {
                "nct_id": "NCTUNSPLIT001",
                "eligibility_text": "Adults with melanoma. No active infection.",
                "eligibility_unsplit_text": "Adults with melanoma. No active infection.",
                "eligibility_section_parse_status": "unsplit",
            }
        )

        self.assertEqual(criteria, [])

    def test_pipeline_exposes_parse_warning_stats_for_unsplit_trials(self) -> None:
        bundle = assess_trial_eligibility_candidates(
            structured_case={"raw_text": "62-year-old male with melanoma."},
            calculation_results=[],
            calculator_matches=[],
            trial_bundle={
                "candidate_ranking": [
                    {
                        "nct_id": "NCTUNSPLIT001",
                        "title": "Unsplit Trial",
                        "eligibility_text": "Adults with melanoma. No active infection.",
                        "eligibility_unsplit_text": "Adults with melanoma. No active infection.",
                        "eligibility_section_parse_status": "unsplit",
                        "eligibility_section_parse_warnings": ["eligibility_section_headings_not_found"],
                    }
                ]
            },
            limit=3,
        )

        trial = bundle["assessed_trials"][0]
        self.assertEqual(trial["criteria"], [])
        self.assertEqual(trial["eligibility_section_parse_status"], "unsplit")
        self.assertTrue(trial["eligibility_unsplit_text_present"])
        self.assertEqual(bundle["parse_warning_stats"]["section_parse_status_counts"]["unsplit"], 1)

    def test_pipeline_uses_calculator_evidence_bundle_for_ecog_met(self) -> None:
        bundle = assess_trial_eligibility_candidates(
            structured_case={
                "raw_text": "62-year-old male with metastatic melanoma.",
                "problem_list": ["metastatic melanoma"],
            },
            calculation_results=[],
            calculator_matches=[],
            calculator_evidence_bundle={
                "schema_version": 1,
                "risk_evidence_items": [
                    {
                        "calculator": "ECOG performance status",
                        "category": "performance_status",
                        "value": "1",
                        "unit": "",
                        "status": "completed",
                        "rationale": "ECOG documented by calculator stage.",
                        "usable_for": ["eligibility_evidence"],
                    }
                ],
            },
            trial_bundle={
                "candidate_ranking": [
                    {
                        "nct_id": "NCTECOG001",
                        "title": "ECOG Trial",
                        "overall_status": "Recruiting",
                        "enrollment_open": True,
                        "eligibility_inclusion_text": "- ECOG performance status 0-1",
                    }
                ]
            },
            limit=3,
        )

        criterion = bundle["assessed_trials"][0]["criteria"][0]
        self.assertEqual(criterion["label"], "met")
        self.assertEqual(criterion["evidence_spans"][0]["source"], "calculator_evidence_bundle.risk_evidence_items[1]")
        self.assertEqual(criterion["evidence_spans"][0]["normalized_concept"], "ECOG")

    def test_pipeline_uses_calculator_evidence_bundle_for_ecog_not_met(self) -> None:
        bundle = assess_trial_eligibility_candidates(
            structured_case={
                "raw_text": "62-year-old male with metastatic melanoma.",
                "problem_list": ["metastatic melanoma"],
            },
            calculation_results=[],
            calculator_matches=[],
            calculator_evidence_bundle={
                "schema_version": 1,
                "risk_evidence_items": [
                    {
                        "calculator": "ECOG performance status",
                        "category": "performance_status",
                        "value": "2",
                        "status": "completed",
                        "usable_for": ["eligibility_evidence"],
                    }
                ],
            },
            trial_bundle={
                "candidate_ranking": [
                    {
                        "nct_id": "NCTECOG002",
                        "title": "ECOG Trial",
                        "overall_status": "Recruiting",
                        "enrollment_open": True,
                        "eligibility_inclusion_text": "- ECOG performance status 0-1",
                    }
                ]
            },
            limit=3,
        )

        criterion = bundle["assessed_trials"][0]["criteria"][0]
        self.assertEqual(criterion["label"], "not_met")
        self.assertEqual(criterion["evidence_spans"][0]["normalized_concept"], "ECOG")

    def test_generic_related_evidence_does_not_force_met(self) -> None:
        bundle = assess_trial_eligibility_candidates(
            structured_case={
                "raw_text": "62-year-old male with melanoma receiving immunotherapy.",
                "problem_list": ["melanoma"],
            },
            calculation_results=[],
            calculator_matches=[],
            trial_bundle={
                "candidate_ranking": [
                    {
                        "nct_id": "NCTGENERIC001",
                        "title": "Generic Melanoma Trial",
                        "overall_status": "Recruiting",
                        "enrollment_open": True,
                        "eligibility_inclusion_text": "- Histologically confirmed metastatic melanoma",
                    }
                ]
            },
            limit=3,
        )

        criterion = bundle["assessed_trials"][0]["criteria"][0]
        self.assertEqual(criterion["label"], "unknown")
        self.assertLess(criterion["confidence"], 0.5)
        self.assertTrue(criterion["evidence_spans"])

    def test_pipeline_builds_cards_and_attaches_optional_llm_judgment(self) -> None:
        chat_client = _FakeEligibilityChatClient(
            """
            {
              "results": [
                {
                  "nct_id": "NCTLLM001",
                  "score": 88,
                  "trial_relevance": "high",
                  "eligibility_assessment": "likely_eligible",
                  "inclusion_matches": [
                    {
                      "criterion_or_text": "Adults with symptomatic severe aortic stenosis",
                      "patient_evidence": "critical aortic stenosis",
                      "status": "met"
                    }
                  ],
                  "exclusion_risks": [],
                  "missing_information": [],
                  "short_reason": "Disease and valve intervention context match."
                }
              ]
            }
            """
        )
        bundle = assess_trial_eligibility_candidates(
            structured_case={
                "raw_text": (
                    "48 M with bicuspid aortic valve and critical aortic stenosis. "
                    "EF was 25%. Cath showed no flow-limiting CAD."
                ),
                "case_summary": "48 M with critical aortic stenosis evaluated for valve replacement.",
                "problem_list": ["critical aortic stenosis", "EF 25%"],
                "known_facts": ["no flow-limiting coronary artery disease"],
            },
            calculation_results=[],
            calculator_matches=[],
            trial_search_intent={
                "primary_conditions": [{"canonical": "Aortic Valve Stenosis"}],
                "interventions_of_interest": [{"canonical": "Aortic Valve Replacement"}],
                "patient_constraints": {"age": 48, "sex": "Male", "key_measurements": {"LVEF": "25%"}},
            },
            trial_bundle={
                "candidate_ranking": [
                    {
                        "nct_id": "NCTLLM001",
                        "title": "Aortic Stenosis Trial",
                        "overall_status": "Recruiting",
                        "enrollment_open": True,
                        "conditions": ["Aortic Valve Stenosis"],
                        "interventions": ["Aortic Valve Replacement"],
                        "eligibility_inclusion_text": "Adults with symptomatic severe aortic stenosis.",
                        "eligibility_exclusion_text": "Active endocarditis.",
                    }
                ]
            },
            limit=1,
            eligibility_chat_client=chat_client,
            llm_model="fake-model",
        )

        self.assertEqual(len(chat_client.calls), 1)
        self.assertEqual(bundle["patient_evidence_card"]["age"], 48)
        self.assertIn("Aortic Valve Stenosis", bundle["patient_evidence_card"]["positive_facts"])
        trial = bundle["assessed_trials"][0]
        self.assertEqual(trial["trial_card"]["nct_id"], "NCTLLM001")
        self.assertEqual(trial["llm_eligibility_assessment"]["eligibility_assessment"], "likely_eligible")
        self.assertEqual(trial["llm_eligibility_assessment"]["score"], 88)
        self.assertEqual(bundle["llm_eligibility_bundle"]["status"], "completed")
        self.assertEqual(
            bundle["llm_eligibility_bundle"]["score_summary"],
            [
                {
                    "nct_id": "NCTLLM001",
                    "score": 88,
                    "trial_relevance": "high",
                    "eligibility_assessment": "likely_eligible",
                    "overall_status": "Recruiting",
                    "trial_status_note": "",
                }
            ],
        )

    def test_llm_judgment_adds_fallback_score_when_response_omits_score(self) -> None:
        chat_client = _FakeEligibilityChatClient(
            """
            {
              "results": [
                {
                  "nct_id": "NCTLLM002",
                  "trial_relevance": "low",
                  "eligibility_assessment": "likely_excluded",
                  "inclusion_matches": [],
                  "exclusion_risks": [],
                  "missing_information": [],
                  "short_reason": "Wrong disease context."
                }
              ]
            }
            """
        )
        bundle = assess_trial_eligibility_candidates(
            structured_case={
                "raw_text": "48 M with critical aortic stenosis.",
                "case_summary": "48 M with critical aortic stenosis.",
                "problem_list": ["critical aortic stenosis"],
            },
            calculation_results=[],
            calculator_matches=[],
            trial_bundle={
                "candidate_ranking": [
                    {
                        "nct_id": "NCTLLM002",
                        "title": "Unrelated Trial",
                        "overall_status": "Recruiting",
                        "enrollment_open": True,
                        "eligibility_inclusion_text": "Adults with unrelated disease.",
                    }
                ]
            },
            limit=1,
            eligibility_chat_client=chat_client,
        )

        llm_result = bundle["assessed_trials"][0]["llm_eligibility_assessment"]
        self.assertEqual(llm_result["score"], 10)
        self.assertEqual(bundle["llm_eligibility_bundle"]["score_summary"][0]["nct_id"], "NCTLLM002")

    def test_llm_judgment_notes_inactive_status_without_changing_score(self) -> None:
        chat_client = _FakeEligibilityChatClient(
            """
            {
              "results": [
                {
                  "nct_id": "NCTLLM003",
                  "score": 78,
                  "trial_relevance": "high",
                  "eligibility_assessment": "possible_with_missing_info",
                  "inclusion_matches": [],
                  "exclusion_risks": [],
                  "missing_information": [],
                  "short_reason": "Patient matches disease and surgery context."
                }
              ]
            }
            """
        )
        bundle = assess_trial_eligibility_candidates(
            structured_case={
                "raw_text": "48 M with critical aortic stenosis undergoing preop workup for valve replacement.",
                "case_summary": "48 M with critical aortic stenosis undergoing preop workup.",
                "problem_list": ["critical aortic stenosis"],
            },
            calculation_results=[],
            calculator_matches=[],
            trial_bundle={
                "candidate_ranking": [
                    {
                        "nct_id": "NCTLLM003",
                        "title": "Withdrawn Aortic Stenosis Trial",
                        "overall_status": "Withdrawn",
                        "enrollment_open": False,
                        "conditions": ["Aortic Valve Stenosis"],
                        "eligibility_inclusion_text": "Scheduled for surgery for aortic stenosis.",
                    }
                ]
            },
            limit=1,
            eligibility_chat_client=chat_client,
        )

        llm_result = bundle["assessed_trials"][0]["llm_eligibility_assessment"]
        self.assertEqual(llm_result["score"], 78)
        self.assertEqual(llm_result["overall_status"], "Withdrawn")
        self.assertIn("not used to lower", llm_result["trial_status_note"])
        self.assertEqual(bundle["llm_eligibility_bundle"]["score_summary"][0]["score"], 78)
        self.assertEqual(
            bundle["llm_eligibility_bundle"]["score_summary"][0]["trial_status_note"],
            llm_result["trial_status_note"],
        )


if __name__ == "__main__":
    unittest.main()
