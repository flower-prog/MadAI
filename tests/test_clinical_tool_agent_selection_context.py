from __future__ import annotations

import unittest

from agent.clinical_tool_agent import ClinicalToolAgent
from agent.graph.types import ClinicalToolJob


class _DummyDocument:
    def __init__(self, pmid: str) -> None:
        self.pmid = pmid
        self.title = f"Calculator {pmid}"
        self.purpose = f"Purpose {pmid}"
        self.eligibility = "Eligible patients."

    def to_dict(self) -> dict[str, str]:
        return {
            "pmid": self.pmid,
            "title": self.title,
            "purpose": self.purpose,
            "eligibility": self.eligibility,
        }


class _MinimalCatalog:
    def get(self, pmid: str) -> _DummyDocument:
        return _DummyDocument(str(pmid))


class _MinimalRetriever:
    def retrieve(self, query: str, top_k: int = 10, *, candidate_pmids=None, backend=None):
        return []


class _MinimalChatClient:
    def complete(self, messages, model=None, temperature=0.0):
        return ""


def _build_candidate_rows(start: int, stop: int, *, channel: str) -> list[dict[str, object]]:
    return [
        {
            "pmid": str(index),
            "title": f"{channel.upper()} Calculator {index}",
            "purpose": f"{channel.upper()} purpose {index}",
            "parameter_names": [f"feature_{index}"],
            "source_channels": [channel],
        }
        for index in range(start, stop)
    ]


def _build_raw_rows(start: int, stop: int, *, channel: str) -> list[dict[str, object]]:
    return [
        {
            "rank": index - start + 1,
            "channel": channel,
            "pmid": str(index),
            "calculator_payload": {
                "pmid": str(index),
                "title": f"{channel.upper()} Calculator {index}",
                "purpose": f"{channel.upper()} purpose {index}",
                "eligibility": "Eligible patients.",
            },
        }
        for index in range(start, stop)
    ]


class ClinicalToolAgentSelectionContextTests(unittest.TestCase):
    def _build_agent(self) -> ClinicalToolAgent:
        return ClinicalToolAgent(
            catalog=_MinimalCatalog(),
            retriever=_MinimalRetriever(),
            chat_client=_MinimalChatClient(),
        )

    def test_run_question_limits_selection_context_to_second_stage_six_candidates(self) -> None:
        agent = self._build_agent()
        job = ClinicalToolJob(
            mode="question",
            text="Which calculator should be selected for this case?",
            top_k=5,
        )
        agent._retrieve_and_rank_candidates = lambda _: {
            "retrieved_tools": _build_candidate_rows(1, 6, channel="bm25"),
            "bm25_raw_top5": _build_raw_rows(1, 6, channel="bm25"),
            "vector_raw_top5": _build_raw_rows(6, 11, channel="vector"),
            "retrieval_batches": [],
            "risk_hints": [],
        }
        agent._assess_candidate_execution_gate = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("selection should not pre-run candidate execution gates")
        )

        seen_pmids: list[list[str]] = []

        def _select_tool(current_job, retrieved, **kwargs):
            seen_pmids.append([str(item.get("pmid")) for item in retrieved])
            return {
                "pmid": "6",
                "title": "VECTOR Calculator 6",
                "purpose": "VECTOR purpose 6",
                "parameter_names": ["feature_6"],
                "reason": "selected from combined top5 context",
                "raw_response": "",
            }

        agent._select_tool_for_question = _select_tool
        agent._execute_calculator = lambda current_job, pmid, **kwargs: {
            "pmid": pmid,
            "status": "completed",
            "final_text": f"Executed {pmid}",
        }

        result = agent._run_question(job)

        self.assertEqual(len(seen_pmids), 1)
        self.assertEqual(len(seen_pmids[0]), 6)
        self.assertEqual({str(index) for index in range(1, 7)}, set(seen_pmids[0]))
        self.assertEqual(len(result["retrieved_tools"]), 6)
        self.assertEqual(result["selected_tool"]["pmid"], "6")

    def test_run_patient_note_limits_selection_context_to_second_stage_six_candidates(self) -> None:
        agent = self._build_agent()
        job = ClinicalToolJob(
            mode="patient_note",
            text="Patient note asking for a calculator.",
            top_k=5,
            max_selected_tools=5,
        )
        agent._retrieve_and_rank_candidates = lambda _: {
            "retrieved_tools": _build_candidate_rows(1, 6, channel="bm25"),
            "bm25_raw_top5": _build_raw_rows(1, 6, channel="bm25"),
            "vector_raw_top5": _build_raw_rows(6, 11, channel="vector"),
            "retrieval_batches": [],
            "risk_hints": ["stroke risk"],
        }
        agent._assess_candidate_execution_gate = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("selection should not pre-run candidate execution gates")
        )

        seen_pmids: list[list[str]] = []
        executed_pmids: list[str] = []

        def _select_tool(current_job, retrieved, **kwargs):
            seen_pmids.append([str(item.get("pmid")) for item in retrieved])
            return {
                "pmid": "6",
                "title": "VECTOR Calculator 6",
                "purpose": "VECTOR purpose 6",
                "parameter_names": ["feature_6"],
                "reason": "selected from combined top5 context",
                "raw_response": "",
            }

        agent._select_tool_for_question = _select_tool
        agent._execute_calculator = lambda current_job, pmid, **kwargs: executed_pmids.append(pmid) or {
            "pmid": pmid,
            "status": "completed",
            "final_text": f"Executed {pmid}",
        }

        result = agent._run_patient_note(job)

        self.assertEqual(len(seen_pmids), 1)
        self.assertEqual(len(seen_pmids[0]), 6)
        self.assertEqual({str(index) for index in range(1, 7)}, set(seen_pmids[0]))
        self.assertEqual(len(result["retrieved_tools"]), 6)
        self.assertEqual(executed_pmids, ["6"])
        self.assertEqual(result["selected_tool"]["pmid"], "6")


if __name__ == "__main__":
    unittest.main()
