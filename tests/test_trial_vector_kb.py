from __future__ import annotations

import json
import shutil
import sys
import unittest
import uuid
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agent.trial_vector_kb import build_trial_chunks, build_trial_record_from_xml_bytes, build_trial_vector_kb


def _trial_xml(
    *,
    nct_id: str,
    title: str,
    summary: str,
    detailed_description: str,
    overall_status: str,
    download_date: str,
    source_url: str,
    intervention_name: str,
) -> str:
    return "\n".join(
        [
            "<clinical_study>",
            "  <required_header>",
            f"    <download_date>{download_date}</download_date>",
            f"    <url>{source_url}</url>",
            "  </required_header>",
            "  <id_info>",
            f"    <nct_id>{nct_id}</nct_id>",
            "  </id_info>",
            f"  <brief_title>{title}</brief_title>",
            f"  <official_title>{title} Official</official_title>",
            "  <acronym>TEST</acronym>",
            "  <brief_summary>",
            f"    <textblock>{summary}</textblock>",
            "  </brief_summary>",
            "  <detailed_description>",
            f"    <textblock>{detailed_description}</textblock>",
            "  </detailed_description>",
            f"  <overall_status>{overall_status}</overall_status>",
            "  <study_type>Interventional</study_type>",
            "  <phase>Phase 2</phase>",
            "  <study_design_info>",
            "    <primary_purpose>Treatment</primary_purpose>",
            "    <intervention_model>Parallel Assignment</intervention_model>",
            "    <allocation>Randomized</allocation>",
            "    <masking>Double</masking>",
            "  </study_design_info>",
            "  <enrollment type=\"Actual\">120</enrollment>",
            "  <condition>Melanoma</condition>",
            "  <condition>Skin Cancer</condition>",
            "  <keyword>melanoma</keyword>",
            "  <keyword>immunotherapy</keyword>",
            "  <condition_browse>",
            "    <mesh_term>Melanoma</mesh_term>",
            "  </condition_browse>",
            "  <intervention>",
            "    <intervention_type>Drug</intervention_type>",
            f"    <intervention_name>{intervention_name}</intervention_name>",
            "    <description>Pembrolizumab intravenous infusion.</description>",
            "  </intervention>",
            "  <intervention_browse>",
            "    <mesh_term>Pembrolizumab</mesh_term>",
            "  </intervention_browse>",
            "  <arm_group>",
            "    <arm_group_label>Experimental Arm</arm_group_label>",
            "    <arm_group_type>Experimental</arm_group_type>",
            "    <description>Pembrolizumab monotherapy arm.</description>",
            "  </arm_group>",
            "  <eligibility>",
            "    <criteria>",
            "      <textblock>Inclusion Criteria: - Histologically confirmed melanoma - ECOG 0-1 Exclusion Criteria: - Active autoimmune disease - Uncontrolled infection</textblock>",
            "    </criteria>",
            "    <gender>All</gender>",
            "    <minimum_age>18 Years</minimum_age>",
            "    <maximum_age>75 Years</maximum_age>",
            "    <healthy_volunteers>No</healthy_volunteers>",
            "  </eligibility>",
            "  <primary_outcome>",
            "    <measure>Objective response rate</measure>",
            "    <description>RECIST response.</description>",
            "    <time_frame>12 weeks</time_frame>",
            "  </primary_outcome>",
            "  <secondary_outcome>",
            "    <measure>Progression-free survival</measure>",
            "    <description>PFS time.</description>",
            "    <time_frame>24 months</time_frame>",
            "  </secondary_outcome>",
            "  <reference>",
            "    <citation>Key melanoma trial publication.</citation>",
            "    <PMID>12345678</PMID>",
            "  </reference>",
            "  <start_date>January 2023</start_date>",
            "  <completion_date>December 2024</completion_date>",
            "  <primary_completion_date>September 2024</primary_completion_date>",
            "  <study_first_submitted>January 10, 2023</study_first_submitted>",
            "  <study_first_posted>January 20, 2023</study_first_posted>",
            "  <last_update_submitted>May 1, 2024</last_update_submitted>",
            "  <last_update_posted>May 15, 2024</last_update_posted>",
            "  <location>",
            "    <facility>",
            "      <address>",
            "        <city>Boston</city>",
            "        <state>Massachusetts</state>",
            "        <country>United States</country>",
            "      </address>",
            "    </facility>",
            "  </location>",
            "  <location_countries>",
            "    <country>United States</country>",
            "  </location_countries>",
            "</clinical_study>",
            "",
        ]
    )


class TrialVectorKbBuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = PROJECT_ROOT / ".tmp_test_artifacts" / f"trial-vector-kb-{uuid.uuid4().hex}"
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_build_trial_record_extracts_schema_fields_and_chunks(self) -> None:
        xml_bytes = _trial_xml(
            nct_id="NCT99900001",
            title="Pembrolizumab for Metastatic Melanoma",
            summary="Evaluate pembrolizumab in metastatic melanoma.",
            detailed_description="This is a detailed trial description. Patients receive pembrolizumab every 3 weeks.",
            overall_status="Recruiting",
            download_date="ClinicalTrials.gov processed this data on May 8, 2023",
            source_url="https://clinicaltrials.gov/show/NCT99900001",
            intervention_name="Pembrolizumab",
        ).encode("utf-8")

        record = build_trial_record_from_xml_bytes(
            xml_bytes,
            source_corpus="totrials_2023",
            source_archive="ClinicalTrials.2023-05-08.trials0.zip",
            source_member_path="NCT9990xxxx/NCT99900001.xml",
        )
        chunks = build_trial_chunks(record, chunk_char_limit=500)

        self.assertEqual(record["nct_id"], "NCT99900001")
        self.assertEqual(record["source_corpus"], "totrials_2023")
        self.assertEqual(record["source_archive"], "ClinicalTrials.2023-05-08.trials0.zip")
        self.assertEqual(record["display_title"], "Pembrolizumab for Metastatic Melanoma")
        self.assertEqual(record["source_url"], "https://clinicaltrials.gov/show/NCT99900001")
        self.assertEqual(record["age_floor_years"], 18.0)
        self.assertEqual(record["age_ceiling_years"], 75.0)
        self.assertIn("Melanoma", record["condition_terms"])
        self.assertIn("Pembrolizumab", record["intervention_terms"])
        self.assertIn("Histologically confirmed melanoma", record["eligibility_inclusion_text"])
        self.assertIn("Active autoimmune disease", record["eligibility_exclusion_text"])
        self.assertIn("title: Pembrolizumab for Metastatic Melanoma", record["overview_text"])

        chunk_types = {chunk["chunk_type"] for chunk in chunks}
        self.assertEqual(
            chunk_types,
            {
                "overview",
                "description",
                "eligibility_inclusion",
                "eligibility_exclusion",
                "outcomes",
                "arms_interventions",
            },
        )
        self.assertTrue(any(chunk["chunk_id"] == "NCT99900001::overview::0" for chunk in chunks))

    def test_build_trial_vector_kb_prefers_newer_snapshot_for_duplicate_nct_id(self) -> None:
        corpus_2021 = self.temp_root / "数据" / "totrials" / "corpus_2021_2022"
        corpus_2023 = self.temp_root / "数据" / "totrials" / "corpus_2023"
        corpus_2021.mkdir(parents=True, exist_ok=True)
        corpus_2023.mkdir(parents=True, exist_ok=True)

        older_zip = corpus_2021 / "ClinicalTrials.2021-04-27.part1.zip"
        newer_zip = corpus_2023 / "ClinicalTrials.2023-05-08.trials0.zip"

        with zipfile.ZipFile(older_zip, "w") as archive:
            archive.writestr(
                "NCT1111xxxx/NCT11110001.xml",
                _trial_xml(
                    nct_id="NCT11110001",
                    title="Older Snapshot Trial",
                    summary="Older summary.",
                    detailed_description="Older description.",
                    overall_status="Completed",
                    download_date="ClinicalTrials.gov processed this data on April 27, 2021",
                    source_url="https://clinicaltrials.gov/show/NCT11110001",
                    intervention_name="OlderDrug",
                ),
            )

        with zipfile.ZipFile(newer_zip, "w") as archive:
            archive.writestr(
                "NCT1111xxxx/NCT11110001.xml",
                _trial_xml(
                    nct_id="NCT11110001",
                    title="Newer Snapshot Trial",
                    summary="Newer summary.",
                    detailed_description="Newer description.",
                    overall_status="Recruiting",
                    download_date="ClinicalTrials.gov processed this data on May 8, 2023",
                    source_url="https://clinicaltrials.gov/show/NCT11110001",
                    intervention_name="NewerDrug",
                ),
            )

        output_dir = self.temp_root / "outputs" / "trial_vector_kb"
        manifest = build_trial_vector_kb(
            input_paths=[corpus_2021, corpus_2023],
            output_dir=output_dir,
            chunk_char_limit=500,
        )

        self.assertEqual(manifest["trial_record_count"], 1)
        self.assertEqual(manifest["duplicate_candidate_count"], 1)

        record_rows = [
            json.loads(line)
            for line in (output_dir / "trial_record.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(record_rows), 1)
        self.assertEqual(record_rows[0]["source_corpus"], "totrials_2023")
        self.assertEqual(record_rows[0]["brief_title"], "Newer Snapshot Trial")
        self.assertEqual(record_rows[0]["overall_status"], "Recruiting")

        chunk_rows = [
            json.loads(line)
            for line in (output_dir / "trial_chunk.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertTrue(chunk_rows)


if __name__ == "__main__":
    unittest.main()
