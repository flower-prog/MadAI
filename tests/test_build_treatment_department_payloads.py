from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def _load_module():
    script_path = Path("/home/yuanzy/MadAI/scripts/build_treatment_department_payloads.py")
    spec = importlib.util.spec_from_file_location("build_treatment_department_payloads", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_xml(path: Path, *, nct_id: str, title: str, overall_status: str) -> None:
    path.write_text(
        "\n".join(
            [
                "<clinical_study>",
                "  <id_info>",
                f"    <nct_id>{nct_id}</nct_id>",
                "  </id_info>",
                f"  <brief_title>{title}</brief_title>",
                f"  <overall_status>{overall_status}</overall_status>",
                "  <study_type>Interventional</study_type>",
                "  <phase>Phase 2</phase>",
                "  <study_design_info>",
                "    <primary_purpose>Treatment</primary_purpose>",
                "  </study_design_info>",
                "  <condition>Condition A</condition>",
                "  <intervention>",
                "    <intervention_name>Drug A</intervention_name>",
                "  </intervention>",
                "  <brief_summary>",
                "    <textblock>Summary text.</textblock>",
                "  </brief_summary>",
                "  <eligibility>",
                "    <criteria>",
                "      <textblock>Eligibility text.</textblock>",
                "    </criteria>",
                "  </eligibility>",
                "</clinical_study>",
                "",
            ]
        ),
        encoding="utf-8",
    )


class BuildTreatmentDepartmentPayloadsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_module()

    def test_map_trial_status_handles_completed_and_unknown(self) -> None:
        completed = self.module.map_trial_status("Completed")
        self.assertEqual(completed["status"], "trial_matched")
        self.assertFalse(completed["enrollment_open"])

        unknown = self.module.map_trial_status("Unknown status")
        self.assertEqual(unknown["status"], "manual_review")

    def test_build_package_writes_department_payloads_for_all_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            package_dir = Path(tmp_dir) / "治疗方案"
            xml_dir = package_dir / "xml"
            xml_dir.mkdir(parents=True, exist_ok=True)

            (package_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
            (package_dir / "README.md").write_text("# temp\n", encoding="utf-8")
            (package_dir / "medai_index.tsv").write_text(
                "\n".join(
                    [
                        "sample_index\tnct_id\tprimary_department\tsecondary_departments\tcopied_xml_relpath\tsource_xml_path",
                        "1\tNCT00000001\t内科\t儿科\txml/NCT00000001.xml\t/source/NCT00000001.xml",
                        "2\tNCT00000002\t肿瘤科\t\txml/NCT00000002.xml\t/source/NCT00000002.xml",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            (package_dir / "medai_smoke.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "sample_index": 1,
                                "nct_id": "NCT00000001",
                                "primary_department": "内科",
                                "secondary_departments": ["儿科"],
                                "copied_xml_relpath": "xml/NCT00000001.xml",
                                "source_xml_path": "/source/NCT00000001.xml",
                                "xml_content": "<clinical_study />",
                                "xml_size_bytes": 17,
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "sample_index": 2,
                                "nct_id": "NCT00000002",
                                "primary_department": "肿瘤科",
                                "secondary_departments": [],
                                "copied_xml_relpath": "xml/NCT00000002.xml",
                                "source_xml_path": "/source/NCT00000002.xml",
                                "xml_content": "<clinical_study />",
                                "xml_size_bytes": 17,
                            },
                            ensure_ascii=False,
                        ),
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            _write_xml(
                xml_dir / "NCT00000001.xml",
                nct_id="NCT00000001",
                title="Completed Trial",
                overall_status="Completed",
            )
            _write_xml(
                xml_dir / "NCT00000002.xml",
                nct_id="NCT00000002",
                title="Terminated Trial",
                overall_status="Terminated",
            )

            result = self.module.build_package(package_dir)
            self.assertEqual(result["sample_size"], 2)
            self.assertEqual(result["medai_status_distribution"]["trial_matched"], 1)
            self.assertEqual(result["medai_status_distribution"]["abandoned"], 1)

            internal_payload = json.loads(
                (package_dir / "内科" / "treatment_trials.json").read_text(encoding="utf-8")
            )
            pediatric_payload = json.loads(
                (package_dir / "儿科" / "treatment_trials.json").read_text(encoding="utf-8")
            )
            oncology_payload = json.loads(
                (package_dir / "肿瘤科" / "treatment_trials.json").read_text(encoding="utf-8")
            )

            self.assertEqual(internal_payload["NCT00000001"]["status"], "trial_matched")
            self.assertEqual(internal_payload["NCT00000001"]["department_role"], "primary")
            self.assertEqual(pediatric_payload["NCT00000001"]["department_role"], "secondary")
            self.assertEqual(oncology_payload["NCT00000002"]["status"], "abandoned")


if __name__ == "__main__":
    unittest.main()
