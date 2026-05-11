from __future__ import annotations

from .types import CriterionAssessment, MissingDataQuestion


def generate_missing_questions(assessments: list[CriterionAssessment]) -> list[MissingDataQuestion]:
    questions_by_key: dict[str, MissingDataQuestion] = {}
    for item in list(assessments or []):
        if item.label != "unknown":
            continue
        condition = (item.condition or item.required_evidence_type or "eligibility").strip()
        key = condition.casefold()
        if key in questions_by_key:
            questions_by_key[key].linked_criteria.append(item.criterion_id)
            for missing_item in item.missing_data:
                if missing_item not in questions_by_key[key].required_data:
                    questions_by_key[key].required_data.append(missing_item)
            continue

        if key in {"ecog"}:
            question = "请确认患者的 ECOG 体能状态评分。"
        elif "platelet" in key or "bilirubin" in key or key in {"anc", "hemoglobin", "ast", "alt", "creatinine clearance", "crcl"}:
            question = f"请提供最近一次 {condition} 数值和采样日期，优先提供方案要求时间窗内的结果。"
        elif "cns" in key or "brain metast" in key:
            question = "请确认患者是否存在未经治疗或不稳定的 CNS/脑转移。"
        elif "pregnan" in key:
            question = "如临床适用，请确认妊娠状态。"
        elif "sex" in key:
            question = "请确认患者性别。"
        elif "age" in key:
            question = "请确认患者年龄。"
        else:
            question = f"请补充可判断该入排标准的信息：{item.raw_text}"

        priority = "high" if item.type == "exclusion" or key in {"ecog", "age", "sex"} else "medium"
        questions_by_key[key] = MissingDataQuestion(
            question_id=f"missing::{len(questions_by_key) + 1:03d}",
            priority=priority,  # type: ignore[arg-type]
            question=question,
            required_data=list(item.missing_data or [condition]),
            linked_criteria=[item.criterion_id],
        )

    return list(questions_by_key.values())
