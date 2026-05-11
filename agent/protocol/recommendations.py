from __future__ import annotations

from typing import Any

from agent.graph.types import GraphState, ProtocolRecommendation, TreatmentRecommendation


def _eligibility_assessment_by_trial(
    eligibility_assessment_bundle: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    assessments: dict[str, dict[str, Any]] = {}
    for item in list((eligibility_assessment_bundle or {}).get("assessed_trials") or []):
        if not isinstance(item, dict):
            continue
        nct_id = str(item.get("nct_id") or "").strip()
        if nct_id:
            assessments[nct_id] = dict(item)
    return assessments


def _coerce_optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _trial_selection_score(candidate: dict[str, Any], assessment: dict[str, Any] | None) -> tuple[int, int, float]:
    status = str(candidate.get("status") or "").strip()
    if status == "abandoned":
        return (0, 0, 0.0)

    aggregate_status = str((assessment or {}).get("aggregate_status") or "").strip()
    eligibility_priority = {
        "likely_eligible": 5,
        "needs_data": 4,
        "evidence_support": 3,
        "ineligible": 1,
        "not_current_option": 0,
    }.get(aggregate_status, 2)
    if aggregate_status in {"ineligible", "not_current_option"}:
        current_priority = 0
    else:
        current_priority = 1
    retrieval_score = _coerce_optional_float(candidate.get("score") or candidate.get("final_score")) or 0.0
    return (current_priority, eligibility_priority, retrieval_score)


def _top_non_abandoned_trial_ids(
    trial_bundle: dict[str, Any],
    *,
    eligibility_assessment_bundle: dict[str, Any] | None = None,
    limit: int = 3,
) -> list[str]:
    assessments = _eligibility_assessment_by_trial(eligibility_assessment_bundle)
    selected_ids: list[str] = []
    for candidate in list(trial_bundle.get("candidate_ranking") or []):
        if str(candidate.get("status") or "").strip() == "abandoned":
            continue
        nct_id = str(candidate.get("nct_id") or "").strip()
        if not nct_id or nct_id in selected_ids:
            continue
        aggregate_status = str(assessments.get(nct_id, {}).get("aggregate_status") or "").strip()
        if aggregate_status in {"ineligible", "not_current_option"}:
            continue
        selected_ids.append(nct_id)
        if len(selected_ids) >= max(int(limit), 1):
            break
    return selected_ids


def _trial_review_actions(trial_bundle: dict[str, Any], *, limit: int = 3) -> list[str]:
    actions: list[str] = []
    for candidate in list(trial_bundle.get("candidate_ranking") or [])[: max(int(limit), 1)]:
        for action in list(candidate.get("actions") or []):
            text = str(action).strip()
            if text and text not in actions:
                actions.append(text)
    return actions


def _normalize_trial_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = [value]
    else:
        try:
            raw_items = list(value)
        except TypeError:
            raw_items = [value]
    deduped: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        normalized = " ".join(str(item or "").split()).strip()
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _trial_candidate_text(candidate: dict[str, Any]) -> str:
    text_parts = [
        str(candidate.get("title") or ""),
        str(candidate.get("brief_summary") or ""),
        str(candidate.get("best_evidence_text") or ""),
        " ; ".join(_normalize_trial_text_list(candidate.get("conditions"))),
        " ; ".join(_normalize_trial_text_list(candidate.get("interventions"))),
        str(candidate.get("primary_purpose") or ""),
    ]
    return " ".join(part for part in text_parts if str(part).strip())


def _candidate_matches_term(candidate: dict[str, Any], term: str) -> bool:
    normalized_term = str(term or "").strip().casefold()
    if not normalized_term:
        return False
    candidate_text = _trial_candidate_text(candidate).casefold()
    return normalized_term in candidate_text


def _select_protocol_trial_candidate(
    trial_bundle: dict[str, Any],
    *,
    eligibility_assessment_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidates = [
        dict(item)
        for item in list(trial_bundle.get("candidate_ranking") or [])
        if isinstance(item, dict)
    ]
    if not candidates:
        return {}
    assessments = _eligibility_assessment_by_trial(eligibility_assessment_bundle)
    ranked = sorted(
        enumerate(candidates),
        key=lambda pair: (
            _trial_selection_score(
                pair[1],
                assessments.get(str(pair[1].get("nct_id") or "").strip()),
            ),
            -pair[0],
        ),
        reverse=True,
    )
    selected = ranked[0][1]
    if _trial_selection_score(
        selected,
        assessments.get(str(selected.get("nct_id") or "").strip()),
    )[0] > 0:
        return selected
    return candidates[0]


def _assess_protocol_trial_candidate(
    candidate: dict[str, Any],
    *,
    query_profile: dict[str, Any],
) -> dict[str, Any]:
    matching_signals: list[str] = []
    conflicts: list[str] = []
    missing_information: list[str] = []

    for term in list(query_profile.get("trial_condition_terms") or []):
        if _candidate_matches_term(candidate, term):
            matching_signals.append(f"Matched condition focus: {term}.")
    for term in list(query_profile.get("trial_intervention_terms") or []):
        if _candidate_matches_term(candidate, term):
            matching_signals.append(f"Matched intervention focus: {term}.")
    for term in list(query_profile.get("trial_intent_terms") or []):
        if _candidate_matches_term(candidate, term):
            matching_signals.append(f"Matched trial intent: {term}.")
        elif term.casefold() == "stroke prevention" and str(candidate.get("primary_purpose") or "").casefold() == "prevention":
            matching_signals.append("Primary purpose aligns with stroke prevention intent.")

    for term in list(query_profile.get("patient_negative_terms") or []):
        if term in {"diabetes", "congestive heart failure"} and _candidate_matches_term(candidate, term):
            conflicts.append(
                f"Trial focus appears to include {term}, which the case explicitly says is absent."
            )

    age_years = _coerce_optional_float(query_profile.get("age_years"))
    age_floor = _coerce_optional_float(candidate.get("age_floor_years"))
    age_ceiling = _coerce_optional_float(candidate.get("age_ceiling_years"))
    if age_years is not None:
        if age_floor is not None and age_years < age_floor:
            conflicts.append(f"Case age {int(age_years)} is below the recorded minimum age {int(age_floor)}.")
        elif age_floor is None:
            missing_information.append("Minimum age is not structured on the trial record.")
        if age_ceiling is not None and age_years > age_ceiling:
            conflicts.append(f"Case age {int(age_years)} is above the recorded maximum age {int(age_ceiling)}.")
        elif age_ceiling is None:
            missing_information.append("Maximum age is not structured on the trial record.")

    patient_gender = str(query_profile.get("gender") or "").strip()
    trial_gender = str(candidate.get("gender") or "").strip()
    if patient_gender:
        if trial_gender and trial_gender not in {"All", patient_gender}:
            conflicts.append(f"Trial gender restriction is {trial_gender}, while the case is {patient_gender}.")
        elif not trial_gender:
            missing_information.append("Trial gender eligibility is not structured on the record.")

    if not str(candidate.get("best_evidence_text") or "").strip():
        missing_information.append("No matched eligibility/evidence chunk was surfaced for manual review.")

    next_checks = _normalize_trial_text_list(candidate.get("actions"))
    if not next_checks:
        next_checks = ["Review the trial inclusion and exclusion criteria against the case manually."]
    if conflicts:
        next_checks.append("Verify whether the apparent mismatch is a true exclusion or just noisy retrieval overlap.")

    fit = "possible_match"
    if str(candidate.get("status") or "").strip() == "abandoned":
        fit = "not_current_option"
    elif conflicts:
        fit = "needs_manual_review"
    elif matching_signals:
        fit = "likely_match"

    return {
        "fit": fit,
        "matching_signals": _normalize_trial_text_list(matching_signals),
        "conflicts": _normalize_trial_text_list(conflicts),
        "missing_information": _normalize_trial_text_list(missing_information),
        "next_checks": _normalize_trial_text_list(next_checks),
    }


def build_protocol_trial_selection(
    trial_bundle: dict[str, Any],
    *,
    structured_case: dict[str, Any],
    eligibility_assessment_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del structured_case
    assessments = _eligibility_assessment_by_trial(eligibility_assessment_bundle)
    selected_trial = _select_protocol_trial_candidate(
        trial_bundle,
        eligibility_assessment_bundle=eligibility_assessment_bundle,
    )
    if not selected_trial:
        return {
            "selected_trial": None,
            "selection_reason": "No trial candidate survived retrieval for the current case.",
            "eligibility_assessment": {
                "fit": "not_available",
                "matching_signals": [],
                "conflicts": [],
                "missing_information": ["No ranked trial candidates were returned."],
                "next_checks": ["Adjust the protocol query profile or broaden the trial corpus before retrying."],
            },
            "trial_status_assessment": {},
            "evidence": [],
            "alternatives": [],
        }

    query_profile = dict(trial_bundle.get("query_profile") or {})
    eligibility_assessment = _assess_protocol_trial_candidate(
        selected_trial,
        query_profile=query_profile,
    )
    selected_nct_id = str(selected_trial.get("nct_id") or "").strip()
    criterion_assessment = assessments.get(selected_nct_id, {})
    if criterion_assessment:
        eligibility_assessment["criterion_level_assessment"] = criterion_assessment
        aggregate_status = str(criterion_assessment.get("aggregate_status") or "").strip()
        if aggregate_status:
            eligibility_assessment["aggregate_status"] = aggregate_status
        if aggregate_status == "likely_eligible":
            eligibility_assessment["fit"] = "likely_match"
            eligibility_assessment["matching_signals"] = _normalize_trial_text_list(
                list(eligibility_assessment.get("matching_signals") or [])
                + ["Criterion-level eligibility assessment did not find blocking criteria."]
            )
        elif aggregate_status == "needs_data":
            eligibility_assessment["fit"] = "needs_manual_review"
            eligibility_assessment["missing_information"] = _normalize_trial_text_list(
                list(eligibility_assessment.get("missing_information") or [])
                + [
                    str(question.get("question") or "").strip()
                    for question in list(criterion_assessment.get("missing_questions") or [])
                    if isinstance(question, dict)
                ]
            )
        elif aggregate_status in {"ineligible", "not_current_option"}:
            eligibility_assessment["fit"] = aggregate_status
            eligibility_assessment["conflicts"] = _normalize_trial_text_list(
                list(eligibility_assessment.get("conflicts") or [])
                + list(criterion_assessment.get("blocking_criteria") or [])
            )
    status_assessment = {
        "status": str(selected_trial.get("status") or ""),
        "overall_status": str(selected_trial.get("overall_status") or ""),
        "enrollment_open": bool(selected_trial.get("enrollment_open")),
        "status_reason": str(selected_trial.get("status_reason") or ""),
    }

    evidence: list[str] = []
    for item in list(eligibility_assessment.get("matching_signals") or []):
        if item not in evidence:
            evidence.append(str(item))
    best_evidence_text = str(selected_trial.get("best_evidence_text") or "").strip()
    if best_evidence_text:
        evidence.append(best_evidence_text[:240])
    brief_summary = str(selected_trial.get("brief_summary") or "").strip()
    if brief_summary:
        evidence.append(brief_summary[:240])

    focus_terms = _normalize_trial_text_list(
        list(query_profile.get("trial_condition_terms") or [])
        + list(query_profile.get("trial_intent_terms") or [])
    )
    selection_reason = (
        f"Selected {selected_trial.get('title') or selected_trial.get('nct_id') or 'the top-ranked trial'} "
        f"because it remained the highest-ranked candidate after retrieval and criterion-level eligibility screening, "
        f"and matched the protocol focus terms "
        f"{', '.join(focus_terms[:3]) or 'from the structured case'}."
    )
    if criterion_assessment.get("aggregate_status"):
        selection_reason += f" Eligibility aggregate: {criterion_assessment['aggregate_status']}."
    if status_assessment["overall_status"]:
        selection_reason += f" Current study status: {status_assessment['overall_status']}."

    alternatives: list[dict[str, Any]] = []
    for candidate in list(trial_bundle.get("candidate_ranking") or []):
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get("nct_id") or "").strip() == str(selected_trial.get("nct_id") or "").strip():
            continue
        alternatives.append(
            {
                "nct_id": str(candidate.get("nct_id") or ""),
                "title": str(candidate.get("title") or ""),
                "status": str(candidate.get("status") or ""),
                "overall_status": str(candidate.get("overall_status") or ""),
                "eligibility_aggregate_status": str(
                    assessments.get(str(candidate.get("nct_id") or "").strip(), {}).get("aggregate_status") or ""
                ),
                "why_not_selected": (
                    "Not selected because criterion-level eligibility found a blocking or non-current status."
                    if str(
                        assessments.get(str(candidate.get("nct_id") or "").strip(), {}).get("aggregate_status") or ""
                    )
                    in {"ineligible", "not_current_option"}
                    else (
                        "Lower final ranking or weaker fit than the selected trial."
                        if str(candidate.get("status") or "").strip() != "abandoned"
                        else "Not selected because the trial is not a current option."
                    )
                ),
            }
        )
        if len(alternatives) >= 3:
            break

    return {
        "selected_trial": dict(selected_trial),
        "selection_reason": selection_reason,
        "eligibility_assessment": eligibility_assessment,
        "trial_status_assessment": status_assessment,
        "evidence": evidence[:5],
        "alternatives": alternatives,
    }


def _augment_treatment_recommendations_with_trials(
    recommendations: list[TreatmentRecommendation],
    trial_bundle: dict[str, Any],
    *,
    eligibility_assessment_bundle: dict[str, Any] | None = None,
    has_completed_results: bool,
) -> list[TreatmentRecommendation]:
    top_trial_ids = _top_non_abandoned_trial_ids(
        trial_bundle,
        eligibility_assessment_bundle=eligibility_assessment_bundle,
        limit=3,
    )
    if has_completed_results and top_trial_ids:
        for recommendation in recommendations:
            recommendation.linked_trials = list(top_trial_ids)

    if has_completed_results or not list(trial_bundle.get("candidate_ranking") or []):
        return recommendations

    candidate_ranking = list(trial_bundle.get("candidate_ranking") or [])
    top_candidate = dict(candidate_ranking[0] or {}) if candidate_ranking else {}
    trial_review_actions = _trial_review_actions(trial_bundle, limit=3)
    if "Review trial eligibility and enrollment details before surfacing any specific study." not in trial_review_actions:
        trial_review_actions.append(
            "Review trial eligibility and enrollment details before surfacing any specific study."
        )
    recommendations.insert(
        0,
        TreatmentRecommendation(
            name="trial candidate review",
            strategy="trial_candidate_review",
            source="trial_retrieval",
            status="manual_review",
            rationale=(
                "Local trial retrieval surfaced candidate studies for the current case, but there is no usable "
                "calculator-backed risk output strong enough to promote a direct trial match. "
                f"Top candidate: {top_candidate.get('title') or top_candidate.get('name') or top_candidate.get('nct_id') or 'unknown trial'}."
            ),
            linked_trials=list(top_trial_ids),
            actions=trial_review_actions,
        ),
    )
    return recommendations


def build_treatment_recommendations(
    state: GraphState,
    *,
    trial_bundle: dict[str, Any] | None = None,
    eligibility_assessment_bundle: dict[str, Any] | None = None,
) -> list[TreatmentRecommendation]:
    completed_results = [item for item in state.calculation_results if item.status == "completed"]
    partial_results = [item for item in state.calculation_results if item.status == "partial"]
    estimated_results = [item for item in state.calculation_results if item.status == "estimated"]
    effective_trial_bundle = dict(trial_bundle or {})

    if completed_results:
        recommendations: list[TreatmentRecommendation] = []
        for artifact in completed_results[:3]:
            recommendations.append(
                TreatmentRecommendation(
                    name=f"{artifact.name} guided treatment",
                    strategy="risk_informed_treatment",
                    source="protocol_reasoning",
                    status="manual_review",
                    rationale=(
                        "A usable risk result is available, so protocol can now derive a treatment direction. "
                        "Any final trial or regimen match still needs explicit downstream evidence."
                    ),
                    linked_calculators=[artifact.linked_calculator] if artifact.linked_calculator else [],
                    actions=[
                        "Map the risk output to treatment thresholds or protocol branches.",
                        "Keep any regimen or trial recommendation evidence-linked and explicitly reviewable.",
                    ],
                )
            )
        return _augment_treatment_recommendations_with_trials(
            recommendations,
            effective_trial_bundle,
            eligibility_assessment_bundle=eligibility_assessment_bundle,
            has_completed_results=True,
        )

    if partial_results:
        recommendations = [
            TreatmentRecommendation(
                name="partial calculator result requires parameter completion",
                strategy="similar_case_fallback",
                source="partial_parameter_gap",
                status="similar_case_fallback",
                rationale=(
                    "A calculator produced only a provisional result because key parameters are still missing, "
                    "so treatment and trial routing should remain provisional until those inputs are completed."
                ),
                linked_calculators=[
                    artifact.linked_calculator for artifact in partial_results if artifact.linked_calculator
                ],
                actions=[
                    "Collect the listed missing calculator inputs before treating the score as final.",
                    "Use the current calculator text only as a provisional lower-bound or partial interpretation.",
                ],
            )
        ]
        return _augment_treatment_recommendations_with_trials(
            recommendations,
            effective_trial_bundle,
            eligibility_assessment_bundle=eligibility_assessment_bundle,
            has_completed_results=False,
        )

    if estimated_results:
        recommendations = [
            TreatmentRecommendation(
                name="similar-case treatment fallback",
                strategy="similar_case_fallback",
                source="estimated_parameter_gap",
                status="similar_case_fallback",
                rationale=(
                    "Calculation is close but not fully executable, so treatment should be chosen only after "
                    "validating the estimated parameter against similar cases."
                ),
                linked_calculators=[
                    artifact.linked_calculator for artifact in estimated_results if artifact.linked_calculator
                ],
                actions=[
                    "Review the estimated parameter against one or more similar cases.",
                    "Keep the treatment recommendation provisional until the missing value is validated.",
                ],
            )
        ]
        return _augment_treatment_recommendations_with_trials(
            recommendations,
            effective_trial_bundle,
            eligibility_assessment_bundle=eligibility_assessment_bundle,
            has_completed_results=False,
        )

    if state.calculator_matches:
        recommendations = [
            TreatmentRecommendation(
                name="similar-case assisted recommendation",
                strategy="similar_case_fallback",
                source="calculator_candidates_without_execution",
                status="similar_case_fallback",
                rationale=(
                    "Candidate calculators were found, but no risk value was strong enough to anchor treatment. "
                    "A clinician should review similar cases and trial options directly."
                ),
                linked_calculators=[match.pmid for match in state.calculator_matches[:3] if match.pmid],
                actions=[
                    "Use retrieved candidate calculators and their eligibility notes as a screening aid.",
                    "Search similar cases or evidence sources before escalating treatment.",
                ],
            )
        ]
        return _augment_treatment_recommendations_with_trials(
            recommendations,
            effective_trial_bundle,
            eligibility_assessment_bundle=eligibility_assessment_bundle,
            has_completed_results=False,
        )

    recommendations = [
        TreatmentRecommendation(
            name="direct treatment advice",
            strategy="direct_advice",
            source="no_calculation_signal",
            status="advice_only",
            rationale=(
                "Neither executable calculators nor valid fallback signals are available, so the graph can only "
                "return a clinician-reviewed recommendation path."
            ),
            actions=[
                "Collect more structured parameters before rerunning MedAI.",
                "If treatment must proceed now, provide direct conservative treatment advice and mark it as low confidence.",
            ],
        )
    ]
    return _augment_treatment_recommendations_with_trials(
        recommendations,
        effective_trial_bundle,
        eligibility_assessment_bundle=eligibility_assessment_bundle,
        has_completed_results=False,
    )


def to_protocol_recommendations(
    recommendations: list[TreatmentRecommendation],
) -> list[ProtocolRecommendation]:
    protocol_recommendations: list[ProtocolRecommendation] = []
    for recommendation in recommendations:
        if recommendation.status in {"matched", "trial_matched"}:
            status = "matched"
        elif recommendation.status == "abandoned":
            status = "insufficient_data"
        else:
            status = "needs_revision"
        protocol_recommendations.append(
            ProtocolRecommendation(
                name=recommendation.name,
                category=recommendation.strategy,
                status=status,
                rationale=recommendation.rationale,
                linked_calculators=list(recommendation.linked_calculators),
                linked_trials=list(recommendation.linked_trials),
                corrections=list(recommendation.actions),
            )
        )
    return protocol_recommendations
