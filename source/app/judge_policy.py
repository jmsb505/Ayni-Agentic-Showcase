from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from enum import Enum
from typing import Protocol

from app.connectors.ports import (
    JudgeChoiceAnswer,
    JudgeEvaluationRequest,
    JudgeEvaluationResult,
)
from app.domain import JsonObject, RunEventType


class JudgeMode(str, Enum):
    OFF = "off"
    OBSERVE = "observe"
    ASSIST = "assist"
    AUTONOMOUS = "autonomous"


class JudgeEvaluator(Protocol):
    def evaluate(
        self,
        request: JudgeEvaluationRequest,
        *,
        deployment_id: str | None = None,
    ) -> JudgeEvaluationResult: ...


ExecutionRecorder = Callable[[RunEventType, JsonObject], None]


def evaluate_judgment(
    mode: JudgeMode,
    evaluator: JudgeEvaluator | None,
    request: JudgeEvaluationRequest,
    *,
    decision_id: str,
    deployment_id: str,
    record: ExecutionRecorder | None,
    requested_label: str,
    completed_label: str,
) -> JudgeEvaluationResult | None:
    if mode is JudgeMode.OFF or evaluator is None or record is None:
        return None

    request_data = request.model_dump(mode="json")
    state_digest = hashlib.sha256(
        json.dumps(request.state, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()
    event_context = {
        "tool_name": "judge.evaluate",
        "decision_id": decision_id,
        "mode": mode.value,
        "state_digest": state_digest,
    }
    record(
        RunEventType.TOOL_REQUESTED,
        {
            **event_context,
            "display_label": requested_label,
            "request": request_data,
        },
    )
    try:
        result = evaluator.evaluate(request, deployment_id=deployment_id)
    except Exception as error:
        failure_code = getattr(error, "code", None)
        if not isinstance(failure_code, str) or not failure_code:
            failure_code = "judge_unavailable"
        record(
            RunEventType.TOOL_FAILED,
            {
                **event_context,
                "display_label": "Judge unavailable; no decision returned",
                "failure_code": failure_code[:120],
            },
        )
        return None

    record(
        RunEventType.TOOL_COMPLETED,
        {
            **event_context,
            "display_label": completed_label,
            "result": result.model_dump(mode="json"),
        },
    )
    return result


def answer_confidence(answer) -> float:
    if answer.confidence is not None:
        return answer.confidence
    return answer.probabilities.get(answer.choice, 0.0)


COMPANY_PROFILE_FIELD_REVIEW_CONFIDENCE = 0.80
COMPANY_PROFILE_FIELD_DECISIONS = frozenset(
    {"acceptable", "revise_missing", "revise_unsupported", "insufficient_evidence"}
)


def assess_company_profile_result(
    result: JudgeEvaluationResult,
    field_question_ids: tuple[str, ...],
) -> tuple[str, float, dict[str, tuple[str, float]], list[str]]:
    """Apply deterministic review policy to a complete typed Judge assessment.

    Jev only classifies each area. This function applies the confidence floor,
    derives the aggregate decision, and returns the only areas eligible for the
    single bounded correction pass. Missing or uncertain checks never become ready.
    """
    overall = result.answers.get("profile_review")
    if not isinstance(overall, JudgeChoiceAnswer) or overall.choice not in {
        "ready",
        "needs_revision",
        "insufficient_evidence",
    }:
        return "insufficient_evidence", 0.0, {}, []

    overall_confidence = answer_confidence(overall)
    assessments: dict[str, tuple[str, float]] = {}
    for question_id in field_question_ids:
        answer = result.answers.get(question_id)
        if (
            not isinstance(answer, JudgeChoiceAnswer)
            or answer.choice not in COMPANY_PROFILE_FIELD_DECISIONS
        ):
            assessments[question_id] = ("insufficient_evidence", 0.0)
            continue
        confidence = answer_confidence(answer)
        decision = (
            answer.choice
            if confidence >= COMPANY_PROFILE_FIELD_REVIEW_CONFIDENCE
            else "insufficient_evidence"
        )
        assessments[question_id] = (decision, confidence)

    revision_fields = [
        question_id
        for question_id, (decision, _confidence) in assessments.items()
        if decision in {"revise_missing", "revise_unsupported"}
    ]
    if revision_fields:
        decision = "needs_revision"
    elif any(
        field_decision != "acceptable" for field_decision, _confidence in assessments.values()
    ):
        decision = "insufficient_evidence"
    elif overall.choice == "needs_revision":
        decision = "needs_revision"
    elif (
        overall.choice == "insufficient_evidence"
        or overall_confidence < COMPANY_PROFILE_FIELD_REVIEW_CONFIDENCE
    ):
        decision = "insufficient_evidence"
    else:
        decision = "ready"

    return decision, overall_confidence, assessments, revision_fields


ASSIST_REJECT_CONFIDENCE = 0.95
AUTONOMOUS_ADVANCE_CONFIDENCE = 0.92
AUTONOMOUS_REJECT_CONFIDENCE = 0.95
LEAD_DECISION_POLICY_VERSION = "company-lead-routing.v2"
COMPANY_PROFILE_QUESTION_VERSION = "company-profile-review.v3"

ARTIFACT_QUALITY_CONFIDENCE = 0.80
LEAD_ARTIFACT_QUALITY_POLICY_VERSION = "lead-artifact-quality.v1"


def assess_artifact_quality_result(
    result: JudgeEvaluationResult,
    *,
    overall_question_id: str,
    check_question_ids: tuple[str, ...],
) -> tuple[str, float, dict[str, tuple[str, float]], list[str]]:
    """Apply a bounded, fail-closed quality policy to a generated artifact.

    Judge answers are signals, not permissions. A correction is eligible only when a
    typed, recognized check requests it at or above the provisional confidence floor.
    Missing or low-confidence answers remain insufficient evidence.
    """
    overall = result.answers.get(overall_question_id)
    if not isinstance(overall, JudgeChoiceAnswer) or overall.choice not in {
        "ready",
        "needs_revision",
        "insufficient_evidence",
    }:
        return "insufficient_evidence", 0.0, {}, []

    overall_confidence = answer_confidence(overall)
    assessments: dict[str, tuple[str, float]] = {}
    for question_id in check_question_ids:
        answer = result.answers.get(question_id)
        if not isinstance(answer, JudgeChoiceAnswer) or answer.choice not in {
            "acceptable",
            "needs_revision",
            "insufficient_evidence",
        }:
            assessments[question_id] = ("insufficient_evidence", 0.0)
            continue

        confidence = answer_confidence(answer)
        decision = (
            answer.choice if confidence >= ARTIFACT_QUALITY_CONFIDENCE else "insufficient_evidence"
        )
        assessments[question_id] = (decision, confidence)

    revision_fields = [
        question_id
        for question_id, (decision, _confidence) in assessments.items()
        if decision == "needs_revision"
    ]
    if revision_fields:
        decision = "needs_revision"
    elif any(
        field_decision != "acceptable" for field_decision, _confidence in assessments.values()
    ):
        decision = "insufficient_evidence"
    elif overall.choice == "needs_revision":
        decision = "needs_revision"
    elif (
        overall.choice == "insufficient_evidence"
        or overall_confidence < ARTIFACT_QUALITY_CONFIDENCE
    ):
        decision = "insufficient_evidence"
    else:
        decision = "ready"

    return decision, overall_confidence, assessments, revision_fields
