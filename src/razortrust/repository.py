from __future__ import annotations

import asyncio
from typing import Any, Literal, Protocol, runtime_checkable
from uuid import UUID, uuid4

from .domain import (
    AnalystOutcome,
    AnalystReviewRecord,
    AnalystReviewSubmission,
    AnalystTrainingExample,
    DecisionResponse,
    EvidenceRecord,
    EvidenceSubmission,
    EvidenceType,
    FeatureVector,
    HoldCase,
    HoldCreate,
    HoldDecision,
    HoldEvaluationInput,
    HoldState,
    HumanReviewAction,
    utc_now,
)


class HoldNotFound(LookupError):
    pass


class InvalidTransition(RuntimeError):
    pass


class HoldRepository(Protocol):
    async def healthcheck(self) -> None: ...

    async def create(self, request: HoldCreate) -> tuple[HoldCase, bool]: ...

    async def get(self, hold_id: UUID) -> HoldCase: ...

    async def list_holds(self, *, merchant_id: str | None, limit: int) -> list[HoldCase]: ...

    async def save_feature_snapshot(
        self,
        hold_id: UUID,
        evaluation_input: HoldEvaluationInput,
        features: FeatureVector,
    ) -> None: ...

    async def get_evaluation_input(self, hold_id: UUID) -> HoldEvaluationInput: ...

    async def save_decision(self, decision: DecisionResponse) -> HoldCase: ...

    async def get_decision(self, hold_id: UUID) -> DecisionResponse: ...

    async def submit_evidence(
        self, hold_id: UUID, submission: EvidenceSubmission
    ) -> tuple[EvidenceRecord, bool]: ...

    async def get_evidence(self, hold_id: UUID) -> EvidenceRecord: ...

    async def record_analyst_review(
        self, hold_id: UUID, submission: AnalystReviewSubmission, analyst_id: str
    ) -> tuple[AnalystReviewRecord, bool]: ...

    async def export_analyst_training_examples(self) -> list[AnalystTrainingExample]: ...


@runtime_checkable
class TransactionalAuditRepository(Protocol):
    async def save_decision_and_audit(
        self,
        decision: DecisionResponse,
        *,
        audit_payload: dict[str, Any],
        trace_id: str,
    ) -> DecisionResponse: ...

    async def get_audit_records(self, hold_id: UUID) -> list[dict[str, Any]]: ...


class InMemoryHoldRepository:
    """Development repository used by local runs and domain tests.

    It is deliberately not presented as durable or multi-process safe. A
    PostgreSQL implementation must satisfy ``HoldRepository`` before it replaces
    this adapter in the application factory.
    """

    def __init__(self) -> None:
        self._cases: dict[UUID, HoldCase] = {}
        self._request_index: dict[UUID, UUID] = {}
        self._decisions: dict[UUID, DecisionResponse] = {}
        self._evaluation_inputs: dict[UUID, HoldEvaluationInput] = {}
        self._features: dict[UUID, FeatureVector] = {}
        self._evidence: dict[UUID, EvidenceRecord] = {}
        self._evidence_request_index: dict[UUID, UUID] = {}
        self._reviews: dict[UUID, list[AnalystReviewRecord]] = {}
        self._reviews_by_request: dict[UUID, AnalystReviewRecord] = {}
        self._review_request_index: dict[UUID, UUID] = {}
        self._lock = asyncio.Lock()

    async def healthcheck(self) -> None:
        return None

    async def create(self, request: HoldCreate) -> tuple[HoldCase, bool]:
        async with self._lock:
            existing_id = self._request_index.get(request.request_id)
            if existing_id is not None:
                return self._cases[existing_id].model_copy(deep=True), False
            hold = HoldCase(hold_id=uuid4(), **request.model_dump())
            self._cases[hold.hold_id] = hold
            self._request_index[request.request_id] = hold.hold_id
            return hold.model_copy(deep=True), True

    async def get(self, hold_id: UUID) -> HoldCase:
        hold = self._cases.get(hold_id)
        if hold is None:
            raise HoldNotFound(str(hold_id))
        return hold.model_copy(deep=True)

    async def list_holds(self, *, merchant_id: str | None, limit: int) -> list[HoldCase]:
        cases = [
            hold
            for hold in self._cases.values()
            if merchant_id is None or hold.merchant_id == merchant_id
        ]
        ordered = sorted(cases, key=lambda hold: hold.updated_at, reverse=True)
        return [hold.model_copy(deep=True) for hold in ordered[:limit]]

    async def save_feature_snapshot(
        self,
        hold_id: UUID,
        evaluation_input: HoldEvaluationInput,
        features: FeatureVector,
    ) -> None:
        async with self._lock:
            if hold_id not in self._cases:
                raise HoldNotFound(str(hold_id))
            self._evaluation_inputs[hold_id] = evaluation_input.model_copy(deep=True)
            self._features[hold_id] = features.model_copy(deep=True)

    async def get_evaluation_input(self, hold_id: UUID) -> HoldEvaluationInput:
        try:
            return self._evaluation_inputs[hold_id].model_copy(deep=True)
        except KeyError as exc:
            raise InvalidTransition("hold must be evaluated before it can be rescored") from exc

    async def save_decision(self, decision: DecisionResponse) -> HoldCase:
        async with self._lock:
            hold = self._cases.get(decision.hold_id)
            if hold is None:
                raise HoldNotFound(str(decision.hold_id))
            if (
                decision.decision == HoldDecision.RELEASE
                or decision.decision == HoldDecision.ESCALATE
            ):
                hold.state = HoldState.HUMAN_REVIEW
            else:
                if hold.evidence_round == 1:
                    raise InvalidTransition("a second evidence round is not allowed")
                hold.state = HoldState.AWAITING_EVIDENCE
            hold.updated_at = utc_now()
            self._decisions[decision.hold_id] = decision.model_copy(deep=True)
            return hold.model_copy(deep=True)

    async def get_decision(self, hold_id: UUID) -> DecisionResponse:
        if hold_id not in self._cases:
            raise HoldNotFound(str(hold_id))
        try:
            return self._decisions[hold_id].model_copy(deep=True)
        except KeyError as exc:
            raise InvalidTransition("hold has not been evaluated") from exc

    async def submit_evidence(
        self, hold_id: UUID, submission: EvidenceSubmission
    ) -> tuple[EvidenceRecord, bool]:
        async with self._lock:
            existing_hold_id = self._evidence_request_index.get(submission.request_id)
            if existing_hold_id is not None:
                if existing_hold_id != hold_id:
                    raise InvalidTransition("evidence request_id already belongs to another hold")
                return self._evidence[hold_id].model_copy(deep=True), False

            hold = self._cases.get(hold_id)
            if hold is None:
                raise HoldNotFound(str(hold_id))
            if hold.state != HoldState.AWAITING_EVIDENCE or hold.evidence_round != 0:
                raise InvalidTransition("evidence is accepted only once after EVIDENCE_NEEDED")

            expected_type = expected_evidence_type(hold.reason_code)
            recency = (
                submission.submitted_at - submission.evidence_observed_at
            ).total_seconds() / 3600
            evidence = EvidenceRecord(
                evidence_id=uuid4(),
                hold_id=hold_id,
                submission=submission,
                recency_hours=recency,
                type_match=expected_type is None or submission.evidence_type == expected_type,
            )
            hold.evidence_round = 1
            hold.state = HoldState.EVIDENCE_SUBMITTED
            hold.updated_at = utc_now()
            self._evidence[hold_id] = evidence
            self._evidence_request_index[submission.request_id] = hold_id
            return evidence.model_copy(deep=True), True

    async def get_evidence(self, hold_id: UUID) -> EvidenceRecord:
        try:
            return self._evidence[hold_id].model_copy(deep=True)
        except KeyError as exc:
            raise InvalidTransition("no evidence has been submitted") from exc

    async def record_analyst_review(
        self, hold_id: UUID, submission: AnalystReviewSubmission, analyst_id: str
    ) -> tuple[AnalystReviewRecord, bool]:
        async with self._lock:
            existing_hold_id = self._review_request_index.get(submission.request_id)
            if existing_hold_id is not None:
                if existing_hold_id != hold_id:
                    raise InvalidTransition("review request_id already belongs to another hold")
                return self._reviews_by_request[submission.request_id].model_copy(deep=True), False
            hold = self._cases.get(hold_id)
            if hold is None:
                raise HoldNotFound(str(hold_id))
            if hold.state not in {HoldState.HUMAN_REVIEW, HoldState.AWAITING_EVIDENCE}:
                raise InvalidTransition("human actions require a pending review or evidence case")
            action = submission.resolved_action()
            decision = submission.resolved_decision()
            ai_recommendation = self._decisions[hold_id].decision
            if (
                submission.action == HumanReviewAction.APPROVE_RELEASE
                and ai_recommendation != HoldDecision.RELEASE
            ):
                raise InvalidTransition(
                    "use OVERRIDE_AI when approving against the AI recommendation"
                )
            if submission.action == HumanReviewAction.OVERRIDE_AI and decision == ai_recommendation:
                raise InvalidTransition("an AI override must change the recommended decision")
            if decision == HoldDecision.EVIDENCE_NEEDED and hold.evidence_round == 1:
                raise InvalidTransition("a second evidence round is not allowed")
            training_label = _training_label(submission)
            record = AnalystReviewRecord(
                review_id=uuid4(),
                hold_id=hold_id,
                analyst_id=analyst_id,
                submission=submission,
                training_label=training_label,
                audit_head_hash="0" * 64,
            )
            if decision == HoldDecision.RELEASE:
                hold.state = HoldState.RESOLVED
            elif decision == HoldDecision.EVIDENCE_NEEDED:
                hold.state = HoldState.AWAITING_EVIDENCE
            elif (
                action == HumanReviewAction.OVERRIDE_AI
                or submission.outcome == AnalystOutcome.FRAUD_CONFIRMED
            ):
                hold.state = HoldState.DENIED
            else:
                hold.state = HoldState.HUMAN_REVIEW
            hold.updated_at = utc_now()
            self._reviews.setdefault(hold_id, []).append(record)
            self._reviews_by_request[submission.request_id] = record
            self._review_request_index[submission.request_id] = hold_id
            return record.model_copy(deep=True), True

    async def update_analyst_review_audit_hash(
        self, hold_id: UUID, audit_head_hash: str
    ) -> AnalystReviewRecord:
        async with self._lock:
            try:
                review = self._reviews[hold_id][-1].model_copy(
                    update={"audit_head_hash": audit_head_hash}
                )
            except KeyError as exc:
                raise InvalidTransition("no analyst review has been recorded") from exc
            self._reviews[hold_id][-1] = review
            self._reviews_by_request[review.submission.request_id] = review
            return review.model_copy(deep=True)

    async def export_analyst_training_examples(self) -> list[AnalystTrainingExample]:
        examples = []
        for hold_id, reviews in self._reviews.items():
            for review in reviews:
                if review.training_label == "UNRESOLVED":
                    continue
                examples.append(
                    AnalystTrainingExample(
                        hold_id=hold_id,
                        features=self._features[hold_id],
                        cohort=self._evaluation_inputs[hold_id].cohort,
                        training_label=review.training_label,
                        outcome_reason_code=review.submission.reason_code,
                        decided_at=review.submission.decided_at,
                        feature_schema_version="1.0.0",
                    )
                )
        return examples


def _training_label(
    submission: AnalystReviewSubmission,
) -> Literal["LEGITIMATE", "RISKY", "UNRESOLVED"]:
    if submission.outcome == AnalystOutcome.FRAUD_CONFIRMED:
        return "RISKY"
    if submission.resolved_decision() == HoldDecision.RELEASE:
        return "LEGITIMATE"
    return "UNRESOLVED"


def expected_evidence_type(reason_code: str | None) -> EvidenceType | None:
    if reason_code is None:
        return None
    normalized = reason_code.upper()
    mapping = {
        "CAMPAIGN": EvidenceType.CAMPAIGN,
        "VOLUME_SPIKE": EvidenceType.CAMPAIGN,
        "FULFILLMENT": EvidenceType.FULFILLMENT,
        "GEO_EXPANSION": EvidenceType.GEO_EXPANSION,
        "AUTH_INCIDENT": EvidenceType.AUTH_INCIDENT,
        "INVENTORY": EvidenceType.INVENTORY,
    }
    return mapping.get(normalized)
