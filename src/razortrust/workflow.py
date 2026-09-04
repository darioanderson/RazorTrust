from __future__ import annotations

from uuid import UUID, uuid4

from .audit import AuditLedger
from .domain import (
    AnalystReviewRecord,
    AnalystReviewSubmission,
    DecisionReason,
    DecisionResponse,
    EvidenceAssessment,
    EvidenceRecord,
    EvidenceSubmission,
    EvidenceVerdict,
    HoldCase,
    HoldCreate,
    HoldDecision,
    HoldEvaluationInput,
    HoldState,
    ModelPrediction,
    PolicyCheck,
    PolicyResult,
    ProbabilityVector,
    utc_now,
)
from .evidence import EvidenceVerifier, LegacyEvidenceVerifier
from .features import FEATURE_SCHEMA_VERSION, build_point_in_time_features
from .merchant_guidance import build_merchant_guidance
from .observability import current_trace_id, domain_span
from .policy import HoldPolicy, PolicyUnavailable
from .policy_config import DEFAULT_DECISION_POLICY
from .repository import (
    HoldRepository,
    InMemoryHoldRepository,
    InvalidTransition,
    TransactionalAuditRepository,
)
from .risk import DEFAULT_THRESHOLDS, CostPolicy, RiskModel


class HoldService:
    """Coordinates the hold lifecycle without hiding model or policy failures."""

    def __init__(
        self,
        repository: HoldRepository,
        risk_model: RiskModel,
        cost_policy: CostPolicy,
        hold_policy: HoldPolicy,
        ledger: AuditLedger,
        evidence_verifier: EvidenceVerifier | None = None,
    ) -> None:
        self.repository = repository
        self.risk_model = risk_model
        self.cost_policy = cost_policy
        self.hold_policy = hold_policy
        self.ledger = ledger
        self.evidence_verifier = evidence_verifier or LegacyEvidenceVerifier()

    async def create_hold(self, request: HoldCreate) -> tuple[HoldCase, bool]:
        hold, created = await self.repository.create(request)
        if created:
            self.ledger.append(
                case_id=hold.hold_id,
                actor_id="hold-intake",
                event_type="HOLD_CREATED",
                payload={
                    "request_id": str(hold.request_id),
                    "merchant_id": hold.merchant_id,
                    "source_event_id": hold.source_event_id,
                    "triggered_at": hold.triggered_at.isoformat(),
                },
            )
        return hold, created

    async def evaluate(
        self, hold_id: UUID, evaluation_input: HoldEvaluationInput
    ) -> DecisionResponse:
        hold = await self.repository.get(hold_id)
        if hold.state != HoldState.OPEN:
            return await self.repository.get_decision(hold_id)
        return await self._score(hold, evaluation_input, evidence=None)

    async def submit_evidence(
        self, hold_id: UUID, submission: EvidenceSubmission
    ) -> EvidenceRecord:
        evidence, created = await self.repository.submit_evidence(hold_id, submission)
        if created:
            self.ledger.append(
                case_id=hold_id,
                actor_id="evidence-service",
                event_type="EVIDENCE_SUBMITTED",
                payload={
                    "evidence_id": str(evidence.evidence_id),
                    "evidence_type": evidence.submission.evidence_type,
                    "content_sha256": evidence.submission.content_sha256,
                    "recency_hours": round(evidence.recency_hours, 3),
                    "type_match": evidence.type_match,
                },
            )
        return evidence

    async def rescore(self, hold_id: UUID) -> DecisionResponse:
        hold = await self.repository.get(hold_id)
        if hold.state != HoldState.EVIDENCE_SUBMITTED or hold.evidence_round != 1:
            raise InvalidTransition("rescore is allowed exactly once after evidence submission")
        evaluation_input = await self.repository.get_evaluation_input(hold_id)
        evidence = await self.repository.get_evidence(hold_id)
        return await self._score(hold, evaluation_input, evidence=evidence)

    async def record_analyst_review(
        self,
        hold_id: UUID,
        submission: AnalystReviewSubmission,
        *,
        analyst_id: str,
    ) -> AnalystReviewRecord:
        review, created = await self.repository.record_analyst_review(
            hold_id, submission, analyst_id
        )
        if created and review.audit_head_hash == "0" * 64:
            action = submission.resolved_action()
            authorized_decision = submission.resolved_decision()
            head_hash = self.ledger.append(
                case_id=hold_id,
                actor_id=analyst_id,
                actor_type="HUMAN",
                event_type="ANALYST_OUTCOME_RECORDED",
                payload={
                    "review_id": str(review.review_id),
                    "outcome": review.submission.outcome,
                    "action": action,
                    "authorized_decision": authorized_decision,
                    "ai_recommendation": ((await self.repository.get_decision(hold_id)).decision),
                    "override": action == "OVERRIDE_AI",
                    "reason_code": review.submission.reason_code,
                    "rationale": review.submission.rationale,
                    "training_label": review.training_label,
                    "policy_reference": review.policy_reference,
                    "attribution": {
                        "human_id": analyst_id,
                        "agent_id": submission.agent_id,
                        "agent_session": submission.agent_session,
                        "delegated_permissions": submission.delegated_permissions,
                        "authorized_amount": submission.authorized_amount,
                        "authorized_item": submission.authorized_item,
                        "infrastructure_provider": submission.infrastructure_provider,
                        "transaction_identity": submission.transaction_identity,
                    },
                },
                policy_version=review.policy_reference,
            )
            review = review.model_copy(update={"audit_head_hash": head_hash})
            if isinstance(self.repository, InMemoryHoldRepository):
                review = await self.repository.update_analyst_review_audit_hash(hold_id, head_hash)
        return review

    async def _score(
        self,
        hold: HoldCase,
        evaluation_input: HoldEvaluationInput,
        evidence: EvidenceRecord | None,
    ) -> DecisionResponse:
        system_error = False
        error_reason: DecisionReason | None = None
        evidence_assessment: EvidenceAssessment | None = None
        try:
            with domain_span(
                "features.build",
                {"hold.id": str(hold.hold_id), "merchant.cohort": evaluation_input.cohort},
            ):
                features = build_point_in_time_features(hold, evaluation_input)
            with domain_span(
                "model.predict",
                {"model.version": self.risk_model.version},
            ):
                prediction = self.risk_model.predict(features, evaluation_input.cohort)
            thresholds = getattr(self.risk_model, "thresholds", DEFAULT_THRESHOLDS)
            candidate, expected_cost = self.cost_policy.choose(
                prediction,
                evidence_round=hold.evidence_round,
                thresholds=thresholds,
            )
            if evidence is not None:
                evidence_assessment = self.evidence_verifier.assess(hold, evidence)
                candidate = (
                    HoldDecision.RELEASE
                    if evidence_assessment.verdict == EvidenceVerdict.SUPPORTED
                    and prediction.probabilities.escalate
                    < DEFAULT_DECISION_POLICY.evidence_release_risk_cap
                    and not prediction.novelty_override
                    else HoldDecision.ESCALATE
                )
                expected_cost = self.cost_policy.expected_cost(prediction, candidate)
            await self.repository.save_feature_snapshot(hold.hold_id, evaluation_input, features)
            self.ledger.append(
                case_id=hold.hold_id,
                actor_id="feature-builder",
                event_type="FEATURE_SNAPSHOT_CREATED",
                payload={
                    "features": features.model_dump(mode="json"),
                    "as_of": hold.triggered_at.isoformat(),
                },
                feature_schema_version=FEATURE_SCHEMA_VERSION,
            )
        except (ArithmeticError, RuntimeError, ValueError):
            system_error = True
            error_reason = DecisionReason.MODEL_ERROR
            candidate = HoldDecision.ESCALATE
            expected_cost = 0.0
            prediction = _failure_prediction(self.risk_model.version)
            thresholds = DEFAULT_THRESHOLDS

        policy_request = PolicyCheck(
            hold_id=hold.hold_id,
            candidate_decision=candidate,
            probabilities=prediction.probabilities,
            thresholds=thresholds,
            novelty_override=prediction.novelty_override,
            evidence_round=hold.evidence_round,
            model_version=prediction.model_version,
            policy_version="hold-policy@1",
            system_error=system_error,
            evidence_assessment=evidence_assessment,
            evidence_release_risk_cap=DEFAULT_DECISION_POLICY.evidence_release_risk_cap,
        )
        try:
            with domain_span(
                "policy.opa",
                {
                    "model.version": prediction.model_version,
                    "evidence.round": hold.evidence_round,
                },
            ):
                policy_result = await self.hold_policy.check(policy_request)
        except PolicyUnavailable:
            error_reason = DecisionReason.POLICY_ERROR
            policy_result = PolicyResult(
                allowed_decision=HoldDecision.ESCALATE,
                guardrail_triggered=True,
                reasons=["POLICY_ENGINE_UNAVAILABLE"],
                policy_version="fail-safe@1",
            )

        reason = (
            DecisionReason.HUMAN_ONLY
            if getattr(self.risk_model, "human_only", False)
            else error_reason or _decision_reason(policy_result, hold.evidence_round)
        )
        trace_id = current_trace_id(uuid4().hex)
        audit_payload = {
            "decision": {
                "final": policy_result.allowed_decision,
                "candidate": candidate,
                "probabilities": prediction.probabilities.model_dump(mode="json"),
                "anomaly_score": prediction.anomaly_score,
                "expected_cost_units": expected_cost,
                "reason": reason,
            },
            "artifacts": {
                "model_version": prediction.model_version,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "policy_version": policy_result.policy_version,
                "cost_matrix_version": self.cost_policy.cost_matrix.cost_matrix_version,
                "cost_matrix_sha256": self.cost_policy.cost_matrix.content_sha256,
            },
            "evidence_round": hold.evidence_round,
            "evidence_assessment": (
                evidence_assessment.model_dump(mode="json")
                if evidence_assessment is not None
                else None
            ),
        }
        decision_created_at = utc_now()
        merchant_guidance = build_merchant_guidance(
            policy_result.allowed_decision,
            prediction.top_features,
            created_at=decision_created_at,
        )
        audit_payload["merchant_guidance"] = (
            merchant_guidance.model_dump(mode="json") if merchant_guidance is not None else None
        )
        response = DecisionResponse(
            hold_id=hold.hold_id,
            trace_id=trace_id,
            decision=policy_result.allowed_decision,
            probabilities=prediction.probabilities,
            calibrated=prediction.calibrated,
            calibration_method=prediction.calibration_method,
            anomaly_score=prediction.anomaly_score,
            expected_cost_units=expected_cost,
            cost_matrix_version=self.cost_policy.cost_matrix.cost_matrix_version,
            cost_matrix_sha256=self.cost_policy.cost_matrix.content_sha256,
            thresholds=thresholds,
            top_features=prediction.top_features,
            model_version=prediction.model_version,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            policy_version=policy_result.policy_version,
            evidence_round=hold.evidence_round,
            novelty_override=prediction.novelty_override,
            reason_code=reason,
            audit_head_hash="0" * 64,
            evidence_assessment=evidence_assessment,
            merchant_guidance=merchant_guidance,
            created_at=decision_created_at,
        )
        if isinstance(self.repository, TransactionalAuditRepository):
            with domain_span("db.save_decision", {"decision": str(response.decision)}):
                response = await self.repository.save_decision_and_audit(
                    response, audit_payload=audit_payload, trace_id=trace_id
                )
        else:
            head_hash = self.ledger.append(
                case_id=hold.hold_id,
                actor_id="risk-decision-service",
                event_type="RISK_DECISION",
                payload=audit_payload,
                model_version=prediction.model_version,
                feature_schema_version=FEATURE_SCHEMA_VERSION,
                policy_version=policy_result.policy_version,
            )
            response = response.model_copy(update={"audit_head_hash": head_hash})
            await self.repository.save_decision(response)
        return response


def _failure_prediction(model_version: str) -> ModelPrediction:
    return ModelPrediction(
        probabilities=ProbabilityVector(release=0.0, evidence_needed=0.0, escalate=1.0),
        calibrated=False,
        calibration_method="unavailable",
        anomaly_score=1.0,
        novelty_override=True,
        top_features=[],
        model_version=model_version,
    )


def _decision_reason(policy_result: PolicyResult, evidence_round: int) -> DecisionReason:
    if policy_result.guardrail_triggered:
        if "NOVELTY_OVERRIDE" in policy_result.reasons:
            return DecisionReason.NOVELTY_DISAGREEMENT
        if "EVIDENCE_ROUND_EXHAUSTED" in policy_result.reasons or evidence_round == 1:
            return DecisionReason.EVIDENCE_EXHAUSTED
        return DecisionReason.POLICY_GUARDRAIL
    return DecisionReason.MODEL_POLICY
