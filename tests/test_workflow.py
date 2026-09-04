from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from test_features import AS_OF, evaluation_input

from razortrust.audit import AuditLedger
from razortrust.domain import (
    AnalystReviewSubmission,
    EvidenceSubmission,
    FeatureContribution,
    HoldCreate,
    HoldDecision,
    ModelPrediction,
    ProbabilityVector,
    utc_now,
)
from razortrust.policy import LocalHoldPolicy, PolicyUnavailable
from razortrust.repository import InMemoryHoldRepository, InvalidTransition
from razortrust.risk import CostPolicy
from razortrust.workflow import HoldService


class FixedPredictionModel:
    version = "test-model@1"

    def predict(self, _features, cohort="pooled") -> ModelPrediction:
        self.last_cohort = cohort
        probabilities = ProbabilityVector(release=0.10, evidence_needed=0.80, escalate=0.10)
        return ModelPrediction(
            probabilities=probabilities,
            calibrated=True,
            calibration_method="test",
            anomaly_score=0.20,
            top_features=[
                FeatureContribution(
                    feature="volume_delta_z",
                    observed_value=0.0,
                    contribution_value=0.1,
                    direction="toward_EVIDENCE_NEEDED",
                    reference="test fixture",
                    attribution_method="test",
                )
            ],
            model_version=self.version,
        )


class FailingModel:
    version = "failing-model@1"

    def predict(self, _features, cohort="pooled"):
        raise RuntimeError("model artifact is unavailable")


class UnavailablePolicy:
    async def check(self, _request):
        raise PolicyUnavailable("OPA is unavailable")


def service(tmp_path: Path, model=None, policy=None) -> HoldService:
    return HoldService(
        InMemoryHoldRepository(),
        model or FixedPredictionModel(),
        CostPolicy(),
        policy or LocalHoldPolicy(),
        AuditLedger(tmp_path / "audit.jsonl"),
    )


async def create_case(hold_service: HoldService):
    hold, _ = await hold_service.create_hold(
        HoldCreate(
            request_id=uuid4(),
            merchant_id="merchant_001",
            source_event_id="settlement_001",
            triggered_at=AS_OF,
            reason_code="CAMPAIGN",
        )
    )
    return hold


@pytest.mark.asyncio
async def test_one_evidence_round_then_release(tmp_path: Path) -> None:
    hold_service = service(tmp_path)
    hold = await create_case(hold_service)
    first = await hold_service.evaluate(hold.hold_id, evaluation_input())
    assert first.decision == HoldDecision.EVIDENCE_NEEDED

    now = utc_now()
    evidence = EvidenceSubmission(
        request_id=uuid4(),
        evidence_type="CAMPAIGN",
        submitted_at=now,
        evidence_observed_at=now - timedelta(hours=2),
        content_sha256="a" * 64,
        metadata={"campaign_id": "campaign_42"},
    )
    await hold_service.submit_evidence(hold.hold_id, evidence)
    second = await hold_service.rescore(hold.hold_id)
    assert second.decision == HoldDecision.RELEASE
    assert second.evidence_round == 1
    assert second.probabilities == first.probabilities
    assert second.evidence_assessment is not None
    assert second.evidence_assessment.verdict == "SUPPORTED"

    with pytest.raises(InvalidTransition, match="only once"):
        await hold_service.submit_evidence(
            hold.hold_id,
            evidence.model_copy(update={"request_id": uuid4()}),
        )


@pytest.mark.asyncio
async def test_runtime_receives_derived_cohort(tmp_path: Path) -> None:
    model = FixedPredictionModel()
    hold_service = service(tmp_path, model=model)
    hold = await create_case(hold_service)
    request = evaluation_input().model_copy(update={"cohort": "SMB_ECOMMERCE"})
    await hold_service.evaluate(hold.hold_id, request)
    assert model.last_cohort == "SMB_ECOMMERCE"


@pytest.mark.asyncio
async def test_human_requests_evidence_then_approves_ai_release(tmp_path: Path) -> None:
    hold_service = service(tmp_path)
    hold = await create_case(hold_service)
    first = await hold_service.evaluate(hold.hold_id, evaluation_input())
    assert first.decision == HoldDecision.EVIDENCE_NEEDED

    await hold_service.record_analyst_review(
        hold.hold_id,
        AnalystReviewSubmission(
            request_id=uuid4(),
            action="REQUEST_EVIDENCE",
            reason_code="CAMPAIGN_PROOF_REQUIRED",
            rationale="The payment spike needs a campaign record before authorization.",
            decided_at=utc_now(),
            agent_id="risk-decision-service",
            agent_session="session-42",
            delegated_permissions=["REVIEW_SETTLEMENT"],
            authorized_amount=12000,
            transaction_identity="settlement_001",
        ),
        analyst_id="analyst-8",
    )
    now = utc_now()
    await hold_service.submit_evidence(
        hold.hold_id,
        EvidenceSubmission(
            request_id=uuid4(),
            evidence_type="CAMPAIGN",
            submitted_at=now,
            evidence_observed_at=now - timedelta(hours=1),
            content_sha256="d" * 64,
        ),
    )
    final = await hold_service.rescore(hold.hold_id)
    assert final.decision == HoldDecision.RELEASE
    assert (await hold_service.repository.get(hold.hold_id)).state == "HUMAN_REVIEW"

    approval = await hold_service.record_analyst_review(
        hold.hold_id,
        AnalystReviewSubmission(
            request_id=uuid4(),
            action="APPROVE_RELEASE",
            reason_code="VERIFIED_CAMPAIGN",
            rationale="The signed campaign evidence matches the settlement window.",
            decided_at=utc_now(),
            agent_id="risk-decision-service",
            agent_session="session-42",
            delegated_permissions=["REVIEW_SETTLEMENT"],
            authorized_amount=12000,
            transaction_identity="settlement_001",
        ),
        analyst_id="analyst-8",
    )
    assert approval.training_label == "LEGITIMATE"
    assert (await hold_service.repository.get(hold.hold_id)).state == "RESOLVED"
    records = hold_service.ledger.records_for_case(hold.hold_id)
    assert records[-1]["payload"]["action"] == "APPROVE_RELEASE"
    assert records[-1]["payload"]["attribution"]["human_id"] == "analyst-8"


def test_override_requires_an_explicit_authorized_decision() -> None:
    with pytest.raises(ValueError, match="authorized_decision is required"):
        AnalystReviewSubmission(
            request_id=uuid4(),
            action="OVERRIDE_AI",
            reason_code="MANUAL_OVERRIDE",
            rationale="A reviewer must state the exact replacement decision.",
            decided_at=utc_now(),
        )


@pytest.mark.asyncio
async def test_contradictory_evidence_cannot_release(tmp_path: Path) -> None:
    hold_service = service(tmp_path)
    hold = await create_case(hold_service)
    first = await hold_service.evaluate(hold.hold_id, evaluation_input())
    now = utc_now()
    await hold_service.submit_evidence(
        hold.hold_id,
        EvidenceSubmission(
            request_id=uuid4(),
            evidence_type="FULFILLMENT",
            submitted_at=now,
            evidence_observed_at=now - timedelta(hours=2),
            content_sha256="b" * 64,
        ),
    )
    second = await hold_service.rescore(hold.hold_id)
    assert second.decision == HoldDecision.ESCALATE
    assert second.probabilities == first.probabilities
    assert second.evidence_assessment is not None
    assert second.evidence_assessment.verdict == "CONTRADICTORY"

    submission = AnalystReviewSubmission(
        request_id=uuid4(),
        outcome="FRAUD_CONFIRMED",
        reason_code="EVIDENCE_CONTRADICTED_ACTIVITY",
        rationale="The submitted fulfilment record did not match the held activity.",
        decided_at=utc_now(),
    )
    review = await hold_service.record_analyst_review(
        hold.hold_id, submission, analyst_id="analyst-7"
    )
    duplicate = await hold_service.record_analyst_review(
        hold.hold_id, submission, analyst_id="analyst-7"
    )
    assert review.training_label == "RISKY"
    assert review.audit_head_hash != "0" * 64
    assert duplicate == review
    assert (await hold_service.repository.get(hold.hold_id)).state == "DENIED"


@pytest.mark.asyncio
async def test_evidence_needed_response_contains_actionable_guidance(tmp_path: Path) -> None:
    hold_service = service(tmp_path)
    hold = await create_case(hold_service)

    decision = await hold_service.evaluate(hold.hold_id, evaluation_input())

    assert decision.merchant_guidance is not None
    assert decision.merchant_guidance.required_evidence
    assert decision.merchant_guidance.template_version == "merchant-guidance@1"


@pytest.mark.asyncio
async def test_model_failure_escalates(tmp_path: Path) -> None:
    hold_service = service(tmp_path, model=FailingModel())
    hold = await create_case(hold_service)
    result = await hold_service.evaluate(hold.hold_id, evaluation_input())
    assert result.decision == HoldDecision.ESCALATE
    assert result.reason_code == "MODEL_ERROR"
    assert result.probabilities.release == 0


@pytest.mark.asyncio
async def test_policy_failure_escalates(tmp_path: Path) -> None:
    hold_service = service(tmp_path, policy=UnavailablePolicy())
    hold = await create_case(hold_service)
    result = await hold_service.evaluate(hold.hold_id, evaluation_input())
    assert result.decision == HoldDecision.ESCALATE
    assert result.reason_code == "POLICY_ERROR"


def test_audit_mutation_breaks_chain(tmp_path: Path) -> None:
    ledger = AuditLedger(tmp_path / "audit.jsonl")
    case_id = uuid4()
    ledger.append(case_id=case_id, actor_id="test", event_type="CREATED", payload={"value": 1})
    record = json.loads(ledger.path.read_text(encoding="utf-8"))
    record["payload"]["value"] = 2
    ledger.path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert ledger.verify() == (False, 0)
