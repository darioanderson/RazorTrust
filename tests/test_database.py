from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from test_features import AS_OF, evaluation_input

from razortrust.audit import AuditLedger
from razortrust.database import Base, SqlHoldRepository
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
from razortrust.policy import LocalHoldPolicy
from razortrust.repository import HoldNotFound, InvalidTransition
from razortrust.risk import CostPolicy
from razortrust.workflow import HoldService


class FixedPredictionModel:
    version = "sql-test-model@1"

    def predict(self, _features, cohort="pooled") -> ModelPrediction:
        del cohort
        probabilities = ProbabilityVector(release=0.10, evidence_needed=0.80, escalate=0.10)
        return ModelPrediction(
            probabilities=probabilities,
            calibrated=True,
            calibration_method="test-fixture",
            anomaly_score=0.20,
            top_features=[
                FeatureContribution(
                    feature="volume_delta_z",
                    observed_value=0.0,
                    contribution_value=0.1,
                    direction="toward_EVIDENCE_NEEDED",
                    reference="test fixture",
                    attribution_method="test fixture",
                )
            ],
            model_version=self.version,
        )


async def sql_repository() -> tuple[SqlHoldRepository, AsyncEngine]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    return SqlHoldRepository(sessions), engine


@pytest.mark.asyncio
async def test_sql_repository_runs_complete_evidence_workflow(tmp_path: Path) -> None:
    repository, engine = await sql_repository()
    try:
        await repository.healthcheck()
        service = HoldService(
            repository,
            FixedPredictionModel(),
            CostPolicy(),
            LocalHoldPolicy(),
            AuditLedger(tmp_path / "sql-audit.jsonl"),
        )
        create = HoldCreate(
            request_id=uuid4(),
            merchant_id="merchant_001",
            source_event_id="settlement_001",
            triggered_at=AS_OF,
            reason_code="CAMPAIGN",
        )
        hold, created = await service.create_hold(create)
        duplicate, duplicate_created = await service.create_hold(create)
        assert created is True
        assert duplicate_created is False
        assert duplicate.hold_id == hold.hold_id
        assert [
            case.hold_id
            for case in await repository.list_holds(merchant_id="merchant_001", limit=10)
        ] == [hold.hold_id]
        assert await repository.list_holds(merchant_id="merchant_unknown", limit=10) == []

        with pytest.raises(InvalidTransition, match="not been evaluated"):
            await repository.get_decision(hold.hold_id)

        first = await service.evaluate(hold.hold_id, evaluation_input())
        assert first.decision == HoldDecision.EVIDENCE_NEEDED
        assert await repository.get_decision(hold.hold_id) == first
        assert await repository.get_evaluation_input(hold.hold_id) == evaluation_input()

        now = utc_now()
        submission = EvidenceSubmission(
            request_id=uuid4(),
            evidence_type="CAMPAIGN",
            submitted_at=now,
            evidence_observed_at=now - timedelta(hours=2),
            content_sha256="b" * 64,
            metadata={"campaign_id": "campaign_42"},
        )
        evidence = await service.submit_evidence(hold.hold_id, submission)
        records_after_first_submission = len(service.ledger.records_for_case(hold.hold_id))
        duplicate_evidence = await service.submit_evidence(hold.hold_id, submission)
        assert duplicate_evidence.evidence_id == evidence.evidence_id
        assert len(service.ledger.records_for_case(hold.hold_id)) == records_after_first_submission
        assert (await repository.get_evidence(hold.hold_id)).recency_hours == pytest.approx(2.0)

        final = await service.rescore(hold.hold_id)
        assert final.decision == HoldDecision.RELEASE
        assert (await repository.get(hold.hold_id)).state == "HUMAN_REVIEW"
        audit_records = await repository.get_audit_records(hold.hold_id)
        assert [record["sequence_no"] for record in audit_records] == [1, 2]
        assert audit_records[-1]["trace_id"] == final.trace_id
        assert audit_records[-1]["record_hash"] == final.audit_head_hash
        async with engine.connect() as connection:
            outbox_count = await connection.scalar(text("SELECT COUNT(*) FROM outbox_events"))
        assert outbox_count == 2

        with pytest.raises(InvalidTransition, match="only once"):
            await service.submit_evidence(
                hold.hold_id,
                submission.model_copy(update={"request_id": uuid4()}),
            )
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_sql_repository_reports_missing_and_invalid_records() -> None:
    repository, engine = await sql_repository()
    missing_id = uuid4()
    try:
        with pytest.raises(HoldNotFound):
            await repository.get(missing_id)
        with pytest.raises(InvalidTransition, match="rescored"):
            await repository.get_evaluation_input(missing_id)
        with pytest.raises(InvalidTransition, match="no evidence"):
            await repository.get_evidence(missing_id)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_sql_analyst_outcome_is_idempotent_audited_and_exportable(tmp_path: Path) -> None:
    repository, engine = await sql_repository()
    try:
        service = HoldService(
            repository,
            FixedPredictionModel(),
            CostPolicy(),
            LocalHoldPolicy(),
            AuditLedger(tmp_path / "analyst-audit.jsonl"),
        )
        hold, _ = await service.create_hold(
            HoldCreate(
                request_id=uuid4(),
                merchant_id="merchant_001",
                source_event_id="settlement_review",
                triggered_at=AS_OF,
                reason_code="CAMPAIGN",
            )
        )
        decision = await service.evaluate(hold.hold_id, evaluation_input())
        assert decision.decision == HoldDecision.EVIDENCE_NEEDED
        now = utc_now()
        await service.submit_evidence(
            hold.hold_id,
            EvidenceSubmission(
                request_id=uuid4(),
                evidence_type="FULFILLMENT",
                submitted_at=now,
                evidence_observed_at=now - timedelta(hours=1),
                content_sha256="c" * 64,
            ),
        )
        final = await service.rescore(hold.hold_id)
        assert final.decision == HoldDecision.ESCALATE
        assert (await repository.get(hold.hold_id)).state == "HUMAN_REVIEW"
        submission = AnalystReviewSubmission(
            request_id=uuid4(),
            outcome="CLEARED",
            reason_code="MANUAL_DOCUMENT_MATCH",
            rationale="Analyst matched the campaign record to the settlement window.",
            decided_at=now,
        )

        first = await service.record_analyst_review(
            hold.hold_id, submission, analyst_id="analyst-1"
        )
        duplicate = await service.record_analyst_review(
            hold.hold_id, submission, analyst_id="analyst-1"
        )

        assert first == duplicate
        assert first.training_label == "LEGITIMATE"
        assert (await repository.get(hold.hold_id)).state == "RESOLVED"
        records = await repository.get_audit_records(hold.hold_id)
        assert records[-1]["event_type"] == "ANALYST_OUTCOME_RECORDED"
        async with engine.connect() as connection:
            review_count = await connection.scalar(text("SELECT COUNT(*) FROM analyst_reviews"))
            analyst_outbox = await connection.scalar(
                text(
                    "SELECT COUNT(*) FROM outbox_events "
                    "WHERE topic = 'razortrust.audit.analyst-outcome'"
                )
            )
        assert review_count == 1
        assert analyst_outbox == 1
        exported = await repository.export_analyst_training_examples()
        assert len(exported) == 1
        assert exported[0].training_label == "LEGITIMATE"
        assert exported[0].outcome_reason_code == "MANUAL_DOCUMENT_MATCH"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_decision_transaction_rolls_back_when_outbox_write_fails(tmp_path: Path) -> None:
    repository, engine = await sql_repository()
    try:
        service = HoldService(
            repository,
            FixedPredictionModel(),
            CostPolicy(),
            LocalHoldPolicy(),
            AuditLedger(tmp_path / "transaction-failure-audit.jsonl"),
        )
        hold, _ = await service.create_hold(
            HoldCreate(
                request_id=uuid4(),
                merchant_id="merchant_failure",
                source_event_id="settlement_failure",
                triggered_at=AS_OF,
                reason_code="CAMPAIGN",
            )
        )
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    "CREATE TRIGGER reject_outbox BEFORE INSERT ON outbox_events "
                    "BEGIN SELECT RAISE(FAIL, 'injected outbox failure'); END"
                )
            )

        with pytest.raises(IntegrityError, match="injected outbox failure"):
            await service.evaluate(hold.hold_id, evaluation_input())

        async with engine.connect() as connection:
            decisions = await connection.scalar(text("SELECT COUNT(*) FROM model_decisions"))
            audits = await connection.scalar(text("SELECT COUNT(*) FROM audit_events"))
            outbox = await connection.scalar(text("SELECT COUNT(*) FROM outbox_events"))
            state = await connection.scalar(
                text("SELECT state FROM hold_cases WHERE hold_id = :hold_id"),
                {"hold_id": hold.hold_id.hex},
            )
        assert (decisions, audits, outbox) == (0, 0, 0)
        assert state == "OPEN"
    finally:
        await engine.dispose()
