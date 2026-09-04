from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any, Literal, cast
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    select,
    text,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .audit import GENESIS_HASH, canonical_json, hash_event
from .domain import (
    AnalystOutcome,
    AnalystReviewRecord,
    AnalystReviewSubmission,
    AnalystTrainingExample,
    DecisionResponse,
    EvidenceRecord,
    EvidenceSubmission,
    FeatureVector,
    HoldCase,
    HoldCreate,
    HoldDecision,
    HoldEvaluationInput,
    HoldState,
    HumanReviewAction,
    utc_now,
)
from .features import FEATURE_SCHEMA_VERSION
from .repository import HoldNotFound, InvalidTransition, expected_evidence_type


class Base(DeclarativeBase):
    pass


class HoldCaseRow(Base):
    __tablename__ = "hold_cases"
    __table_args__ = (
        UniqueConstraint("request_id", name="uq_hold_request_id"),
        CheckConstraint("evidence_round IN (0, 1)", name="ck_hold_evidence_round"),
    )

    hold_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    request_id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    merchant_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    source_event_id: Mapped[str] = mapped_column(String(128), nullable=False)
    triggered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    reason_code: Mapped[str | None] = mapped_column(String(128))
    state: Mapped[str] = mapped_column(String(32), nullable=False)
    evidence_round: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class FeatureSnapshotRow(Base):
    __tablename__ = "feature_snapshots"

    snapshot_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    hold_id: Mapped[UUID] = mapped_column(
        Uuid,
        ForeignKey("hold_cases.hold_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    feature_schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    as_of_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    evaluation_input: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    features: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ModelDecisionRow(Base):
    __tablename__ = "model_decisions"

    decision_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    hold_id: Mapped[UUID] = mapped_column(
        Uuid,
        ForeignKey("hold_cases.hold_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    feature_schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    policy_version: Mapped[str] = mapped_column(String(128), nullable=False)
    decision: Mapped[str] = mapped_column(String(32), nullable=False)
    decision_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class EvidenceSubmissionRow(Base):
    __tablename__ = "evidence_submissions"
    __table_args__ = (UniqueConstraint("request_id", name="uq_evidence_request_id"),)

    evidence_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    hold_id: Mapped[UUID] = mapped_column(
        Uuid,
        ForeignKey("hold_cases.hold_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    request_id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    evidence_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    recency_hours: Mapped[float] = mapped_column(nullable=False)
    type_match: Mapped[bool] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class AuditEventRow(Base):
    __tablename__ = "audit_events"
    __table_args__ = (UniqueConstraint("case_id", "sequence_no", name="uq_audit_case_sequence"),)

    event_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    case_id: Mapped[UUID] = mapped_column(Uuid, nullable=False, index=True)
    sequence_no: Mapped[int] = mapped_column(Integer, nullable=False)
    event_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    canonical_payload: Mapped[str] = mapped_column(Text, nullable=False)
    previous_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    record_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class OutboxEventRow(Base):
    __tablename__ = "outbox_events"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    topic: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ModelManifestRow(Base):
    __tablename__ = "model_manifests"

    model_version: Mapped[str] = mapped_column(String(128), primary_key=True)
    manifest: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    signature_b64: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class PolicyVersionRow(Base):
    __tablename__ = "policy_versions"

    policy_version: Mapped[str] = mapped_column(String(128), primary_key=True)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class AnalystReviewRow(Base):
    __tablename__ = "analyst_reviews"
    __table_args__ = (UniqueConstraint("request_id", name="uq_analyst_review_request_id"),)

    review_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    hold_id: Mapped[UUID] = mapped_column(
        Uuid,
        ForeignKey("hold_cases.hold_id", ondelete="CASCADE"),
        nullable=False,
    )
    request_id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    analyst_id: Mapped[str] = mapped_column(String(128), nullable=False)
    review_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    training_label: Mapped[str] = mapped_column(String(32), nullable=False)
    audit_head_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class SqlHoldRepository:
    """SQLAlchemy implementation of the hold repository contract."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def healthcheck(self) -> None:
        async with self._sessions() as session:
            await session.execute(text("SELECT 1"))

    async def create(self, request: HoldCreate) -> tuple[HoldCase, bool]:
        async with self._sessions() as session:
            existing = await session.scalar(
                select(HoldCaseRow).where(HoldCaseRow.request_id == request.request_id)
            )
            if existing is not None:
                return _case_from_row(existing), False
            now = utc_now()
            row = HoldCaseRow(
                hold_id=uuid4(),
                **request.model_dump(),
                state=HoldState.OPEN,
                evidence_round=0,
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                existing = await session.scalar(
                    select(HoldCaseRow).where(HoldCaseRow.request_id == request.request_id)
                )
                if existing is None:
                    raise
                return _case_from_row(existing), False
            return _case_from_row(row), True

    async def get(self, hold_id: UUID) -> HoldCase:
        async with self._sessions() as session:
            return _case_from_row(await _get_hold_row(session, hold_id))

    async def list_holds(self, *, merchant_id: str | None, limit: int) -> list[HoldCase]:
        async with self._sessions() as session:
            query = select(HoldCaseRow)
            if merchant_id is not None:
                query = query.where(HoldCaseRow.merchant_id == merchant_id)
            rows = (
                await session.scalars(query.order_by(HoldCaseRow.updated_at.desc()).limit(limit))
            ).all()
            return [_case_from_row(row) for row in rows]

    async def save_feature_snapshot(
        self,
        hold_id: UUID,
        evaluation_input: HoldEvaluationInput,
        features: FeatureVector,
    ) -> None:
        payload = features.model_dump(mode="json")
        async with self._sessions() as session:
            hold = await _get_hold_row(session, hold_id)
            session.add(
                FeatureSnapshotRow(
                    hold_id=hold_id,
                    feature_schema_version=FEATURE_SCHEMA_VERSION,
                    as_of_timestamp=hold.triggered_at,
                    evaluation_input=evaluation_input.model_dump(mode="json"),
                    features=payload,
                    content_sha256=hashlib.sha256(canonical_json(payload)).hexdigest(),
                    created_at=utc_now(),
                )
            )
            await session.commit()

    async def get_evaluation_input(self, hold_id: UUID) -> HoldEvaluationInput:
        async with self._sessions() as session:
            row = await session.scalar(
                select(FeatureSnapshotRow)
                .where(FeatureSnapshotRow.hold_id == hold_id)
                .order_by(FeatureSnapshotRow.created_at.desc())
                .limit(1)
            )
            if row is None:
                raise InvalidTransition("hold must be evaluated before it can be rescored")
            return HoldEvaluationInput.model_validate(row.evaluation_input)

    async def save_decision(self, decision: DecisionResponse) -> HoldCase:
        async with self._sessions() as session:
            hold = await _get_hold_row(session, decision.hold_id)
            hold.state = _state_for_decision(decision.decision, hold.evidence_round)
            hold.updated_at = utc_now()
            session.add(
                ModelDecisionRow(
                    hold_id=decision.hold_id,
                    model_version=decision.model_version,
                    feature_schema_version=decision.feature_schema_version,
                    policy_version=decision.policy_version,
                    decision=decision.decision,
                    decision_payload=decision.model_dump(mode="json"),
                    created_at=decision.created_at,
                )
            )
            await session.commit()
            return _case_from_row(hold)

    async def save_decision_and_audit(
        self,
        decision: DecisionResponse,
        *,
        audit_payload: dict[str, Any],
        trace_id: str,
    ) -> DecisionResponse:
        async with self._sessions() as session:
            hold = await _get_hold_row(session, decision.hold_id, for_update=True)
            latest = await session.scalar(
                select(AuditEventRow)
                .where(AuditEventRow.case_id == decision.hold_id)
                .order_by(AuditEventRow.sequence_no.desc())
                .limit(1)
            )
            previous_hash = latest.record_hash if latest is not None else GENESIS_HASH
            sequence_no = (latest.sequence_no + 1) if latest is not None else 1
            now = utc_now()
            event_id = uuid4()
            event = {
                "schema_version": "2.0",
                "event_id": str(event_id),
                "case_id": str(decision.hold_id),
                "sequence_no": sequence_no,
                "trace_id": trace_id,
                "event_type": "RISK_DECISION",
                "timestamp": now.isoformat().replace("+00:00", "Z"),
                "payload": audit_payload,
            }
            record_hash = hash_event(event, previous_hash)
            persisted_decision = decision.model_copy(update={"audit_head_hash": record_hash})
            record = {
                **event,
                "previous_hash": previous_hash,
                "record_hash": record_hash,
            }
            hold.state = _state_for_decision(decision.decision, hold.evidence_round)
            hold.updated_at = now
            session.add_all(
                [
                    ModelDecisionRow(
                        hold_id=decision.hold_id,
                        model_version=decision.model_version,
                        feature_schema_version=decision.feature_schema_version,
                        policy_version=decision.policy_version,
                        decision=decision.decision,
                        decision_payload=persisted_decision.model_dump(mode="json"),
                        created_at=decision.created_at,
                    ),
                    AuditEventRow(
                        event_id=event_id,
                        case_id=decision.hold_id,
                        sequence_no=sequence_no,
                        event_payload=record,
                        canonical_payload=canonical_json(record).decode("utf-8"),
                        previous_hash=previous_hash,
                        record_hash=record_hash,
                        trace_id=trace_id,
                        created_at=now,
                    ),
                    OutboxEventRow(
                        topic="razortrust.audit.risk-decision",
                        payload=record,
                        created_at=now,
                        published_at=None,
                    ),
                ]
            )
            await session.commit()
            return persisted_decision

    async def get_audit_records(self, hold_id: UUID) -> list[dict[str, Any]]:
        async with self._sessions() as session:
            await _get_hold_row(session, hold_id)
            rows = (
                await session.scalars(
                    select(AuditEventRow)
                    .where(AuditEventRow.case_id == hold_id)
                    .order_by(AuditEventRow.sequence_no)
                )
            ).all()
            return [row.event_payload for row in rows]

    async def get_decision(self, hold_id: UUID) -> DecisionResponse:
        async with self._sessions() as session:
            await _get_hold_row(session, hold_id)
            row = await session.scalar(
                select(ModelDecisionRow)
                .where(ModelDecisionRow.hold_id == hold_id)
                .order_by(ModelDecisionRow.created_at.desc())
                .limit(1)
            )
            if row is None:
                raise InvalidTransition("hold has not been evaluated")
            return DecisionResponse.model_validate(row.decision_payload)

    async def submit_evidence(
        self, hold_id: UUID, submission: EvidenceSubmission
    ) -> tuple[EvidenceRecord, bool]:
        async with self._sessions() as session:
            existing = await session.scalar(
                select(EvidenceSubmissionRow).where(
                    EvidenceSubmissionRow.request_id == submission.request_id
                )
            )
            if existing is not None:
                if existing.hold_id != hold_id:
                    raise InvalidTransition("evidence request_id already belongs to another hold")
                return _evidence_from_row(existing), False

            hold = await _get_hold_row(session, hold_id, for_update=True)
            if hold.state != HoldState.AWAITING_EVIDENCE or hold.evidence_round != 0:
                raise InvalidTransition("evidence is accepted only once after EVIDENCE_NEEDED")
            expected_type = expected_evidence_type(hold.reason_code)
            evidence = EvidenceRecord(
                evidence_id=uuid4(),
                hold_id=hold_id,
                submission=submission,
                recency_hours=(
                    submission.submitted_at - submission.evidence_observed_at
                ).total_seconds()
                / 3600,
                type_match=expected_type is None or submission.evidence_type == expected_type,
            )
            session.add(
                EvidenceSubmissionRow(
                    evidence_id=evidence.evidence_id,
                    hold_id=hold_id,
                    request_id=submission.request_id,
                    evidence_payload=submission.model_dump(mode="json"),
                    recency_hours=evidence.recency_hours,
                    type_match=evidence.type_match,
                    created_at=evidence.created_at,
                )
            )
            hold.evidence_round = 1
            hold.state = HoldState.EVIDENCE_SUBMITTED
            hold.updated_at = utc_now()
            try:
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                existing = await session.scalar(
                    select(EvidenceSubmissionRow).where(
                        EvidenceSubmissionRow.request_id == submission.request_id
                    )
                )
                if existing is not None:
                    if existing.hold_id != hold_id:
                        raise InvalidTransition(
                            "evidence request_id already belongs to another hold"
                        ) from exc
                    return _evidence_from_row(existing), False
                raise InvalidTransition(
                    "evidence has already been submitted for this hold"
                ) from exc
            return evidence, True

    async def get_evidence(self, hold_id: UUID) -> EvidenceRecord:
        async with self._sessions() as session:
            row = await session.scalar(
                select(EvidenceSubmissionRow).where(EvidenceSubmissionRow.hold_id == hold_id)
            )
            if row is None:
                raise InvalidTransition("no evidence has been submitted")
            return _evidence_from_row(row)

    async def record_analyst_review(
        self, hold_id: UUID, submission: AnalystReviewSubmission, analyst_id: str
    ) -> tuple[AnalystReviewRecord, bool]:
        async with self._sessions() as session:
            existing = await session.scalar(
                select(AnalystReviewRow).where(AnalystReviewRow.request_id == submission.request_id)
            )
            if existing is not None:
                if existing.hold_id != hold_id:
                    raise InvalidTransition("review request_id already belongs to another hold")
                return _review_from_row(existing), False
            hold = await _get_hold_row(session, hold_id, for_update=True)
            if hold.state not in {HoldState.HUMAN_REVIEW, HoldState.AWAITING_EVIDENCE}:
                raise InvalidTransition("human actions require a pending review or evidence case")
            latest = await session.scalar(
                select(AuditEventRow)
                .where(AuditEventRow.case_id == hold_id)
                .order_by(AuditEventRow.sequence_no.desc())
                .limit(1)
            )
            previous_hash = latest.record_hash if latest is not None else GENESIS_HASH
            sequence_no = (latest.sequence_no + 1) if latest is not None else 1
            now = utc_now()
            review_id = uuid4()
            action = submission.resolved_action()
            decision = submission.resolved_decision()
            model_decision = await session.scalar(
                select(ModelDecisionRow)
                .where(ModelDecisionRow.hold_id == hold_id)
                .order_by(ModelDecisionRow.created_at.desc())
                .limit(1)
            )
            if model_decision is None:
                raise InvalidTransition("hold has no AI recommendation")
            ai_recommendation = HoldDecision(model_decision.decision)
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
            training_label: Literal["LEGITIMATE", "RISKY", "UNRESOLVED"] = "UNRESOLVED"
            if submission.outcome == AnalystOutcome.FRAUD_CONFIRMED:
                training_label = "RISKY"
            elif decision == HoldDecision.RELEASE:
                training_label = "LEGITIMATE"
            event_id = uuid4()
            event = {
                "schema_version": "2.0",
                "event_id": str(event_id),
                "case_id": str(hold_id),
                "sequence_no": sequence_no,
                "trace_id": "analyst-" + review_id.hex,
                "event_type": "ANALYST_OUTCOME_RECORDED",
                "timestamp": now.isoformat().replace("+00:00", "Z"),
                "payload": {
                    "review_id": str(review_id),
                    "analyst_id": analyst_id,
                    "outcome": submission.outcome,
                    "action": action,
                    "ai_recommendation": ai_recommendation,
                    "authorized_decision": decision,
                    "override": action == HumanReviewAction.OVERRIDE_AI,
                    "reason_code": submission.reason_code,
                    "rationale": submission.rationale,
                    "training_label": training_label,
                    "policy_reference": "human-review@1",
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
            }
            record_hash = hash_event(event, previous_hash)
            audit_record = {**event, "previous_hash": previous_hash, "record_hash": record_hash}
            review = AnalystReviewRecord(
                review_id=review_id,
                hold_id=hold_id,
                analyst_id=analyst_id,
                submission=submission,
                training_label=training_label,
                audit_head_hash=record_hash,
                created_at=now,
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
            hold.updated_at = now
            session.add_all(
                [
                    AnalystReviewRow(
                        review_id=review_id,
                        hold_id=hold_id,
                        request_id=submission.request_id,
                        analyst_id=analyst_id,
                        review_payload=submission.model_dump(mode="json"),
                        training_label=training_label,
                        audit_head_hash=record_hash,
                        created_at=now,
                    ),
                    AuditEventRow(
                        event_id=event_id,
                        case_id=hold_id,
                        sequence_no=sequence_no,
                        event_payload=audit_record,
                        canonical_payload=canonical_json(audit_record).decode("utf-8"),
                        previous_hash=previous_hash,
                        record_hash=record_hash,
                        trace_id=event["trace_id"],
                        created_at=now,
                    ),
                    OutboxEventRow(
                        topic="razortrust.audit.analyst-outcome",
                        payload=audit_record,
                        created_at=now,
                        published_at=None,
                    ),
                ]
            )
            await session.commit()
            return review, True

    async def export_analyst_training_examples(self) -> list[AnalystTrainingExample]:
        examples: list[AnalystTrainingExample] = []
        async with self._sessions() as session:
            reviews = (await session.scalars(select(AnalystReviewRow))).all()
            for review in reviews:
                if review.training_label == "UNRESOLVED":
                    continue
                snapshot = await session.scalar(
                    select(FeatureSnapshotRow)
                    .where(FeatureSnapshotRow.hold_id == review.hold_id)
                    .order_by(FeatureSnapshotRow.created_at.desc())
                    .limit(1)
                )
                if snapshot is None:
                    raise InvalidTransition("reviewed hold is missing its feature snapshot")
                submission = AnalystReviewSubmission.model_validate(review.review_payload)
                evaluation_input = HoldEvaluationInput.model_validate(snapshot.evaluation_input)
                examples.append(
                    AnalystTrainingExample(
                        hold_id=review.hold_id,
                        features=FeatureVector.model_validate(snapshot.features),
                        cohort=evaluation_input.cohort,
                        training_label=cast(Literal["LEGITIMATE", "RISKY"], review.training_label),
                        outcome_reason_code=submission.reason_code,
                        decided_at=submission.decided_at,
                        feature_schema_version=snapshot.feature_schema_version,
                    )
                )
        return examples


async def _get_hold_row(
    session: AsyncSession,
    hold_id: UUID,
    *,
    for_update: bool = False,
) -> HoldCaseRow:
    query = select(HoldCaseRow).where(HoldCaseRow.hold_id == hold_id)
    if for_update:
        query = query.with_for_update()
    row = await session.scalar(query)
    if row is None:
        raise HoldNotFound(str(hold_id))
    return row


def _case_from_row(row: HoldCaseRow) -> HoldCase:
    return HoldCase(
        hold_id=row.hold_id,
        request_id=row.request_id,
        merchant_id=row.merchant_id,
        source_event_id=row.source_event_id,
        triggered_at=_as_utc(row.triggered_at),
        reason_code=row.reason_code,
        state=HoldState(row.state),
        evidence_round=row.evidence_round,
        created_at=_as_utc(row.created_at),
        updated_at=_as_utc(row.updated_at),
    )


def _evidence_from_row(row: EvidenceSubmissionRow) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=row.evidence_id,
        hold_id=row.hold_id,
        submission=EvidenceSubmission.model_validate(row.evidence_payload),
        recency_hours=row.recency_hours,
        type_match=row.type_match,
        created_at=_as_utc(row.created_at),
    )


def _review_from_row(row: AnalystReviewRow) -> AnalystReviewRecord:
    return AnalystReviewRecord(
        review_id=row.review_id,
        hold_id=row.hold_id,
        analyst_id=row.analyst_id,
        submission=AnalystReviewSubmission.model_validate(row.review_payload),
        training_label=cast(Literal["LEGITIMATE", "RISKY", "UNRESOLVED"], row.training_label),
        audit_head_hash=row.audit_head_hash,
        created_at=_as_utc(row.created_at),
    )


def _state_for_decision(decision: HoldDecision, evidence_round: int) -> HoldState:
    if decision == HoldDecision.RELEASE:
        return HoldState.HUMAN_REVIEW
    if decision == HoldDecision.ESCALATE:
        return HoldState.HUMAN_REVIEW
    if evidence_round == 1:
        raise InvalidTransition("a second evidence round is not allowed")
    return HoldState.AWAITING_EVIDENCE


def _as_utc(value: datetime) -> datetime:
    """Normalize driver-returned timestamps; SQLite drops timezone metadata in tests."""
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
