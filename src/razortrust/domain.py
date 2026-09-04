from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from itertools import pairwise
from typing import Any, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)


def utc_now() -> datetime:
    return datetime.now(UTC)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class HoldDecision(StrEnum):
    RELEASE = "RELEASE"
    EVIDENCE_NEEDED = "EVIDENCE_NEEDED"
    ESCALATE = "ESCALATE"


class HoldState(StrEnum):
    OPEN = "OPEN"
    AWAITING_EVIDENCE = "AWAITING_EVIDENCE"
    EVIDENCE_SUBMITTED = "EVIDENCE_SUBMITTED"
    RESOLVED = "RESOLVED"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    DENIED = "DENIED"


class ActorRole(StrEnum):
    MERCHANT = "MERCHANT"
    RISK_ANALYST = "RISK_ANALYST"
    RISK_SERVICE = "RISK_SERVICE"
    ADMIN = "ADMIN"
    EVIDENCE_SERVICE = "EVIDENCE_SERVICE"


class Principal(StrictModel):
    id: str = Field(min_length=1, max_length=128)
    role: ActorRole
    merchant_id: str | None = Field(default=None, max_length=128)


class AuthorizationRequest(StrictModel):
    principal: Principal
    action: str
    resource_type: str
    resource_id: str | None = None
    merchant_id: str | None = None


class AuthorizationDecision(StrictModel):
    allowed: bool
    reasons: list[str]
    policy_version: str


class DecisionReason(StrEnum):
    MODEL_POLICY = "MODEL_POLICY"
    NOVELTY_DISAGREEMENT = "NOVELTY_DISAGREEMENT"
    POLICY_GUARDRAIL = "POLICY_GUARDRAIL"
    MODEL_ERROR = "MODEL_ERROR"
    HUMAN_ONLY = "HUMAN_ONLY"
    POLICY_ERROR = "POLICY_ERROR"
    EVIDENCE_EXHAUSTED = "EVIDENCE_EXHAUSTED"


class EvidenceType(StrEnum):
    CAMPAIGN = "CAMPAIGN"
    FULFILLMENT = "FULFILLMENT"
    GEO_EXPANSION = "GEO_EXPANSION"
    AUTH_INCIDENT = "AUTH_INCIDENT"
    INVENTORY = "INVENTORY"
    OTHER = "OTHER"


class HoldCreate(StrictModel):
    request_id: UUID
    merchant_id: str = Field(min_length=1, max_length=128)
    source_event_id: str = Field(min_length=1, max_length=128)
    triggered_at: datetime
    reason_code: str | None = Field(default=None, max_length=128)

    @field_validator("triggered_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("triggered_at must include a timezone")
        return value.astimezone(UTC)


class HoldCase(StrictModel):
    hold_id: UUID
    request_id: UUID
    merchant_id: str
    source_event_id: str
    triggered_at: datetime
    reason_code: str | None
    state: HoldState = HoldState.OPEN
    evidence_round: int = Field(default=0, ge=0, le=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class MerchantBaseline(StrictModel):
    volume_mean: float = Field(gt=0)
    volume_std: float = Field(gt=0)
    gmv_mean: float = Field(gt=0)
    gmv_std: float = Field(gt=0)
    ticket_size_mean: float = Field(gt=0)
    ticket_size_std: float = Field(gt=0)
    refund_rate_mean: float = Field(ge=0, le=1)
    refund_rate_std: float = Field(gt=0)
    chargeback_rate_mean: float = Field(ge=0, le=1)
    chargeback_rate_std: float = Field(gt=0)
    known_devices: set[str] = Field(default_factory=set)
    known_geos: set[str] = Field(default_factory=set)
    amount_bin_edges: list[float]
    amount_bin_probabilities: list[float]

    @field_serializer("known_devices", "known_geos")
    def serialize_sets(self, value: set[str]) -> list[str]:
        return sorted(value)

    @model_validator(mode="after")
    def validate_histogram(self) -> MerchantBaseline:
        if len(self.amount_bin_edges) != len(self.amount_bin_probabilities) + 1:
            raise ValueError(
                "amount_bin_edges must have one more item than amount_bin_probabilities"
            )
        if any(right <= left for left, right in pairwise(self.amount_bin_edges)):
            raise ValueError("amount_bin_edges must be strictly increasing")
        if any(value < 0 for value in self.amount_bin_probabilities):
            raise ValueError("amount_bin_probabilities cannot be negative")
        total = sum(self.amount_bin_probabilities)
        if not 0.999 <= total <= 1.001:
            raise ValueError("amount_bin_probabilities must sum to 1")
        return self


class TransactionEvent(StrictModel):
    transaction_id: str = Field(min_length=1, max_length=128)
    merchant_id: str = Field(min_length=1, max_length=128)
    customer_id: str | None = Field(default=None, min_length=1, max_length=128)
    ring_id: str | None = Field(default=None, min_length=1, max_length=128)
    timestamp: datetime
    amount: float = Field(gt=0)
    device_fingerprint: str = Field(min_length=1, max_length=256)
    customer_geo: str = Field(min_length=1, max_length=64)
    auth_status: Literal["APPROVED", "FAILED"]
    refund_timestamp: datetime | None = None
    chargeback_timestamp: datetime | None = None

    @field_validator("timestamp", "refund_timestamp", "chargeback_timestamp")
    @classmethod
    def normalize_timestamps(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("transaction timestamps must include a timezone")
        return value.astimezone(UTC)


class HoldEvaluationInput(StrictModel):
    baseline: MerchantBaseline
    transactions: list[TransactionEvent]
    cohort: str = Field(default="pooled", min_length=1, max_length=128)
    window_hours: int = Field(default=24, ge=1, le=168)


class FeatureVector(StrictModel):
    volume_delta_z: float
    gmv_delta_z: float
    ticket_size_delta_z: float
    new_device_ratio: float = Field(ge=0, le=1)
    new_geo_ratio: float = Field(ge=0, le=1)
    refund_rate_delta_z: float
    chargeback_rate_delta_z: float
    failed_auth_ratio: float = Field(ge=0, le=1)
    volume_trend_slope: float
    interarrival_time_cv: float = Field(ge=0)
    device_entropy: float = Field(ge=0)
    geo_entropy: float = Field(ge=0)
    amount_distribution_kl: float = Field(ge=0)


class ProbabilityVector(StrictModel):
    release: float = Field(ge=0, le=1)
    evidence_needed: float = Field(ge=0, le=1)
    escalate: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def probabilities_sum_to_one(self) -> ProbabilityVector:
        if abs(self.release + self.evidence_needed + self.escalate - 1.0) > 1e-6:
            raise ValueError("probabilities must sum to 1")
        return self


class Thresholds(StrictModel):
    release: float = Field(ge=0, le=1)
    escalate: float = Field(ge=0, le=1)


class FeatureContribution(StrictModel):
    feature: str
    observed_value: float
    contribution_value: float
    direction: Literal["toward_RELEASE", "toward_EVIDENCE_NEEDED", "toward_ESCALATE"]
    reference: str
    attribution_method: str
    model_output_explained: HoldDecision | None = None
    template_id: str | None = None


class ModelPrediction(StrictModel):
    probabilities: ProbabilityVector
    calibrated: bool
    calibration_method: str
    # Backward-compatible empirical percentile used by existing policy thresholds.
    anomaly_score: float = Field(ge=0, le=1)
    anomaly_raw_score: float | None = None
    anomaly_reference_max: float | None = None
    anomaly_tail_excess: float | None = Field(default=None, ge=0)
    anomaly_reference_size: int | None = Field(default=None, ge=1)
    anomaly_model_version: str | None = None
    anomaly_reference_mode: str | None = None
    novelty_override: bool = False
    top_features: list[FeatureContribution]
    model_version: str


class EvidenceVerdict(StrEnum):
    SUPPORTED = "SUPPORTED"
    INSUFFICIENT = "INSUFFICIENT"
    CONTRADICTORY = "CONTRADICTORY"


class EvidenceAssessment(StrictModel):
    verdict: EvidenceVerdict
    expected_type: EvidenceType | None
    submitted_type: EvidenceType
    subject_match: bool
    time_match: bool
    verification_mode: Literal["MOCKED", "SIGNED_PROVIDER_ATTESTATION", "UNVERIFIED"] = "UNVERIFIED"
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verifier_version: str = "evidence-verifier@2"
    attestation_key_id: str | None = None


class PolicyCheck(StrictModel):
    hold_id: UUID
    candidate_decision: HoldDecision
    probabilities: ProbabilityVector
    thresholds: Thresholds
    novelty_override: bool
    evidence_round: int = Field(ge=0)
    model_version: str
    policy_version: str
    system_error: bool = False
    evidence_assessment: EvidenceAssessment | None = None
    evidence_release_risk_cap: float = Field(default=0.20, ge=0, le=1)


class PolicyResult(StrictModel):
    allowed_decision: HoldDecision
    guardrail_triggered: bool
    reasons: list[str]
    policy_version: str


class EvidenceRequirement(StrictModel):
    evidence_type: EvidenceType
    title: str = Field(min_length=1, max_length=160)
    description: str = Field(min_length=1, max_length=500)


class MerchantGuidance(StrictModel):
    summary: str = Field(min_length=1, max_length=500)
    reasons: list[str] = Field(min_length=1, max_length=3)
    required_evidence: list[EvidenceRequirement] = Field(min_length=1, max_length=3)
    submit_by: datetime
    next_step: str = Field(min_length=1, max_length=500)
    template_version: str = "merchant-guidance@1"


class DecisionResponse(StrictModel):
    hold_id: UUID
    trace_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    decision: HoldDecision
    probabilities: ProbabilityVector
    calibrated: bool
    calibration_method: str
    anomaly_score: float
    expected_cost_units: float
    cost_matrix_version: str
    cost_matrix_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    thresholds: Thresholds
    top_features: list[FeatureContribution]
    model_version: str
    feature_schema_version: str
    policy_version: str
    evidence_round: int
    novelty_override: bool
    reason_code: DecisionReason
    audit_head_hash: str
    evidence_assessment: EvidenceAssessment | None = None
    merchant_guidance: MerchantGuidance | None = None
    created_at: datetime = Field(default_factory=utc_now)


class EvidenceSubmission(StrictModel):
    request_id: UUID
    evidence_type: EvidenceType
    submitted_at: datetime
    evidence_observed_at: datetime
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("submitted_at", "evidence_observed_at")
    @classmethod
    def evidence_timestamps_require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("evidence timestamps must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def observation_precedes_submission(self) -> EvidenceSubmission:
        if self.evidence_observed_at > self.submitted_at:
            raise ValueError("evidence_observed_at cannot be later than submitted_at")
        forbidden = {"evidence_type_match", "evidence_recency_hours"}.intersection(self.metadata)
        if forbidden:
            names = ", ".join(sorted(forbidden))
            raise ValueError(f"server-derived evidence fields are not accepted: {names}")
        sensitive_names = {
            "content",
            "customer_id",
            "device_fingerprint",
            "document",
            "evidence_text",
            "raw",
        }
        present_sensitive = sensitive_names.intersection(key.lower() for key in self.metadata)
        if present_sensitive:
            names = ", ".join(sorted(present_sensitive))
            raise ValueError(f"evidence content is not accepted in metadata: {names}")
        if len(self.metadata) > 20:
            raise ValueError("evidence metadata cannot contain more than 20 fields")
        for key, value in self.metadata.items():
            if len(key) > 64:
                raise ValueError("evidence metadata keys cannot exceed 64 characters")
            if not isinstance(value, str | int | float | bool | type(None)):
                raise ValueError("evidence metadata values must be scalar")
            if isinstance(value, str) and len(value) > 256:
                raise ValueError("evidence metadata string values cannot exceed 256 characters")
        if len(json.dumps(self.metadata, separators=(",", ":"))) > 2048:
            raise ValueError("evidence metadata cannot exceed 2048 serialized characters")
        return self


class EvidenceRecord(StrictModel):
    evidence_id: UUID
    hold_id: UUID
    submission: EvidenceSubmission
    recency_hours: float = Field(ge=0)
    type_match: bool
    created_at: datetime = Field(default_factory=utc_now)


class AnalystOutcome(StrEnum):
    CLEARED = "CLEARED"
    FRAUD_CONFIRMED = "FRAUD_CONFIRMED"
    CONTINUE_HOLD = "CONTINUE_HOLD"


class HumanReviewAction(StrEnum):
    APPROVE_RELEASE = "APPROVE_RELEASE"
    REQUEST_EVIDENCE = "REQUEST_EVIDENCE"
    ESCALATE = "ESCALATE"
    OVERRIDE_AI = "OVERRIDE_AI"


class AnalystReviewSubmission(StrictModel):
    request_id: UUID
    outcome: AnalystOutcome | None = None
    action: HumanReviewAction | None = None
    authorized_decision: HoldDecision | None = None
    reason_code: str = Field(min_length=1, max_length=128)
    rationale: str = Field(min_length=1, max_length=1000)
    decided_at: datetime
    agent_id: str | None = Field(default=None, min_length=1, max_length=128)
    agent_session: str | None = Field(default=None, min_length=1, max_length=128)
    delegated_permissions: list[str] = Field(default_factory=list, max_length=20)
    authorized_amount: float | None = Field(default=None, ge=0)
    authorized_item: str | None = Field(default=None, min_length=1, max_length=256)
    infrastructure_provider: str | None = Field(default=None, min_length=1, max_length=128)
    transaction_identity: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("decided_at")
    @classmethod
    def decision_timestamp_requires_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("decided_at must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_human_action(self) -> AnalystReviewSubmission:
        if (self.outcome is None) == (self.action is None):
            raise ValueError("provide exactly one of outcome or action")
        if self.action == HumanReviewAction.OVERRIDE_AI and self.authorized_decision is None:
            raise ValueError("authorized_decision is required when overriding AI")
        if self.action != HumanReviewAction.OVERRIDE_AI and self.authorized_decision is not None:
            raise ValueError("authorized_decision is accepted only when overriding AI")
        if len(set(self.delegated_permissions)) != len(self.delegated_permissions):
            raise ValueError("delegated_permissions cannot contain duplicates")
        if any(not value or len(value) > 128 for value in self.delegated_permissions):
            raise ValueError("delegated_permissions entries must contain 1 to 128 characters")
        return self

    def resolved_action(self) -> HumanReviewAction:
        if self.action is not None:
            return self.action
        legacy = {
            AnalystOutcome.CLEARED: HumanReviewAction.APPROVE_RELEASE,
            AnalystOutcome.FRAUD_CONFIRMED: HumanReviewAction.ESCALATE,
            AnalystOutcome.CONTINUE_HOLD: HumanReviewAction.ESCALATE,
        }
        assert self.outcome is not None
        return legacy[self.outcome]

    def resolved_decision(self) -> HoldDecision:
        action = self.resolved_action()
        if action == HumanReviewAction.APPROVE_RELEASE:
            return HoldDecision.RELEASE
        if action == HumanReviewAction.REQUEST_EVIDENCE:
            return HoldDecision.EVIDENCE_NEEDED
        if action == HumanReviewAction.OVERRIDE_AI:
            assert self.authorized_decision is not None
            return self.authorized_decision
        return HoldDecision.ESCALATE


class AnalystReviewRecord(StrictModel):
    review_id: UUID
    hold_id: UUID
    analyst_id: str = Field(min_length=1, max_length=128)
    submission: AnalystReviewSubmission
    training_label: Literal["LEGITIMATE", "RISKY", "UNRESOLVED"]
    policy_reference: str = "human-review@1"
    audit_head_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime = Field(default_factory=utc_now)


class AnalystTrainingExample(StrictModel):
    hold_id: UUID
    features: FeatureVector
    cohort: str
    training_label: Literal["LEGITIMATE", "RISKY"]
    outcome_reason_code: str
    decided_at: datetime
    feature_schema_version: str
