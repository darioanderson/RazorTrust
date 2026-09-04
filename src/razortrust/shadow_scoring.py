from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import Field
from sqlalchemy import JSON, Boolean, DateTime, Float, String, Uuid, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base
from .domain import FeatureContribution, FeatureVector, HoldDecision, ProbabilityVector, StrictModel
from .features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION
from .live_features import (
    BASELINE_METHOD_VERSION,
    CHARGEBACK_MAPPING_VERSION,
    KNOWLEDGE_TIME_POLICY,
    SOURCE_CONTRACT_VERSION,
    FeatureContractPreview,
)
from .risk import DEFAULT_THRESHOLDS, CostPolicy, RiskModel, UnavailableRiskModel

SHADOW_SCORE_CONTRACT_VERSION = "shadow-score-contract@1"
SHADOW_POLICY_MODE = "OBSERVE_ONLY"
PRODUCTION_ACTION_BLOCKER = (
    "SHADOW_ONLY; CHAMPION_IS_SYNTHETIC_MECHANICS_RELEASE; "
    "LIVE ACTION REQUIRES MATURED LABELS, SHADOW VALIDATION, AND EXPLICIT PROMOTION"
)


class ShadowScoreResponse(StrictModel):
    provider: str = "RAZORPAY"
    shadow_score_id: UUID
    source_mode: Literal["LIVE", "SYNTHETIC_FIXTURE_MECHANICS_ONLY"]
    source_id: str
    account_id: str
    as_of: datetime
    status: Literal["BLOCKED", "SCORED"]
    blockers: list[str]
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    feature_vector: dict[str, float] | None = None
    feature_vector_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    feature_provenance: dict[str, dict[str, object]] | None = None
    model_version: str
    calibration_method: str | None = None
    probabilities: ProbabilityVector | None = None
    # Empirical percentile retained for backward-compatible policy behavior.
    anomaly_score: float | None = Field(default=None, ge=0, le=1)
    anomaly_raw_score: float | None = None
    anomaly_reference_max: float | None = None
    anomaly_tail_excess: float | None = Field(default=None, ge=0)
    anomaly_reference_size: int | None = Field(default=None, ge=1)
    anomaly_model_version: str | None = None
    anomaly_reference_mode: str | None = None
    novelty_override: bool = False
    top_features: list[FeatureContribution] = Field(default_factory=list)
    shadow_recommendation: HoldDecision | None = None
    expected_cost_units: float | None = None
    policy_version: str
    shadow_policy_mode: str = SHADOW_POLICY_MODE
    production_action_eligible: bool = False
    production_action_blocker: str = PRODUCTION_ACTION_BLOCKER
    score_contract_version: str = SHADOW_SCORE_CONTRACT_VERSION
    created_at: datetime
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ShadowScoreStats(StrictModel):
    total_runs: int = 0
    scored_runs: int = 0
    blocked_runs: int = 0
    live_runs: int = 0
    fixture_runs: int = 0
    last_run_at: datetime | None = None


class ShadowScoreRow(Base):
    __tablename__ = "razorpay_shadow_score_runs"

    shadow_score_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    source_mode: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    source_id: Mapped[str] = mapped_column(String(160), nullable=False, index=True)
    account_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    blockers: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    feature_schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    feature_vector: Mapped[dict[str, float] | None] = mapped_column(JSON, nullable=True)
    feature_vector_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    feature_provenance: Mapped[dict[str, dict[str, object]] | None] = mapped_column(
        JSON, nullable=True
    )
    model_version: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    calibration_method: Mapped[str | None] = mapped_column(String(64), nullable=True)
    probabilities: Mapped[dict[str, float] | None] = mapped_column(JSON, nullable=True)
    anomaly_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    anomaly_raw_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    anomaly_reference_max: Mapped[float | None] = mapped_column(Float, nullable=True)
    anomaly_tail_excess: Mapped[float | None] = mapped_column(Float, nullable=True)
    anomaly_reference_size: Mapped[int | None] = mapped_column(nullable=True)
    anomaly_model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    anomaly_reference_mode: Mapped[str | None] = mapped_column(String(64), nullable=True)
    novelty_override: Mapped[bool] = mapped_column(Boolean, nullable=False)
    top_features: Mapped[list[dict[str, object]]] = mapped_column(JSON, nullable=False)
    shadow_recommendation: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    expected_cost_units: Mapped[float | None] = mapped_column(Float, nullable=True)
    policy_version: Mapped[str] = mapped_column(String(128), nullable=False)
    shadow_policy_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    production_action_eligible: Mapped[bool] = mapped_column(Boolean, nullable=False)
    production_action_blocker: Mapped[str] = mapped_column(String(512), nullable=False)
    score_contract_version: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    payload_sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _policy_version(risk_model: RiskModel) -> str:
    bundle = getattr(risk_model, "bundle", None)
    return str(getattr(bundle, "policy_version", "shadow-cost-policy@1"))


def _live_feature_provenance(preview: FeatureContractPreview) -> dict[str, dict[str, object]]:
    common: dict[str, object] = {
        "knowledge_time_policy": KNOWLEDGE_TIME_POLICY,
        "as_of": preview.as_of.isoformat(),
        "current_window_start": preview.current_window_start.isoformat(),
        "baseline_start": preview.baseline_start.isoformat(),
        "baseline_end": preview.baseline_end.isoformat(),
        "baseline_method_version": BASELINE_METHOD_VERSION,
        "source_contract_version": SOURCE_CONTRACT_VERSION,
    }
    specs = {
        "volume_delta_z": (
            ["processor_payment_observations"],
            "current 24h payment volume vs aligned 30d daily baseline",
        ),
        "gmv_delta_z": (
            ["processor_payment_observations"],
            "current 24h captured/attempted GMV vs aligned 30d daily baseline",
        ),
        "ticket_size_delta_z": (
            ["processor_payment_observations"],
            "current ticket-size mean vs historical ticket-size baseline",
        ),
        "new_device_ratio": (
            ["processor_payment_observations", "first_party_transaction_telemetry"],
            "share of current transactions on device pseudonyms unseen in baseline",
        ),
        "new_geo_ratio": (
            ["processor_payment_observations", "first_party_transaction_telemetry"],
            "share of current transactions in coarse geographies unseen in baseline",
        ),
        "refund_rate_delta_z": (
            ["processor_payment_observations", "processor_refund_observations"],
            "current refund rate vs point-in-time historical refund-rate baseline",
        ),
        "chargeback_rate_delta_z": (
            ["processor_payment_observations", "processor_dispute_observations"],
            f"current chargeback rate vs historical baseline using {CHARGEBACK_MAPPING_VERSION}",
        ),
        "failed_auth_ratio": (
            [
                "processor_payment_observations",
                "first_party_transaction_telemetry",
                "razorpay_checkout_payment_attempts",
            ],
            "current failed payment attempts divided by mapped payment attempts",
        ),
        "volume_trend_slope": (
            ["processor_payment_observations"],
            "time-ordered current-window transaction-volume trend",
        ),
        "interarrival_time_cv": (
            ["processor_payment_observations"],
            "coefficient of variation of current-window interarrival times",
        ),
        "device_entropy": (
            ["processor_payment_observations", "first_party_transaction_telemetry"],
            "Shannon entropy of current-window account-scoped device pseudonyms",
        ),
        "geo_entropy": (
            ["processor_payment_observations", "first_party_transaction_telemetry"],
            "Shannon entropy of current-window coarse geographies",
        ),
        "amount_distribution_kl": (
            ["processor_payment_observations"],
            "KL divergence of current amount distribution from historical baseline bins",
        ),
    }
    return {
        feature: {
            **common,
            "source_tables": tables,
            "derivation": derivation,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
        }
        for feature, (tables, derivation) in specs.items()
    }


def fixture_feature_vector(name: str) -> FeatureVector:
    fixtures: dict[str, dict[str, float]] = {
        "normal_baseline": {
            "volume_delta_z": 0.15,
            "gmv_delta_z": 0.10,
            "ticket_size_delta_z": 0.05,
            "new_device_ratio": 0.05,
            "new_geo_ratio": 0.02,
            "refund_rate_delta_z": -0.10,
            "chargeback_rate_delta_z": -0.10,
            "failed_auth_ratio": 0.02,
            "volume_trend_slope": 0.08,
            "interarrival_time_cv": 0.75,
            "device_entropy": 1.20,
            "geo_entropy": 0.70,
            "amount_distribution_kl": 0.05,
        },
        "novel_risk_burst": {
            "volume_delta_z": 4.10,
            "gmv_delta_z": 4.40,
            "ticket_size_delta_z": 2.60,
            "new_device_ratio": 0.85,
            "new_geo_ratio": 0.70,
            "refund_rate_delta_z": 3.10,
            "chargeback_rate_delta_z": 3.60,
            "failed_auth_ratio": 0.55,
            "volume_trend_slope": 2.20,
            "interarrival_time_cv": 0.25,
            "device_entropy": 3.20,
            "geo_entropy": 2.50,
            "amount_distribution_kl": 1.60,
        },
    }
    if name not in fixtures:
        raise ValueError(f"unknown fixture: {name}")
    return FeatureVector.model_validate(fixtures[name])


class SqlShadowScoreStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def healthcheck(self) -> None:
        async with self._sessions() as session:
            await session.execute(select(func.count()).select_from(ShadowScoreRow))

    async def persist(self, response: ShadowScoreResponse) -> ShadowScoreResponse:
        row = ShadowScoreRow(
            shadow_score_id=response.shadow_score_id,
            source_mode=response.source_mode,
            source_id=response.source_id,
            account_id=response.account_id,
            as_of=response.as_of,
            status=response.status,
            blockers=response.blockers,
            feature_schema_version=response.feature_schema_version,
            feature_vector=response.feature_vector,
            feature_vector_sha256=response.feature_vector_sha256,
            feature_provenance=response.feature_provenance,
            model_version=response.model_version,
            calibration_method=response.calibration_method,
            probabilities=(response.probabilities.model_dump() if response.probabilities else None),
            anomaly_score=response.anomaly_score,
            anomaly_raw_score=response.anomaly_raw_score,
            anomaly_reference_max=response.anomaly_reference_max,
            anomaly_tail_excess=response.anomaly_tail_excess,
            anomaly_reference_size=response.anomaly_reference_size,
            anomaly_model_version=response.anomaly_model_version,
            anomaly_reference_mode=response.anomaly_reference_mode,
            novelty_override=response.novelty_override,
            top_features=[item.model_dump(mode="json") for item in response.top_features],
            shadow_recommendation=(
                response.shadow_recommendation.value if response.shadow_recommendation else None
            ),
            expected_cost_units=response.expected_cost_units,
            policy_version=response.policy_version,
            shadow_policy_mode=response.shadow_policy_mode,
            production_action_eligible=False,
            production_action_blocker=response.production_action_blocker,
            score_contract_version=response.score_contract_version,
            created_at=response.created_at,
            payload_sha256=response.payload_sha256,
        )
        async with self._sessions() as session:
            session.add(row)
            await session.commit()
        return response

    async def stats(self) -> ShadowScoreStats:
        async with self._sessions() as session:
            total = int(await session.scalar(select(func.count()).select_from(ShadowScoreRow)) or 0)
            scored = int(
                await session.scalar(
                    select(func.count())
                    .select_from(ShadowScoreRow)
                    .where(ShadowScoreRow.status == "SCORED")
                )
                or 0
            )
            blocked = int(
                await session.scalar(
                    select(func.count())
                    .select_from(ShadowScoreRow)
                    .where(ShadowScoreRow.status == "BLOCKED")
                )
                or 0
            )
            live = int(
                await session.scalar(
                    select(func.count())
                    .select_from(ShadowScoreRow)
                    .where(ShadowScoreRow.source_mode == "LIVE")
                )
                or 0
            )
            fixtures = int(
                await session.scalar(
                    select(func.count())
                    .select_from(ShadowScoreRow)
                    .where(ShadowScoreRow.source_mode == "SYNTHETIC_FIXTURE_MECHANICS_ONLY")
                )
                or 0
            )
            last = await session.scalar(select(func.max(ShadowScoreRow.created_at)))
        return ShadowScoreStats(
            total_runs=total,
            scored_runs=scored,
            blocked_runs=blocked,
            live_runs=live,
            fixture_runs=fixtures,
            last_run_at=last,
        )


class ShadowScoringService:
    def __init__(
        self,
        store: SqlShadowScoreStore,
        risk_model: RiskModel,
        cost_policy: CostPolicy,
    ) -> None:
        self._store = store
        self._risk_model = risk_model
        self._cost_policy = cost_policy

    async def score_live_preview(self, preview: FeatureContractPreview) -> ShadowScoreResponse:
        blockers = list(preview.blockers)
        if isinstance(self._risk_model, UnavailableRiskModel):
            blockers.append("MODEL_UNAVAILABLE")
        if self._risk_model.version.startswith("development-rules@"):
            blockers.append("TRAINED_SIGNED_MODEL_REQUIRED")
        if preview.feature_vector is None or preview.feature_vector_sha256 is None:
            return await self._persist_blocked(
                source_mode="LIVE",
                source_id=f"feature-contract:{preview.account_id}:{preview.as_of.isoformat()}",
                account_id=preview.account_id,
                as_of=preview.as_of,
                blockers=sorted(set(blockers or ["FEATURE_VECTOR_UNAVAILABLE"])),
                feature_vector=None,
                feature_vector_sha256=None,
                feature_provenance=None,
            )
        if blockers:
            return await self._persist_blocked(
                source_mode="LIVE",
                source_id=f"feature-contract:{preview.account_id}:{preview.as_of.isoformat()}",
                account_id=preview.account_id,
                as_of=preview.as_of,
                blockers=sorted(set(blockers)),
                feature_vector=preview.feature_vector,
                feature_vector_sha256=preview.feature_vector_sha256,
                feature_provenance=_live_feature_provenance(preview),
            )
        vector = FeatureVector.model_validate(preview.feature_vector)
        if tuple(vector.model_dump()) != FEATURE_COLUMNS:
            raise RuntimeError(
                "shadow scorer received a feature vector outside the locked contract"
            )
        return await self._score_and_persist(
            source_mode="LIVE",
            source_id=f"feature-contract:{preview.account_id}:{preview.as_of.isoformat()}",
            account_id=preview.account_id,
            as_of=preview.as_of,
            vector=vector,
            feature_vector_sha256=preview.feature_vector_sha256,
            provenance=_live_feature_provenance(preview),
        )

    async def score_fixture(self, *, fixture_name: str, as_of: datetime) -> ShadowScoreResponse:
        if isinstance(self._risk_model, UnavailableRiskModel):
            return await self._persist_blocked(
                source_mode="SYNTHETIC_FIXTURE_MECHANICS_ONLY",
                source_id=fixture_name,
                account_id=f"fixture:{fixture_name}",
                as_of=as_of,
                blockers=["MODEL_UNAVAILABLE"],
                feature_vector=None,
                feature_vector_sha256=None,
                feature_provenance=None,
            )
        if self._risk_model.version.startswith("development-rules@"):
            return await self._persist_blocked(
                source_mode="SYNTHETIC_FIXTURE_MECHANICS_ONLY",
                source_id=fixture_name,
                account_id=f"fixture:{fixture_name}",
                as_of=as_of,
                blockers=["TRAINED_SIGNED_MODEL_REQUIRED"],
                feature_vector=None,
                feature_vector_sha256=None,
                feature_provenance=None,
            )
        vector = fixture_feature_vector(fixture_name)
        dumped = vector.model_dump()
        feature_hash = _canonical_sha256(
            {
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "source_mode": "SYNTHETIC_FIXTURE_MECHANICS_ONLY",
                "fixture_name": fixture_name,
                "features": dumped,
            }
        )
        provenance: dict[str, dict[str, object]] = {
            feature: {
                "source": "SYNTHETIC_FIXTURE_MECHANICS_ONLY",
                "fixture_name": fixture_name,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "claim_restriction": "mechanics validation only; not real Razorpay performance",
            }
            for feature in FEATURE_COLUMNS
        }
        return await self._score_and_persist(
            source_mode="SYNTHETIC_FIXTURE_MECHANICS_ONLY",
            source_id=fixture_name,
            account_id=f"fixture:{fixture_name}",
            as_of=as_of,
            vector=vector,
            feature_vector_sha256=feature_hash,
            provenance=provenance,
        )

    async def _score_and_persist(
        self,
        *,
        source_mode: Literal["LIVE", "SYNTHETIC_FIXTURE_MECHANICS_ONLY"],
        source_id: str,
        account_id: str,
        as_of: datetime,
        vector: FeatureVector,
        feature_vector_sha256: str,
        provenance: dict[str, dict[str, object]],
    ) -> ShadowScoreResponse:
        prediction = self._risk_model.predict(vector, cohort="pooled")
        thresholds = getattr(self._risk_model, "thresholds", DEFAULT_THRESHOLDS)
        recommendation, expected_cost = self._cost_policy.choose(
            prediction,
            evidence_round=0,
            thresholds=thresholds,
        )
        now = datetime.now(UTC)
        base_payload = {
            "source_mode": source_mode,
            "source_id": source_id,
            "account_id": account_id,
            "as_of": as_of.isoformat(),
            "feature_vector_sha256": feature_vector_sha256,
            "model_version": prediction.model_version,
            "calibration_method": prediction.calibration_method,
            "probabilities": prediction.probabilities.model_dump(),
            "anomaly_score": prediction.anomaly_score,
            "anomaly_raw_score": prediction.anomaly_raw_score,
            "anomaly_reference_max": prediction.anomaly_reference_max,
            "anomaly_tail_excess": prediction.anomaly_tail_excess,
            "anomaly_reference_size": prediction.anomaly_reference_size,
            "anomaly_model_version": prediction.anomaly_model_version,
            "anomaly_reference_mode": prediction.anomaly_reference_mode,
            "novelty_override": prediction.novelty_override,
            "shadow_recommendation": recommendation.value,
            "policy_version": _policy_version(self._risk_model),
            "score_contract_version": SHADOW_SCORE_CONTRACT_VERSION,
            "created_at": now.isoformat(),
        }
        response = ShadowScoreResponse(
            shadow_score_id=uuid4(),
            source_mode=source_mode,
            source_id=source_id,
            account_id=account_id,
            as_of=as_of,
            status="SCORED",
            blockers=[],
            feature_vector=vector.model_dump(),
            feature_vector_sha256=feature_vector_sha256,
            feature_provenance=provenance,
            model_version=prediction.model_version,
            calibration_method=prediction.calibration_method,
            probabilities=prediction.probabilities,
            anomaly_score=prediction.anomaly_score,
            anomaly_raw_score=prediction.anomaly_raw_score,
            anomaly_reference_max=prediction.anomaly_reference_max,
            anomaly_tail_excess=prediction.anomaly_tail_excess,
            anomaly_reference_size=prediction.anomaly_reference_size,
            anomaly_model_version=prediction.anomaly_model_version,
            anomaly_reference_mode=prediction.anomaly_reference_mode,
            novelty_override=prediction.novelty_override,
            top_features=prediction.top_features,
            shadow_recommendation=recommendation,
            expected_cost_units=expected_cost,
            policy_version=_policy_version(self._risk_model),
            created_at=now,
            payload_sha256=_canonical_sha256(base_payload),
        )
        return await self._store.persist(response)

    async def _persist_blocked(
        self,
        *,
        source_mode: Literal["LIVE", "SYNTHETIC_FIXTURE_MECHANICS_ONLY"],
        source_id: str,
        account_id: str,
        as_of: datetime,
        blockers: list[str],
        feature_vector: dict[str, float] | None,
        feature_vector_sha256: str | None,
        feature_provenance: dict[str, dict[str, object]] | None,
    ) -> ShadowScoreResponse:
        now = datetime.now(UTC)
        base_payload = {
            "source_mode": source_mode,
            "source_id": source_id,
            "account_id": account_id,
            "as_of": as_of.isoformat(),
            "status": "BLOCKED",
            "blockers": blockers,
            "feature_vector_sha256": feature_vector_sha256,
            "model_version": self._risk_model.version,
            "created_at": now.isoformat(),
        }
        response = ShadowScoreResponse(
            shadow_score_id=uuid4(),
            source_mode=source_mode,
            source_id=source_id,
            account_id=account_id,
            as_of=as_of,
            status="BLOCKED",
            blockers=blockers,
            feature_vector=feature_vector,
            feature_vector_sha256=feature_vector_sha256,
            feature_provenance=feature_provenance,
            model_version=self._risk_model.version,
            policy_version=_policy_version(self._risk_model),
            created_at=now,
            payload_sha256=_canonical_sha256(base_payload),
        )
        return await self._store.persist(response)
