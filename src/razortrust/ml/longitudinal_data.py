from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import fmean, pstdev
from uuid import NAMESPACE_URL, uuid5

import pandas as pd

from ..domain import HoldCase, HoldDecision, HoldEvaluationInput, MerchantBaseline, TransactionEvent
from ..features import FEATURE_COLUMNS, build_point_in_time_features
from ..synthetic import LEGITIMATE_FAMILIES, RISK_FAMILIES

LONGITUDINAL_DATASET_VERSION = "synthetic-v3.1-longitudinal"
LONGITUDINAL_GENERATOR_VERSION = "3.1.0"
DEFAULT_REFERENCE_TIME = datetime(2026, 8, 28, 12, 0, tzinfo=UTC)
COHORTS = ("retail", "travel", "digital_goods")

COHORT_BASE = {
    "retail": {"daily_volume": 8.0, "ticket": 1200.0},
    "travel": {"daily_volume": 4.0, "ticket": 6200.0},
    "digital_goods": {"daily_volume": 12.0, "ticket": 750.0},
}

DIRECT_RELEASE_FAMILIES = {"baseline", "seasonal_peak"}
EVIDENCE_RESOLVABLE_FAMILIES = set(LEGITIMATE_FAMILIES) - DIRECT_RELEASE_FAMILIES


@dataclass(frozen=True)
class LongitudinalConfig:
    seed: int = 20260903
    merchants_per_family: int = 80
    history_days: int = 60
    baseline_days: int = 30
    gap_hours: int = 24
    current_hours: int = 24
    confounder_probability: float = 0.18
    reference_time: datetime = DEFAULT_REFERENCE_TIME

    def validate(self) -> None:
        if self.merchants_per_family < 2:
            raise ValueError("merchants_per_family must be >= 2")
        if self.history_days < self.baseline_days + 2:
            raise ValueError("history_days must exceed baseline_days by at least two days")
        if self.baseline_days < 7:
            raise ValueError("baseline_days must be >= 7")
        if self.gap_hours < 1 or self.current_hours < 1:
            raise ValueError("gap_hours/current_hours must be positive")
        if not 0.0 <= self.confounder_probability <= 0.5:
            raise ValueError("confounder_probability must be in [0, 0.5]")
        if self.reference_time.tzinfo is None or self.reference_time.utcoffset() is None:
            raise ValueError("reference_time must be timezone-aware")


@dataclass(frozen=True)
class MerchantShape:
    merchant_id: str
    family: str
    cohort: str
    base_daily_volume: float
    base_ticket: float
    growth_per_day: float
    refund_rate: float
    chargeback_rate: float
    auth_fail_rate: float
    device_pool_size: int
    home_geos: tuple[str, ...]
    weekday_amplitude: float
    amount_sigma: float


@dataclass(frozen=True)
class FamilyEffect:
    volume_multiplier: float
    amount_multiplier: float
    new_device_rate: float
    new_geo_rate: float
    auth_fail_rate: float | None = None
    refund_rate: float | None = None
    chargeback_rate: float | None = None
    ring_mode: bool = False


@dataclass(frozen=True)
class LongitudinalRecord:
    row: dict[str, object]
    baseline_window_start: datetime
    baseline_window_end: datetime
    current_window_start: datetime
    current_window_end: datetime
    baseline_event_count: int
    current_event_count: int
    processed_event_count: int


def _positive_std(values: list[float], floor: float) -> float:
    if len(values) < 2:
        return floor
    return max(float(pstdev(values)), floor)


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("cannot calculate quantile of empty values")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    left = int(math.floor(pos))
    right = int(math.ceil(pos))
    if left == right:
        return sorted_values[left]
    weight = pos - left
    return sorted_values[left] * (1.0 - weight) + sorted_values[right] * weight


def _histogram(amounts: list[float], edges: list[float]) -> list[float]:
    counts = [0] * (len(edges) - 1)
    for amount in amounts:
        target = len(counts) - 1
        for index in range(len(counts)):
            left, right = edges[index], edges[index + 1]
            if left <= amount < right or (index == len(counts) - 1 and amount <= right):
                target = index
                break
        counts[target] += 1
    smoothing = 1e-3
    total = sum(counts) + smoothing * len(counts)
    return [(count + smoothing) / total for count in counts]


def estimate_baseline_from_history(
    events: list[TransactionEvent], *, baseline_start: datetime, baseline_end: datetime
) -> MerchantBaseline:
    """Estimate the MerchantBaseline from actual historical events known by baseline_end."""
    baseline_events = sorted(
        (event for event in events if baseline_start <= event.timestamp < baseline_end),
        key=lambda event: event.timestamp,
    )
    if not baseline_events:
        raise ValueError("baseline history is empty")

    day_count = max(1, int((baseline_end - baseline_start).total_seconds() // 86400))
    daily_volume: list[float] = []
    daily_gmv: list[float] = []
    daily_ticket: list[float] = []
    daily_refund: list[float] = []
    daily_chargeback: list[float] = []

    for day in range(day_count):
        day_start = baseline_start + timedelta(days=day)
        day_end = min(day_start + timedelta(days=1), baseline_end)
        bucket = [event for event in baseline_events if day_start <= event.timestamp < day_end]
        volume = len(bucket)
        gmv = sum(event.amount for event in bucket)
        daily_volume.append(float(volume))
        daily_gmv.append(float(gmv))
        if bucket:
            daily_ticket.append(float(gmv / volume))
            known_refunds = sum(
                event.refund_timestamp is not None and event.refund_timestamp <= baseline_end
                for event in bucket
            )
            known_chargebacks = sum(
                event.chargeback_timestamp is not None
                and event.chargeback_timestamp <= baseline_end
                for event in bucket
            )
            daily_refund.append(known_refunds / volume)
            daily_chargeback.append(known_chargebacks / volume)
        else:
            daily_ticket.append(0.0)
            daily_refund.append(0.0)
            daily_chargeback.append(0.0)

    amounts = sorted(event.amount for event in baseline_events)
    q25 = _quantile(amounts, 0.25)
    q50 = _quantile(amounts, 0.50)
    q90 = _quantile(amounts, 0.90)
    max_edge = max(_quantile(amounts, 0.995), q90 * 1.5, q50 + 1.0)
    edges = [0.0, max(1.0, q25), max(q25 + 1.0, q50), max(q50 + 1.0, q90), max_edge]
    for index in range(1, len(edges)):
        if edges[index] <= edges[index - 1]:
            edges[index] = edges[index - 1] + 1.0

    nonzero_ticket = [value for value in daily_ticket if value > 0]
    volume_mean = max(fmean(daily_volume), 0.1)
    gmv_mean = max(fmean(daily_gmv), 1.0)
    ticket_mean = max(fmean(nonzero_ticket or amounts), 1.0)

    return MerchantBaseline(
        volume_mean=volume_mean,
        volume_std=_positive_std(daily_volume, max(1.0, volume_mean * 0.10)),
        gmv_mean=gmv_mean,
        gmv_std=_positive_std(daily_gmv, max(100.0, gmv_mean * 0.10)),
        ticket_size_mean=ticket_mean,
        ticket_size_std=_positive_std(nonzero_ticket or amounts, max(10.0, ticket_mean * 0.10)),
        refund_rate_mean=min(max(fmean(daily_refund), 0.0), 1.0),
        refund_rate_std=_positive_std(daily_refund, 0.005),
        chargeback_rate_mean=min(max(fmean(daily_chargeback), 0.0), 1.0),
        chargeback_rate_std=_positive_std(daily_chargeback, 0.0025),
        known_devices={event.device_fingerprint for event in baseline_events},
        known_geos={event.customer_geo for event in baseline_events},
        amount_bin_edges=edges,
        amount_bin_probabilities=_histogram(amounts, edges),
    )


def _merchant_shape(rng: random.Random, *, merchant_id: str, family: str) -> MerchantShape:
    cohort = rng.choice(COHORTS)
    base = COHORT_BASE[cohort]
    daily_volume = max(1.8, rng.lognormvariate(math.log(base["daily_volume"]), 0.32))
    base_ticket = max(60.0, rng.lognormvariate(math.log(base["ticket"]), 0.38))
    return MerchantShape(
        merchant_id=merchant_id,
        family=family,
        cohort=cohort,
        base_daily_volume=daily_volume,
        base_ticket=base_ticket,
        growth_per_day=rng.uniform(-0.0025, 0.0065),
        refund_rate=min(max(rng.betavariate(2.2, 90.0), 0.002), 0.09),
        chargeback_rate=min(max(rng.betavariate(1.5, 240.0), 0.0005), 0.035),
        auth_fail_rate=min(max(rng.betavariate(2.0, 55.0), 0.004), 0.12),
        device_pool_size=rng.randint(6, 24),
        home_geos=tuple(rng.sample(["IN-MH", "IN-KA", "IN-DL", "IN-UP", "IN-TN"], 3)),
        weekday_amplitude=rng.uniform(0.05, 0.32),
        amount_sigma=rng.uniform(0.22, 0.58),
    )


def _family_effect(rng: random.Random, family: str, confounder_probability: float) -> FamilyEffect:
    # The ranges deliberately overlap between legitimate and risky families.
    ranges: dict[str, dict[str, tuple[float, float] | bool]] = {
        "baseline": {
            "volume": (0.85, 1.25),
            "amount": (0.90, 1.15),
            "new_device": (0.03, 0.20),
            "new_geo": (0.01, 0.12),
        },
        "flash_sale": {
            "volume": (1.35, 3.25),
            "amount": (0.75, 1.25),
            "new_device": (0.12, 0.48),
            "new_geo": (0.03, 0.18),
        },
        "influencer_campaign": {
            "volume": (1.25, 2.70),
            "amount": (0.80, 1.30),
            "new_device": (0.28, 0.72),
            "new_geo": (0.04, 0.28),
        },
        "geo_expansion": {
            "volume": (0.95, 1.90),
            "amount": (0.85, 1.25),
            "new_device": (0.08, 0.36),
            "new_geo": (0.28, 0.82),
        },
        "product_launch": {
            "volume": (1.05, 2.60),
            "amount": (1.10, 2.75),
            "new_device": (0.10, 0.48),
            "new_geo": (0.03, 0.24),
        },
        "auth_provider_incident": {
            "volume": (0.80, 1.40),
            "amount": (0.85, 1.15),
            "new_device": (0.03, 0.22),
            "new_geo": (0.01, 0.15),
        },
        "seasonal_peak": {
            "volume": (1.30, 3.10),
            "amount": (0.85, 1.45),
            "new_device": (0.08, 0.40),
            "new_geo": (0.02, 0.22),
        },
        "card_testing_style": {
            "volume": (1.20, 3.20),
            "amount": (0.08, 0.55),
            "new_device": (0.35, 0.95),
            "new_geo": (0.05, 0.45),
        },
        "refund_abuse_style": {
            "volume": (0.85, 1.65),
            "amount": (0.85, 1.45),
            "new_device": (0.04, 0.30),
            "new_geo": (0.01, 0.18),
        },
        "synthetic_volume_spike": {
            "volume": (1.55, 4.40),
            "amount": (0.90, 2.20),
            "new_device": (0.08, 0.50),
            "new_geo": (0.03, 0.25),
        },
        "distributed_device_ring": {
            "volume": (0.85, 1.85),
            "amount": (0.70, 1.45),
            "new_device": (0.10, 0.48),
            "new_geo": (0.02, 0.24),
            "ring": True,
        },
        "slow_low_ring": {
            "volume": (0.65, 1.20),
            "amount": (0.65, 1.25),
            "new_device": (0.08, 0.35),
            "new_geo": (0.02, 0.20),
            "ring": True,
        },
        "account_takeover_shift": {
            "volume": (0.75, 1.75),
            "amount": (1.20, 3.60),
            "new_device": (0.35, 0.92),
            "new_geo": (0.28, 0.88),
        },
        "mixed_evasion": {
            "volume": (0.95, 2.10),
            "amount": (0.85, 2.05),
            "new_device": (0.18, 0.68),
            "new_geo": (0.10, 0.52),
        },
    }
    spec = ranges[family]
    volume_range = spec["volume"]
    amount_range = spec["amount"]
    device_range = spec["new_device"]
    geo_range = spec["new_geo"]
    assert isinstance(volume_range, tuple)
    assert isinstance(amount_range, tuple)
    assert isinstance(device_range, tuple)
    assert isinstance(geo_range, tuple)
    effect = FamilyEffect(
        volume_multiplier=rng.uniform(*volume_range),
        amount_multiplier=rng.uniform(*amount_range),
        new_device_rate=rng.uniform(*device_range),
        new_geo_rate=rng.uniform(*geo_range),
        ring_mode=bool(spec.get("ring", False)),
    )

    auth: float | None = None
    refund: float | None = None
    chargeback: float | None = None
    if family == "auth_provider_incident":
        auth = rng.uniform(0.08, 0.34)
    elif family == "card_testing_style":
        auth = rng.uniform(0.28, 0.82)
    elif family == "refund_abuse_style":
        refund = rng.uniform(0.12, 0.58)
    elif family == "account_takeover_shift":
        auth = rng.uniform(0.07, 0.32)
    elif family == "mixed_evasion":
        auth = rng.uniform(0.06, 0.28)
        refund = rng.uniform(0.04, 0.22)
        chargeback = rng.uniform(0.01, 0.09)

    # Benign confounders occasionally inherit a risk-like secondary signal, while some
    # risky families become only partial attacks. This prevents one-feature shortcuts.
    if rng.random() < confounder_probability:
        if family in LEGITIMATE_FAMILIES:
            choice = rng.choice(("auth", "refund", "new_geo"))
            if choice == "auth":
                auth = max(auth or 0.0, rng.uniform(0.07, 0.24))
            elif choice == "refund":
                refund = max(refund or 0.0, rng.uniform(0.06, 0.18))
            else:
                effect = FamilyEffect(
                    effect.volume_multiplier,
                    effect.amount_multiplier,
                    effect.new_device_rate,
                    max(effect.new_geo_rate, rng.uniform(0.18, 0.46)),
                    auth,
                    refund,
                    chargeback,
                    effect.ring_mode,
                )
        else:
            shrink = rng.uniform(0.45, 0.80)
            effect = FamilyEffect(
                1.0 + (effect.volume_multiplier - 1.0) * shrink,
                1.0 + (effect.amount_multiplier - 1.0) * shrink,
                effect.new_device_rate * shrink,
                effect.new_geo_rate * shrink,
                auth * shrink if auth is not None else None,
                refund * shrink if refund is not None else None,
                chargeback * shrink if chargeback is not None else None,
                effect.ring_mode,
            )

    return FamilyEffect(
        effect.volume_multiplier,
        effect.amount_multiplier,
        effect.new_device_rate,
        effect.new_geo_rate,
        auth if auth is not None else effect.auth_fail_rate,
        refund if refund is not None else effect.refund_rate,
        chargeback if chargeback is not None else effect.chargeback_rate,
        effect.ring_mode,
    )


def _poisson(rng: random.Random, lam: float) -> int:
    # Knuth is efficient enough for the small per-day rates used here.
    lam = max(lam, 0.01)
    if lam > 30:
        return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))
    limit = math.exp(-lam)
    product = 1.0
    k = 0
    while product > limit:
        product *= rng.random()
        k += 1
    return max(0, k - 1)


def _make_event(
    rng: random.Random,
    shape: MerchantShape,
    *,
    transaction_id: str,
    timestamp: datetime,
    amount_multiplier: float,
    new_device_rate: float,
    new_geo_rate: float,
    auth_fail_rate: float,
    refund_rate: float,
    chargeback_rate: float,
    ring_id: str | None,
    event_index: int,
) -> TransactionEvent:
    new_device = rng.random() < new_device_rate
    if ring_id is not None:
        device = f"{ring_id}_device_{event_index % 12}"
        customer_id = f"{ring_id}_customer_{event_index % 30}"
    elif new_device:
        device = f"{shape.merchant_id}_novel_device_{transaction_id[-8:]}"
        customer_id = f"{shape.merchant_id}_customer_{event_index % 80}"
    else:
        device = f"{shape.merchant_id}_device_{event_index % shape.device_pool_size}"
        customer_id = f"{shape.merchant_id}_customer_{event_index % 80}"

    if rng.random() < new_geo_rate:
        geo = rng.choice(["GB-LND", "SG-01", "AE-DU", "US-CA", "IN-GJ", "IN-RJ"])
    else:
        geo = rng.choice(shape.home_geos)

    auth_failed = rng.random() < auth_fail_rate
    refund = (not auth_failed) and rng.random() < refund_rate
    chargeback = (not auth_failed) and rng.random() < chargeback_rate
    amount = max(
        10.0,
        rng.lognormvariate(
            math.log(max(shape.base_ticket * amount_multiplier, 10.0)), shape.amount_sigma
        ),
    )
    return TransactionEvent(
        transaction_id=transaction_id,
        merchant_id=shape.merchant_id,
        customer_id=customer_id,
        ring_id=ring_id,
        timestamp=timestamp,
        amount=round(amount, 2),
        device_fingerprint=device,
        customer_geo=geo,
        auth_status="FAILED" if auth_failed else "APPROVED",
        refund_timestamp=timestamp + timedelta(hours=rng.randint(4, 96)) if refund else None,
        chargeback_timestamp=timestamp + timedelta(days=rng.randint(2, 14)) if chargeback else None,
    )


def _simulate_history(
    rng: random.Random,
    shape: MerchantShape,
    *,
    start: datetime,
    end: datetime,
) -> list[TransactionEvent]:
    events: list[TransactionEvent] = []
    day = 0
    cursor = start
    event_index = 0
    while cursor < end:
        weekday = cursor.weekday()
        seasonal = 1.0 + shape.weekday_amplitude * math.sin((weekday / 7.0) * 2.0 * math.pi)
        growth = max(0.55, 1.0 + shape.growth_per_day * day)
        day_volume = _poisson(rng, shape.base_daily_volume * seasonal * growth)
        for local_index in range(day_volume):
            seconds = rng.randint(0, 86399)
            timestamp = min(cursor + timedelta(seconds=seconds), end - timedelta(microseconds=1))
            events.append(
                _make_event(
                    rng,
                    shape,
                    transaction_id=f"hist_{shape.merchant_id}_{day:04d}_{local_index:04d}",
                    timestamp=timestamp,
                    amount_multiplier=max(0.60, rng.gauss(1.0, 0.08)),
                    new_device_rate=0.02,
                    new_geo_rate=0.01,
                    auth_fail_rate=shape.auth_fail_rate,
                    refund_rate=shape.refund_rate,
                    chargeback_rate=shape.chargeback_rate,
                    ring_id=None,
                    event_index=event_index,
                )
            )
            event_index += 1
        cursor += timedelta(days=1)
        day += 1
    return events


def _simulate_current(
    rng: random.Random,
    shape: MerchantShape,
    effect: FamilyEffect,
    *,
    start: datetime,
    end: datetime,
    family_index: int,
    merchant_offset: int,
) -> list[TransactionEvent]:
    duration_days = (end - start).total_seconds() / 86400.0
    expected = shape.base_daily_volume * duration_days * effect.volume_multiplier
    count = max(3, _poisson(rng, expected))
    ring_id = None
    if effect.ring_mode:
        ring_id = f"shared_{family_index:02d}_{merchant_offset // 8:04d}"

    auth_rate = effect.auth_fail_rate if effect.auth_fail_rate is not None else shape.auth_fail_rate
    refund_rate = effect.refund_rate if effect.refund_rate is not None else shape.refund_rate
    cb_rate = (
        effect.chargeback_rate if effect.chargeback_rate is not None else shape.chargeback_rate
    )
    events: list[TransactionEvent] = []
    for index in range(count):
        seconds = rng.randint(0, max(1, int((end - start).total_seconds()) - 1))
        timestamp = start + timedelta(seconds=seconds)
        events.append(
            _make_event(
                rng,
                shape,
                transaction_id=f"cur_{shape.merchant_id}_{index:05d}",
                timestamp=timestamp,
                amount_multiplier=effect.amount_multiplier * max(0.55, rng.gauss(1.0, 0.12)),
                new_device_rate=effect.new_device_rate,
                new_geo_rate=effect.new_geo_rate,
                auth_fail_rate=auth_rate,
                refund_rate=refund_rate,
                chargeback_rate=cb_rate,
                ring_id=ring_id,
                event_index=index,
            )
        )
    return sorted(events, key=lambda event: event.timestamp)


def generate_record(
    *,
    rng: random.Random,
    config: LongitudinalConfig,
    family: str,
    family_index: int,
    merchant_offset: int,
) -> LongitudinalRecord:
    merchant_number = family_index * config.merchants_per_family + merchant_offset
    merchant_id = f"lv31_m_{merchant_number:06d}"
    shape = _merchant_shape(rng, merchant_id=merchant_id, family=family)

    jitter_days = merchant_offset % 19
    hold_time = config.reference_time + timedelta(days=jitter_days)
    current_end = hold_time
    current_start = current_end - timedelta(hours=config.current_hours)
    baseline_end = current_start
    baseline_start = baseline_end - timedelta(days=config.baseline_days)
    history_start = hold_time - timedelta(days=config.history_days)

    history = _simulate_history(rng, shape, start=history_start, end=current_start)
    baseline = estimate_baseline_from_history(
        history, baseline_start=baseline_start, baseline_end=baseline_end
    )
    effect = _family_effect(rng, family, config.confounder_probability)
    current = _simulate_current(
        rng,
        shape,
        effect,
        start=current_start,
        end=current_end,
        family_index=family_index,
        merchant_offset=merchant_offset,
    )

    hold_id = uuid5(NAMESPACE_URL, f"razortrust:v31:{config.seed}:{merchant_id}")
    hold = HoldCase(
        hold_id=hold_id,
        request_id=hold_id,
        merchant_id=merchant_id,
        source_event_id=f"synthetic_v31_{merchant_number:06d}",
        triggered_at=hold_time,
        reason_code=family.upper(),
    )
    feature_vector = build_point_in_time_features(
        hold,
        HoldEvaluationInput(
            baseline=baseline,
            transactions=current,
            cohort=shape.cohort,
            window_hours=config.current_hours,
        ),
    )

    legitimate = family in LEGITIMATE_FAMILIES
    if family in DIRECT_RELEASE_FAMILIES:
        target = HoldDecision.RELEASE
    elif family in EVIDENCE_RESOLVABLE_FAMILIES:
        target = HoldDecision.EVIDENCE_NEEDED
    else:
        target = HoldDecision.ESCALATE

    row: dict[str, object] = {
        **feature_vector.model_dump(),
        "merchant_id": merchant_id,
        "cohort": shape.cohort,
        "scenario_family": family,
        "true_risk_state": "LEGITIMATE" if legitimate else "RISKY",
        "evidence_resolvable": family in EVIDENCE_RESOLVABLE_FAMILIES,
        "attack_family": None if legitimate else family,
        "operational_target": target.value,
        "hold_triggered_at": hold_time.isoformat(),
        "dataset_version": LONGITUDINAL_DATASET_VERSION,
        "generator_version": LONGITUDINAL_GENERATOR_VERSION,
        "seed": config.seed,
    }
    return LongitudinalRecord(
        row=row,
        baseline_window_start=baseline_start,
        baseline_window_end=baseline_end,
        current_window_start=current_start,
        current_window_end=current_end,
        baseline_event_count=sum(baseline_start <= e.timestamp < baseline_end for e in history),
        current_event_count=len(current),
        processed_event_count=len(history) + len(current),
    )


def generate_longitudinal_frame(
    config: LongitudinalConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    config.validate()
    rng = random.Random(config.seed)
    rows: list[dict[str, object]] = []
    total_events = 0
    baseline_events = 0
    current_events = 0
    interval_violations = 0

    families = LEGITIMATE_FAMILIES + RISK_FAMILIES
    for family_index, family in enumerate(families):
        for merchant_offset in range(config.merchants_per_family):
            record = generate_record(
                rng=rng,
                config=config,
                family=family,
                family_index=family_index,
                merchant_offset=merchant_offset,
            )
            rows.append(record.row)
            total_events += record.processed_event_count
            baseline_events += record.baseline_event_count
            current_events += record.current_event_count
            if record.baseline_window_end > record.current_window_start:
                interval_violations += 1
            if record.current_window_end > config.reference_time + timedelta(days=18):
                # This is a sanity bound, not a model input.
                interval_violations += 1

    metadata_columns = (
        "merchant_id",
        "cohort",
        "scenario_family",
        "true_risk_state",
        "evidence_resolvable",
        "attack_family",
        "operational_target",
        "hold_triggered_at",
        "dataset_version",
        "generator_version",
        "seed",
    )
    frame = pd.DataFrame(rows, columns=(*FEATURE_COLUMNS, *metadata_columns))
    if tuple(frame.columns[: len(FEATURE_COLUMNS)]) != FEATURE_COLUMNS:
        raise RuntimeError("longitudinal frame violates locked 13-feature order")
    numeric = frame.loc[:, list(FEATURE_COLUMNS)]
    if not all(math.isfinite(float(value)) for value in numeric.to_numpy().ravel()):
        raise RuntimeError("non-finite feature detected")

    family_counts = Counter(frame["scenario_family"].astype(str))
    report: dict[str, object] = {
        "dataset_version": LONGITUDINAL_DATASET_VERSION,
        "generator_version": LONGITUDINAL_GENERATOR_VERSION,
        "seed": config.seed,
        "merchant_count": int(frame["merchant_id"].nunique()),
        "hold_window_count": int(len(frame)),
        "processed_transaction_count": int(total_events),
        "baseline_transaction_count": int(baseline_events),
        "current_transaction_count": int(current_events),
        "family_count": len(families),
        "family_counts": dict(sorted(family_counts.items())),
        "cohort_counts": dict(sorted(Counter(frame["cohort"].astype(str)).items())),
        "interval_overlap_violations": interval_violations,
        "feature_nan_count": int(numeric.isna().sum().sum()),
        "feature_zero_variance_count": int((numeric.std(axis=0) == 0).sum()),
    }
    return frame, report


def _canonical_frame_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.to_dict(orient="records"):
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":"), default=str).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def write_longitudinal_dataset(
    output_dir: str | Path, config: LongitudinalConfig
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame, report = generate_longitudinal_frame(config)

    feature_path = output / "hold_windows.csv.gz"
    frame.to_csv(feature_path, index=False, compression="gzip")

    family_summary = (
        frame.groupby(["scenario_family", "true_risk_state"], as_index=False)[list(FEATURE_COLUMNS)]
        .mean(numeric_only=True)
        .sort_values("scenario_family")
    )
    family_summary.to_csv(output / "family_feature_means.csv", index=False)

    feature_summary = frame.loc[:, list(FEATURE_COLUMNS)].describe().T
    feature_summary.to_csv(output / "feature_summary.csv")

    report.update(
        {
            "feature_schema_version": "1.0.0",
            "feature_columns": list(FEATURE_COLUMNS),
            "frame_sha256": _canonical_frame_sha256(frame),
            "output_files": {
                "hold_windows": feature_path.name,
                "family_feature_means": "family_feature_means.csv",
                "feature_summary": "feature_summary.csv",
            },
            "production_action_eligible": False,
            "promotion_eligible": False,
            "scope": "SYNTHETIC_LONGITUDINAL_RESEARCH_ONLY",
        }
    )
    manifest_bytes = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    (output / "dataset_manifest.json").write_bytes(manifest_bytes)
    return report


def future_mutation_invariance_check(config: LongitudinalConfig) -> bool:
    """A deliberately extreme event after the hold must not affect the feature vector."""
    config.validate()
    rng = random.Random(config.seed + 991)
    family = "baseline"
    shape = _merchant_shape(rng, merchant_id="future_mutation_probe", family=family)
    hold_time = config.reference_time
    current_start = hold_time - timedelta(hours=config.current_hours)
    baseline_end = current_start
    baseline_start = baseline_end - timedelta(days=config.baseline_days)
    history_start = hold_time - timedelta(days=config.history_days)
    history = _simulate_history(rng, shape, start=history_start, end=current_start)
    baseline = estimate_baseline_from_history(
        history, baseline_start=baseline_start, baseline_end=baseline_end
    )
    effect = _family_effect(rng, family, config.confounder_probability)
    current = _simulate_current(
        rng,
        shape,
        effect,
        start=current_start,
        end=hold_time,
        family_index=0,
        merchant_offset=0,
    )
    hold_id = uuid5(NAMESPACE_URL, "razortrust:v31:future-mutation-probe")
    hold = HoldCase(
        hold_id=hold_id,
        request_id=hold_id,
        merchant_id=shape.merchant_id,
        source_event_id="future_probe",
        triggered_at=hold_time,
        reason_code="BASELINE",
    )
    before = build_point_in_time_features(
        hold,
        HoldEvaluationInput(baseline=baseline, transactions=current, cohort=shape.cohort),
    )
    future = TransactionEvent(
        transaction_id="future_extreme",
        merchant_id=shape.merchant_id,
        customer_id="future_customer",
        timestamp=hold_time + timedelta(seconds=1),
        amount=999999999.0,
        device_fingerprint="future_device",
        customer_geo="ZZ-XX",
        auth_status="FAILED",
        refund_timestamp=hold_time + timedelta(seconds=2),
        chargeback_timestamp=hold_time + timedelta(seconds=3),
    )
    after = build_point_in_time_features(
        hold,
        HoldEvaluationInput(
            baseline=baseline, transactions=current + [future], cohort=shape.cohort
        ),
    )
    return before == after
