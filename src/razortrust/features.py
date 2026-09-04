from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timedelta
from itertools import pairwise
from statistics import fmean, pstdev

from .domain import FeatureVector, HoldCase, HoldEvaluationInput, TransactionEvent

FEATURE_SCHEMA_VERSION = "1.0.0"
FEATURE_COLUMNS = (
    "volume_delta_z",
    "gmv_delta_z",
    "ticket_size_delta_z",
    "new_device_ratio",
    "new_geo_ratio",
    "refund_rate_delta_z",
    "chargeback_rate_delta_z",
    "failed_auth_ratio",
    "volume_trend_slope",
    "interarrival_time_cv",
    "device_entropy",
    "geo_entropy",
    "amount_distribution_kl",
)


def _z_score(value: float, mean: float, std: float) -> float:
    return (value - mean) / std


def _entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def _coefficient_of_variation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = fmean(values)
    return 0.0 if mean == 0 else pstdev(values) / mean


def _hourly_volume_slope(
    events: list[TransactionEvent], window_hours: int, end_time: datetime
) -> float:
    counts = [0] * window_hours
    start_time = end_time - timedelta(hours=window_hours)
    for event in events:
        bucket = min(window_hours - 1, int((event.timestamp - start_time).total_seconds() // 3600))
        if bucket >= 0:
            counts[bucket] += 1
    x_mean = (window_hours - 1) / 2
    denominator = sum((index - x_mean) ** 2 for index in range(window_hours))
    if denominator == 0:
        return 0.0
    y_mean = fmean(counts)
    return (
        sum((index - x_mean) * (count - y_mean) for index, count in enumerate(counts)) / denominator
    )


def _histogram_probabilities(amounts: list[float], edges: list[float]) -> list[float]:
    counts = [0] * (len(edges) - 1)
    for amount in amounts:
        if amount < edges[0]:
            counts[0] += 1
            continue
        if amount > edges[-1]:
            counts[-1] += 1
            continue
        for index, (left, right) in enumerate(pairwise(edges)):
            if left <= amount < right or (index == len(counts) - 1 and amount == right):
                counts[index] += 1
                break
    smoothing = 1e-6
    total = sum(counts) + smoothing * len(counts)
    return [(count + smoothing) / total for count in counts]


def _kl_divergence(current: list[float], reference: list[float]) -> float:
    smoothing = 1e-12
    return sum(
        value * math.log(value / max(reference[index], smoothing))
        for index, value in enumerate(current)
    )


def build_point_in_time_features(hold: HoldCase, evaluation: HoldEvaluationInput) -> FeatureVector:
    """Build the locked feature vector using only facts known at the hold timestamp."""
    start = hold.triggered_at - timedelta(hours=evaluation.window_hours)
    events = sorted(
        (
            event
            for event in evaluation.transactions
            if event.merchant_id == hold.merchant_id
            and start <= event.timestamp <= hold.triggered_at
        ),
        key=lambda event: event.timestamp,
    )
    if not events:
        raise ValueError("no transactions are available in the hold window")

    baseline = evaluation.baseline
    amounts = [event.amount for event in events]
    volume = len(events)
    gmv = sum(amounts)
    ticket_size = fmean(amounts)
    refunds_known = sum(
        event.refund_timestamp is not None and event.refund_timestamp <= hold.triggered_at
        for event in events
    )
    chargebacks_known = sum(
        event.chargeback_timestamp is not None and event.chargeback_timestamp <= hold.triggered_at
        for event in events
    )
    failed_auths = sum(event.auth_status == "FAILED" for event in events)
    interarrival_seconds = [
        (right.timestamp - left.timestamp).total_seconds() for left, right in pairwise(events)
    ]
    current_amount_distribution = _histogram_probabilities(amounts, baseline.amount_bin_edges)

    features = FeatureVector(
        volume_delta_z=_z_score(volume, baseline.volume_mean, baseline.volume_std),
        gmv_delta_z=_z_score(gmv, baseline.gmv_mean, baseline.gmv_std),
        ticket_size_delta_z=_z_score(
            ticket_size, baseline.ticket_size_mean, baseline.ticket_size_std
        ),
        new_device_ratio=sum(
            event.device_fingerprint not in baseline.known_devices for event in events
        )
        / volume,
        new_geo_ratio=sum(event.customer_geo not in baseline.known_geos for event in events)
        / volume,
        refund_rate_delta_z=_z_score(
            refunds_known / volume, baseline.refund_rate_mean, baseline.refund_rate_std
        ),
        chargeback_rate_delta_z=_z_score(
            chargebacks_known / volume,
            baseline.chargeback_rate_mean,
            baseline.chargeback_rate_std,
        ),
        failed_auth_ratio=failed_auths / volume,
        volume_trend_slope=_hourly_volume_slope(events, evaluation.window_hours, hold.triggered_at),
        interarrival_time_cv=_coefficient_of_variation(interarrival_seconds),
        device_entropy=_entropy([event.device_fingerprint for event in events]),
        geo_entropy=_entropy([event.customer_geo for event in events]),
        amount_distribution_kl=_kl_divergence(
            current_amount_distribution,
            baseline.amount_bin_probabilities,
        ),
    )
    if tuple(features.model_dump()) != FEATURE_COLUMNS:
        raise RuntimeError("feature schema order does not match the locked contract")
    return features
