from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from ..domain import TransactionEvent


def point_in_time_sequence_features(
    merchant_id: str,
    transactions: list[TransactionEvent],
    cutoff: datetime,
    *,
    window_hours: int = 24,
) -> dict[str, float]:
    """Engineered temporal baseline required before admitting an LSTM/Transformer."""
    events = sorted(
        (
            event
            for event in transactions
            if event.merchant_id == merchant_id
            and cutoff - timedelta(hours=window_hours) <= event.timestamp < cutoff
        ),
        key=lambda event: event.timestamp,
    )
    if not events:
        return {
            "burst_score_10m": 0.0,
            "amount_autocorrelation": 0.0,
            "auth_failure_run_max": 0.0,
            "new_device_transition_rate": 0.0,
        }
    burst_max = 1
    left = 0
    for right, event in enumerate(events):
        while event.timestamp - events[left].timestamp > timedelta(minutes=10):
            left += 1
        burst_max = max(burst_max, right - left + 1)
    amounts = np.asarray([event.amount for event in events], dtype=float)
    autocorrelation = (
        float(np.corrcoef(amounts[:-1], amounts[1:])[0, 1])
        if len(amounts) >= 3 and amounts[:-1].std() > 0 and amounts[1:].std() > 0
        else 0.0
    )
    failure_run = 0
    failure_run_max = 0
    for event in events:
        failure_run = failure_run + 1 if event.auth_status == "FAILED" else 0
        failure_run_max = max(failure_run_max, failure_run)
    device_transitions = sum(
        left.device_fingerprint != right.device_fingerprint
        for left, right in zip(events, events[1:], strict=False)
    )
    return {
        "burst_score_10m": float(burst_max),
        "amount_autocorrelation": autocorrelation,
        "auth_failure_run_max": float(failure_run_max),
        "new_device_transition_rate": float(device_transitions / max(1, len(events) - 1)),
    }
