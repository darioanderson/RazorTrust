from __future__ import annotations

import math

from razortrust.features import FEATURE_COLUMNS
from razortrust.ml.longitudinal_data import (
    LongitudinalConfig,
    future_mutation_invariance_check,
    generate_longitudinal_frame,
)


def test_longitudinal_generator_builds_locked_features_from_history() -> None:
    frame, report = generate_longitudinal_frame(
        LongitudinalConfig(seed=3101, merchants_per_family=2, history_days=40, baseline_days=30)
    )
    assert len(frame) == 14 * 2
    assert tuple(frame.columns[:13]) == FEATURE_COLUMNS
    assert report["family_count"] == 14
    assert report["interval_overlap_violations"] == 0
    assert report["feature_nan_count"] == 0
    assert report["feature_zero_variance_count"] == 0
    assert all(
        math.isfinite(float(x)) for x in frame.loc[:, list(FEATURE_COLUMNS)].to_numpy().ravel()
    )


def test_longitudinal_generator_is_deterministic() -> None:
    config = LongitudinalConfig(
        seed=3102, merchants_per_family=2, history_days=40, baseline_days=30
    )
    first, report_first = generate_longitudinal_frame(config)
    second, report_second = generate_longitudinal_frame(config)
    assert first.to_dict(orient="records") == second.to_dict(orient="records")
    assert report_first == report_second


def test_future_events_never_change_original_features() -> None:
    config = LongitudinalConfig(
        seed=3103, merchants_per_family=2, history_days=40, baseline_days=30
    )
    assert future_mutation_invariance_check(config)


def test_legitimate_and_risky_families_have_overlap_not_trivial_separation() -> None:
    frame, _ = generate_longitudinal_frame(
        LongitudinalConfig(seed=3104, merchants_per_family=5, history_days=40, baseline_days=30)
    )
    legitimate = frame[frame["true_risk_state"] == "LEGITIMATE"]
    risky = frame[frame["true_risk_state"] == "RISKY"]
    # At least one major feature must have overlapping central ranges. This protects
    # against reverting to a generator where one feature trivially defines risk.
    overlap_count = 0
    for feature in ("volume_delta_z", "gmv_delta_z", "new_device_ratio", "new_geo_ratio"):
        l_lo, l_hi = legitimate[feature].quantile([0.10, 0.90])
        r_lo, r_hi = risky[feature].quantile([0.10, 0.90])
        if max(l_lo, r_lo) <= min(l_hi, r_hi):
            overlap_count += 1
    assert overlap_count >= 2
