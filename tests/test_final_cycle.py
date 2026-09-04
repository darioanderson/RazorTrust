from __future__ import annotations

import numpy as np
import pandas as pd

from razortrust.ml.final_cycle import (
    FINAL_FEATURES,
    add_final_features,
    select_release_threshold,
    temporal_splits,
)
from razortrust.ml.v3_research import V3A_FEATURES


def test_threshold_never_weakens_false_release_gate() -> None:
    scores = np.array([0.95, 0.90, 0.80, 0.70, 0.10])
    labels = np.array([1, 0, 1, 0, 0])
    selected = select_release_threshold(scores, labels, max_false_release_rate=0.05)
    assert selected.false_release_rate <= 0.05
    assert selected.true_release_recall == 0.5


def test_final_features_are_finite_and_do_not_use_metadata() -> None:
    row = {feature: 0.5 for feature in V3A_FEATURES}
    row.update(
        merchant_id="merchant-1",
        scenario_family="baseline",
        operational_target="RELEASE",
        hold_triggered_at="2026-01-01T00:00:00Z",
    )
    result = add_final_features(pd.DataFrame([row]))
    assert np.isfinite(result.loc[:, FINAL_FEATURES].to_numpy()).all()


def test_temporal_folds_are_merchant_isolated() -> None:
    rows = []
    for family in ("a", "b"):
        for index in range(10):
            rows.append(
                {
                    "merchant_id": f"{family}-{index}",
                    "scenario_family": family,
                    "hold_triggered_at": pd.Timestamp("2026-01-01", tz="UTC")
                    + pd.Timedelta(index, unit="D"),
                }
            )
    frame = pd.DataFrame(rows)
    for fit, validation in temporal_splits(frame):
        assert set(frame.iloc[fit].merchant_id).isdisjoint(frame.iloc[validation].merchant_id)
        for family in ("a", "b"):
            fit_frame = frame.iloc[fit]
            validation_frame = frame.iloc[validation]
            fit_times = fit_frame.loc[fit_frame.scenario_family == family]
            validation_times = validation_frame.loc[validation_frame.scenario_family == family]
            assert fit_times.hold_triggered_at.max() < validation_times.hold_triggered_at.min()
