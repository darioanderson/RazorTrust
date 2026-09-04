from __future__ import annotations

import numpy as np

from razortrust.ml.phase3b_v21_diagnostics import (
    embedding_drift_report,
    fit_sigmoid_score_calibrator,
    select_threshold_strategies,
    threshold_sweep,
)


def test_sigmoid_calibration_is_finite_and_monotonic() -> None:
    raw = np.asarray([0.02, 0.08, 0.20, 0.55, 0.80, 0.95], dtype=float)
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=int)
    calibrator = fit_sigmoid_score_calibrator(raw, labels)
    calibrated = calibrator.transform(raw)
    assert np.isfinite(calibrated).all()
    assert np.all(np.diff(calibrated) >= 0.0)
    assert np.all((calibrated > 0.0) & (calibrated < 1.0))


def test_threshold_strategies_respect_constraints_when_available() -> None:
    probabilities = np.asarray([0.05, 0.15, 0.25, 0.40, 0.60, 0.80, 0.90, 0.95])
    labels = np.asarray([0, 0, 0, 1, 0, 1, 1, 1], dtype=int)
    thresholds = select_threshold_strategies(
        probabilities,
        labels,
        recall_floor=0.75,
        sample_fpr_cap=0.34,
    )
    assert thresholds["max_f1_high_tie"] is not None
    assert thresholds["max_f1_low_tie"] is not None
    assert thresholds["recall_floor"] is not None
    assert thresholds["sample_fpr_cap"] is not None


def test_embedding_drift_report_is_finite() -> None:
    rng = np.random.default_rng(7)
    base = rng.normal(size=(20, 4))
    report = embedding_drift_report(
        {
            "classifier_train": base,
            "calibration": base + 0.05,
            "policy": base + 0.10,
            "test": base + 0.15,
        }
    )
    transitions = report["transitions"]
    assert set(transitions) == {
        "classifier_train_to_calibration",
        "calibration_to_policy",
        "policy_to_test",
    }
    for metrics in transitions.values():
        assert all(np.isfinite(float(value)) for value in metrics.values())


def test_posthoc_threshold_sweep_has_fixed_grid() -> None:
    probabilities = np.asarray([0.1, 0.3, 0.7, 0.9], dtype=float)
    labels = np.asarray([0, 0, 1, 1], dtype=int)
    rows = threshold_sweep(labels, probabilities)
    assert len(rows) == 19
    assert rows[0]["threshold"] == 0.05
    assert rows[-1]["threshold"] == 0.95
