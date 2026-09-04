from __future__ import annotations

import numpy as np
import pandas as pd

from razortrust.ml.drift_validation import (
    chronological_drift_partitions,
    inject_standard_deviation_shift,
    severity_from_signals,
)


def test_chronological_drift_partitions_are_disjoint_and_ordered() -> None:
    parts = chronological_drift_partitions(1_000)
    assert len(parts.train) == 550
    assert len(parts.calibration) == 150
    assert len(parts.policy) == 150
    assert len(parts.monitor) == 150

    assert parts.train[-1] < parts.calibration[0]
    assert parts.calibration[-1] < parts.policy[0]
    assert parts.policy[-1] < parts.monitor[0]

    all_primary = np.concatenate(
        [parts.train, parts.calibration, parts.policy, parts.monitor]
    )
    assert len(np.unique(all_primary)) == len(all_primary)


def test_injection_changes_only_selected_features() -> None:
    reference = pd.DataFrame(
        {
            "V1": np.linspace(-2.0, 2.0, 100),
            "V2": np.linspace(1.0, 4.0, 100),
            "V3": np.linspace(10.0, 20.0, 100),
        }
    )
    current = reference.iloc[:20].copy()
    shifted = inject_standard_deviation_shift(
        current,
        reference=reference,
        features=["V1", "V3"],
        strength=0.5,
    )
    assert not np.allclose(shifted["V1"], current["V1"])
    assert np.allclose(shifted["V2"], current["V2"])
    assert not np.allclose(shifted["V3"], current["V3"])


def test_severity_requires_both_signal_types_for_red() -> None:
    stable_delta = {
        "pr_auc": 0.0,
        "recall": 0.0,
        "sample_fpr": 0.0,
        "brier": 0.0,
        "log_loss": 0.0,
    }
    severity, checks = severity_from_signals(
        drifted_feature_share=0.0,
        score_drifted=False,
        performance_delta_values=stable_delta,
    )
    assert severity == "GREEN"
    assert checks["performance_degraded"] is False

    degraded = dict(stable_delta)
    degraded["recall"] = -0.2
    severity, _ = severity_from_signals(
        drifted_feature_share=0.0,
        score_drifted=False,
        performance_delta_values=degraded,
    )
    assert severity == "AMBER"

    severity, _ = severity_from_signals(
        drifted_feature_share=0.25,
        score_drifted=False,
        performance_delta_values=degraded,
    )
    assert severity == "RED"
