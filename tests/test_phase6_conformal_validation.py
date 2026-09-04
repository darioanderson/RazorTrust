from __future__ import annotations

import numpy as np
import pandas as pd

from razortrust.domain import HoldDecision
from razortrust.ml.conformal_validation import (
    shift_model_matrix,
    wilson_interval,
)
from razortrust.ml.uncertainty import ApsConformalAbstainer


def test_wilson_interval_contains_observed_rate() -> None:
    lower, upper = wilson_interval(90, 100)
    assert lower < 0.90 < upper


def test_shift_model_matrix_changes_only_selected_features() -> None:
    reference = pd.DataFrame(
        {
            "a": np.linspace(-2.0, 2.0, 100),
            "b": np.linspace(1.0, 4.0, 100),
        }
    )
    current = reference.iloc[:20].copy()
    shifted = shift_model_matrix(
        current,
        reference=reference,
        features=["a"],
        strength_reference_std=0.5,
    )
    assert not np.allclose(shifted["a"], current["a"])
    assert np.allclose(shifted["b"], current["b"])


def test_conservative_policy_never_releases_ambiguous_sets() -> None:
    abstainer = ApsConformalAbstainer(confidence_level=0.90)
    ambiguous_sets = (
        np.asarray([True, True, False]),
        np.asarray([True, False, True]),
        np.asarray([False, True, True]),
        np.asarray([True, True, True]),
        np.asarray([False, False, False]),
    )
    for prediction_set in ambiguous_sets:
        decision = abstainer.policy_decision(prediction_set, False)
        assert decision != HoldDecision.RELEASE
