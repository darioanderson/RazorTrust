from __future__ import annotations

import numpy as np

from razortrust.ml.phase2c_family_generalization import (
    consensus_signal,
    family_robust_upper_tail,
)


def test_family_robust_threshold_uses_most_conservative_family() -> None:
    a = np.linspace(0.0, 1.0, 30)
    b = np.linspace(3.0, 5.0, 30)
    scores = np.concatenate([a, b])
    families = np.asarray(["a"] * len(a) + ["b"] * len(b))
    result = family_robust_upper_tail(scores, families, target_false_alarm_rate=0.10)
    assert result.threshold == max(result.family_thresholds.values())
    assert result.family_thresholds["b"] > result.family_thresholds["a"]


def test_consensus_rules_are_distinct() -> None:
    recon = np.asarray([True, True, False, False])
    latent = np.asarray([False, True, True, False])
    isolation = np.asarray([False, False, True, True])
    assert consensus_signal("AE_ANY", recon, latent, isolation).tolist() == [
        True,
        True,
        True,
        False,
    ]
    assert consensus_signal("AE_BOTH", recon, latent, isolation).tolist() == [
        False,
        True,
        False,
        False,
    ]
    assert consensus_signal("TWO_OF_THREE", recon, latent, isolation).tolist() == [
        False,
        True,
        True,
        False,
    ]
