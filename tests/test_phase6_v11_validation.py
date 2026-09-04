from __future__ import annotations

import numpy as np

from razortrust.ml.conformal_v11_validation import (
    corrected_v11_gate,
    exact_undercoverage_test,
)
from razortrust.ml.uncertainty import ApsConformalAbstainer


def test_exact_undercoverage_test_accepts_conservative_coverage() -> None:
    probabilities = np.asarray(
        [[0.98, 0.01, 0.01], [0.01, 0.98, 0.01], [0.01, 0.01, 0.98]]
        * 40
    )
    labels = np.asarray([0, 1, 2] * 40)
    abstainer = ApsConformalAbstainer(
        confidence_level=0.90,
        seed=42,
    ).conformalize(
        probabilities[:60],
        labels[:60],
    )
    result = exact_undercoverage_test(
        abstainer,
        probabilities[60:],
        labels[60:],
        target_coverage=0.90,
        significance_level=0.05,
    )
    assert result["empirical_coverage"] >= 0.90
    assert result["statistically_significant_undercoverage"] is False


def test_corrected_gate_does_not_penalize_overcoverage() -> None:
    undercoverage = {
        "statistically_significant_undercoverage": False,
    }
    routing = {
        "ambiguous_release_invariant_violations": 0,
        "empty_release_invariant_violations": 0,
        "full_release_invariant_violations": 0,
    }
    result = corrected_v11_gate(
        empirical_coverage=0.98,
        target_coverage=0.90,
        undercoverage_test=undercoverage,
        routing=routing,
    )
    assert result["pass"] is True


def test_corrected_gate_blocks_empirical_undercoverage() -> None:
    undercoverage = {
        "statistically_significant_undercoverage": False,
    }
    routing = {
        "ambiguous_release_invariant_violations": 0,
        "empty_release_invariant_violations": 0,
        "full_release_invariant_violations": 0,
    }
    result = corrected_v11_gate(
        empirical_coverage=0.89,
        target_coverage=0.90,
        undercoverage_test=undercoverage,
        routing=routing,
    )
    assert result["pass"] is False
