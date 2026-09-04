from __future__ import annotations

import numpy as np
from scipy.stats import binomtest

from .uncertainty import ApsConformalAbstainer


def exact_undercoverage_test(
    abstainer: ApsConformalAbstainer,
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    target_coverage: float,
    significance_level: float = 0.05,
) -> dict[str, object]:
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    sets = abstainer.prediction_sets(probabilities)

    covered = sets[np.arange(len(labels)), labels]
    successes = int(np.sum(covered))
    trials = int(len(labels))
    empirical = successes / trials

    result = binomtest(
        successes,
        n=trials,
        p=float(target_coverage),
        alternative="less",
    )
    evidence_of_undercoverage = float(result.pvalue) < significance_level

    return {
        "successes": successes,
        "trials": trials,
        "empirical_coverage": round(float(empirical), 8),
        "target_coverage": float(target_coverage),
        "alternative": "less",
        "p_value": round(float(result.pvalue), 12),
        "significance_level": float(significance_level),
        "statistically_significant_undercoverage": evidence_of_undercoverage,
    }


def corrected_v11_gate(
    *,
    empirical_coverage: float,
    target_coverage: float,
    undercoverage_test: dict[str, object],
    routing: dict[str, object],
) -> dict[str, object]:
    empirical_gate = empirical_coverage >= target_coverage
    no_significant_undercoverage = not bool(
        undercoverage_test["statistically_significant_undercoverage"]
    )
    routing_invariants = (
        int(routing["ambiguous_release_invariant_violations"]) == 0
        and int(routing["empty_release_invariant_violations"]) == 0
        and int(routing["full_release_invariant_violations"]) == 0
    )

    passed = (
        empirical_gate
        and no_significant_undercoverage
        and routing_invariants
    )
    return {
        "empirical_coverage_at_least_nominal": empirical_gate,
        "no_significant_evidence_of_undercoverage": no_significant_undercoverage,
        "conservative_routing_invariants_hold": routing_invariants,
        "pass": passed,
    }
