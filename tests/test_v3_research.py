from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from razortrust.ml.v3_research import (
    LABEL_TO_INDEX,
    UNKNOWN_FAMILIES,
    V3A_FEATURES,
    V3B_CANDIDATES,
    add_v3b_candidates,
    create_research_partitions,
    fail_closed_action,
    select_thresholds,
)


def _frame() -> pd.DataFrame:
    rows = []
    families = [*UNKNOWN_FAMILIES, "baseline", "flash_sale", "account_takeover_shift"]
    targets = {
        "geo_expansion": "EVIDENCE_NEEDED",
        "seasonal_peak": "RELEASE",
        "slow_low_ring": "ESCALATE",
        "baseline": "RELEASE",
        "flash_sale": "EVIDENCE_NEEDED",
        "account_takeover_shift": "ESCALATE",
    }
    for family in families:
        for index in range(20):
            row = {feature: float(index + 1) / 20 for feature in V3A_FEATURES}
            row.update(
                merchant_id=f"{family}-{index}",
                scenario_family=family,
                operational_target=targets[family],
                hold_triggered_at=f"2026-01-{index + 1:02d}T00:00:00Z",
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_research_partitions_are_chronological_disjoint_and_hold_out_families() -> None:
    frame = _frame()
    partition, manifest = create_research_partitions(frame)
    assert set(partition[frame.scenario_family.isin(UNKNOWN_FAMILIES)]) == {"unknown_family"}
    known = frame[~frame.scenario_family.isin(UNKNOWN_FAMILIES)]
    assert (partition.loc[known.index] == "development").sum() == 42
    assert (partition.loc[known.index] == "v3b_gate").sum() == 9
    assert (partition.loc[known.index] == "sealed").sum() == 9
    assert len(manifest["content_sha256"]) == 64


def test_v3b_candidate_features_are_finite_and_do_not_mutate_v3a() -> None:
    frame = _frame()
    original = frame.loc[:, V3A_FEATURES].copy()
    enriched = add_v3b_candidates(frame)
    assert tuple(enriched.loc[:, V3B_CANDIDATES].columns) == V3B_CANDIDATES
    assert np.isfinite(enriched.loc[:, V3B_CANDIDATES].to_numpy()).all()
    pd.testing.assert_frame_equal(frame.loc[:, V3A_FEATURES], original)


def test_locked_threshold_target_enforces_both_constraints() -> None:
    labels = np.asarray(
        [LABEL_TO_INDEX["RELEASE"]] * 5
        + [LABEL_TO_INDEX["EVIDENCE_NEEDED"]] * 5
        + [LABEL_TO_INDEX["ESCALATE"]] * 5
    )
    probabilities = np.asarray(
        [[0.90, 0.08, 0.02]] * 5 + [[0.02, 0.90, 0.08]] * 5 + [[0.01, 0.09, 0.90]] * 5
    )
    selected = select_thresholds(probabilities, labels)
    assert selected.false_release_rate <= 0.05
    assert selected.true_release_recall >= 0.20


def test_locked_threshold_target_rejects_infeasible_probabilities() -> None:
    labels = np.asarray([0, 1, 2])
    probabilities = np.asarray([[0.1, 0.8, 0.1], [0.9, 0.1, 0.0], [0.9, 0.0, 0.1]])
    with pytest.raises(ValueError, match="locked false-release and recall"):
        select_thresholds(probabilities, labels)


def test_inference_failure_fails_closed_to_evidence() -> None:
    frame = _frame().iloc[[0]]
    labels = np.asarray([0, 1, 2])
    probabilities = np.asarray([[0.9, 0.1, 0.0], [0.0, 0.9, 0.1], [0.0, 0.1, 0.9]])
    thresholds = select_thresholds(probabilities, labels)
    action, error = fail_closed_action(None, None, frame, V3A_FEATURES, thresholds)
    assert action == "EVIDENCE_NEEDED"
    assert error == "RuntimeError"
