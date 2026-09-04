from __future__ import annotations

import pytest

from razortrust.domain import HoldDecision
from razortrust.ml.cost_calibration import OutcomeCostRecord, calibrate_cost_matrix


def _records(per_cell: int):
    return [
        OutcomeCostRecord(
            decision=action,
            true_state=true_state,
            fraud_loss=float(index if true_state == HoldDecision.ESCALATE else 0),
            settlement_delay_loss=float(index if action != HoldDecision.RELEASE else 0),
            review_operations_cost=float(action == HoldDecision.ESCALATE),
        )
        for action in HoldDecision
        for true_state in HoldDecision
        for index in range(1, per_cell + 1)
    ]


def test_cost_calibration_requires_coverage_and_produces_hashed_matrix() -> None:
    report = calibrate_cost_matrix(_records(3), version="observed-cost@1", minimum_cell_count=3)

    assert report.cell_counts == [[3, 3, 3], [3, 3, 3], [3, 3, 3]]
    assert len(report.content_sha256) == 64
    assert report.matrix[0][2] > report.matrix[0][0]


def test_cost_calibration_rejects_sparse_cells() -> None:
    with pytest.raises(ValueError, match="requires 4"):
        calibrate_cost_matrix(_records(3), version="observed-cost@1", minimum_cell_count=4)
