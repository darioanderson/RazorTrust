from __future__ import annotations

import hashlib

import numpy as np
from pydantic import Field

from ..audit import canonical_json
from ..domain import HoldDecision, StrictModel


class OutcomeCostRecord(StrictModel):
    decision: HoldDecision
    true_state: HoldDecision
    fraud_loss: float = Field(ge=0)
    settlement_delay_loss: float = Field(ge=0)
    review_operations_cost: float = Field(ge=0)

    @property
    def total_cost(self) -> float:
        return self.fraud_loss + self.settlement_delay_loss + self.review_operations_cost


class CalibratedCostMatrix(StrictModel):
    schema_version: str = "1.0"
    cost_matrix_version: str
    actions: list[HoldDecision]
    true_states: list[HoldDecision]
    matrix: list[list[float]]
    cell_counts: list[list[int]]
    estimator: str = "winsorized-mean-p99"
    content_sha256: str


def calibrate_cost_matrix(
    records: list[OutcomeCostRecord],
    *,
    version: str,
    minimum_cell_count: int = 30,
) -> CalibratedCostMatrix:
    if minimum_cell_count < 1:
        raise ValueError("minimum_cell_count must be positive")
    order = list(HoldDecision)
    matrix: list[list[float]] = []
    counts: list[list[int]] = []
    for action in order:
        action_costs: list[float] = []
        action_counts: list[int] = []
        for true_state in order:
            values = np.asarray(
                [
                    record.total_cost
                    for record in records
                    if record.decision == action and record.true_state == true_state
                ],
                dtype=float,
            )
            if len(values) < minimum_cell_count:
                raise ValueError(
                    f"cost cell {action}/{true_state} has {len(values)} rows; "
                    f"requires {minimum_cell_count}"
                )
            upper = float(np.quantile(values, 0.99))
            action_costs.append(round(float(np.mean(np.minimum(values, upper))), 6))
            action_counts.append(len(values))
        matrix.append(action_costs)
        counts.append(action_counts)
    payload = {
        "schema_version": "1.0",
        "cost_matrix_version": version,
        "actions": order,
        "true_states": order,
        "matrix": matrix,
        "cell_counts": counts,
        "estimator": "winsorized-mean-p99",
    }
    return CalibratedCostMatrix(
        schema_version="1.0",
        cost_matrix_version=version,
        actions=order,
        true_states=order,
        matrix=matrix,
        cell_counts=counts,
        estimator="winsorized-mean-p99",
        content_sha256=hashlib.sha256(canonical_json(payload)).hexdigest(),
    )
