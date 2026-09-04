from __future__ import annotations

import hashlib
import json
from importlib.resources import files

from pydantic import model_validator

from .audit import canonical_json
from .domain import HoldDecision, StrictModel


class CostMatrixArtifact(StrictModel):
    schema_version: str
    cost_matrix_version: str
    currency: str
    actions: list[HoldDecision]
    true_states: list[HoldDecision]
    matrix: list[list[float]]

    @model_validator(mode="after")
    def validate_shape_and_order(self) -> CostMatrixArtifact:
        expected = list(HoldDecision)
        if self.actions != expected or self.true_states != expected:
            raise ValueError(
                "cost matrix actions and true states must use canonical decision order"
            )
        if len(self.matrix) != 3 or any(len(row) != 3 for row in self.matrix):
            raise ValueError("cost matrix must be 3 by 3")
        if any(value < 0 for row in self.matrix for value in row):
            raise ValueError("cost matrix values cannot be negative")
        return self

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.model_dump(mode="json"))).hexdigest()

    def expected_costs(
        self, probabilities: tuple[float, float, float]
    ) -> dict[HoldDecision, float]:
        return {
            action: sum(
                cost * probability for cost, probability in zip(row, probabilities, strict=True)
            )
            for action, row in zip(self.actions, self.matrix, strict=True)
        }


def load_cost_matrix() -> CostMatrixArtifact:
    resource = files("razortrust").joinpath("config", "cost_matrix.v2.json")
    return CostMatrixArtifact.model_validate(json.loads(resource.read_text(encoding="utf-8")))


DEFAULT_COST_MATRIX = load_cost_matrix()
