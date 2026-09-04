from __future__ import annotations

import hashlib
import json
from importlib.resources import files

from .audit import canonical_json
from .domain import HoldDecision, StrictModel


class DecisionPolicyConfig(StrictModel):
    schema_version: str
    policy_version: str
    maximum_evidence_rounds: int
    evidence_release_risk_cap: float
    maximum_false_release_rate: float
    system_error_action: HoldDecision
    policy_unavailable_action: HoldDecision

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.model_dump(mode="json"))).hexdigest()


def load_decision_policy() -> DecisionPolicyConfig:
    resource = files("razortrust").joinpath("config", "decision_policy.v2.json")
    return DecisionPolicyConfig.model_validate(json.loads(resource.read_text(encoding="utf-8")))


DEFAULT_DECISION_POLICY = load_decision_policy()
