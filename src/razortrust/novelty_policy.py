from __future__ import annotations

import json
from importlib.resources import files

from pydantic import Field

from .domain import StrictModel


class NoveltyPolicyConfig(StrictModel):
    schema_version: str
    policy_version: str
    strong_signal_percentile: float = Field(ge=0, le=1)
    review_signal_count: int = Field(ge=1)
    override_signal_count: int = Field(ge=1)
    release_allowed_during_override: bool
    signals: list[str]


def load_novelty_policy() -> NoveltyPolicyConfig:
    resource = files("razortrust").joinpath("config", "novelty_policy.v1.json")
    return NoveltyPolicyConfig.model_validate(json.loads(resource.read_text(encoding="utf-8")))


DEFAULT_NOVELTY_POLICY = load_novelty_policy()
