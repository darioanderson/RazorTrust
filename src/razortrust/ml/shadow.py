from __future__ import annotations

from collections import Counter

from pydantic import Field

from ..domain import HoldDecision, StrictModel


class ShadowGate(StrictModel):
    maximum_disagreement_rate: float = Field(default=0.05, ge=0, le=1)
    maximum_unsafe_release_rate: float = Field(default=0.0, ge=0, le=1)
    minimum_cases: int = Field(default=1000, ge=1)


class ShadowEvaluation(StrictModel):
    incumbent_version: str
    candidate_version: str
    case_count: int
    disagreement_count: int
    disagreement_rate: float
    unsafe_release_count: int
    unsafe_release_rate: float
    transition_counts: dict[str, int]
    gate_passed: bool
    gate_reasons: list[str]
    evaluation_mode: str = "SHADOW_NO_ENFORCEMENT"


def evaluate_shadow_decisions(
    incumbent: list[HoldDecision],
    candidate: list[HoldDecision],
    *,
    incumbent_version: str,
    candidate_version: str,
    gate: ShadowGate | None = None,
) -> ShadowEvaluation:
    gate = gate or ShadowGate()
    if len(incumbent) != len(candidate) or not incumbent:
        raise ValueError("shadow decision sequences must have the same non-zero length")
    transitions = Counter(f"{old}->{new}" for old, new in zip(incumbent, candidate, strict=True))
    disagreement_count = sum(old != new for old, new in zip(incumbent, candidate, strict=True))
    unsafe_release_count = sum(
        new == HoldDecision.RELEASE and old != HoldDecision.RELEASE
        for old, new in zip(incumbent, candidate, strict=True)
    )
    case_count = len(incumbent)
    disagreement_rate = disagreement_count / case_count
    unsafe_release_rate = unsafe_release_count / case_count
    reasons: list[str] = []
    if case_count < gate.minimum_cases:
        reasons.append("INSUFFICIENT_SHADOW_CASES")
    if disagreement_rate > gate.maximum_disagreement_rate:
        reasons.append("DISAGREEMENT_RATE_EXCEEDED")
    if unsafe_release_rate > gate.maximum_unsafe_release_rate:
        reasons.append("UNSAFE_RELEASE_RATE_EXCEEDED")
    return ShadowEvaluation(
        incumbent_version=incumbent_version,
        candidate_version=candidate_version,
        case_count=case_count,
        disagreement_count=disagreement_count,
        disagreement_rate=round(disagreement_rate, 8),
        unsafe_release_count=unsafe_release_count,
        unsafe_release_rate=round(unsafe_release_rate, 8),
        transition_counts=dict(sorted(transitions.items())),
        gate_passed=not reasons,
        gate_reasons=reasons or ["SHADOW_GATE_PASSED"],
    )
