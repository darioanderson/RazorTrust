from __future__ import annotations

from razortrust.alerts import CriticalFailureMonitor
from razortrust.domain import HoldDecision
from razortrust.ml.shadow import ShadowGate, evaluate_shadow_decisions


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, str | int]]] = []

    def critical(self, code: str, context: dict[str, str | int]) -> None:
        self.events.append((code, context))


def test_shadow_gate_blocks_candidate_unsafe_releases() -> None:
    report = evaluate_shadow_decisions(
        [HoldDecision.ESCALATE, HoldDecision.RELEASE, HoldDecision.RELEASE],
        [HoldDecision.RELEASE, HoldDecision.RELEASE, HoldDecision.RELEASE],
        incumbent_version="v1",
        candidate_version="v2",
        gate=ShadowGate(
            minimum_cases=3,
            maximum_disagreement_rate=0.5,
            maximum_unsafe_release_rate=0,
        ),
    )

    assert report.gate_passed is False
    assert report.unsafe_release_count == 1
    assert "UNSAFE_RELEASE_RATE_EXCEEDED" in report.gate_reasons


def test_critical_failure_monitor_alerts_first_and_repeated_failures() -> None:
    sink = RecordingSink()
    monitor = CriticalFailureMonitor(sink, repeat_threshold=3)

    assert [monitor.record("OPA_UNAVAILABLE") for _ in range(4)] == [True, False, True, False]
    assert [event[1]["occurrences"] for event in sink.events] == [1, 3]
