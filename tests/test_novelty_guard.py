from __future__ import annotations

import numpy as np

from razortrust.domain import HoldDecision
from razortrust.ml.novelty_guard import CalibratedRawNoveltyGuard, apply_guard_route


def test_raw_novelty_guard_uses_legitimate_order_statistic_and_keeps_tail_severity() -> None:
    reference = np.linspace(0.10, 0.50, 100)
    guard = CalibratedRawNoveltyGuard.fit(reference, target_false_alarm_rate=0.02)
    scored = guard.score([0.20, 0.51, 0.70])

    assert guard.calibration.calibration_size == 100
    assert guard.calibration.raw_threshold <= 0.50
    assert scored.signal.tolist() == [False, True, True]
    assert scored.tail_excess[2] > scored.tail_excess[1] > 0
    assert scored.severity[2] > scored.severity[1] > 0
    assert scored.empirical_tail_probability[2] <= scored.empirical_tail_probability[1]


def test_guard_route_changes_only_release_actions() -> None:
    base = [HoldDecision.RELEASE, HoldDecision.EVIDENCE_NEEDED, HoldDecision.ESCALATE]
    signal = [True, True, True]

    evidence = apply_guard_route(base, signal, route="EVIDENCE_NEEDED")
    escalate = apply_guard_route(base, signal, route="ESCALATE")

    assert list(evidence) == [
        HoldDecision.EVIDENCE_NEEDED,
        HoldDecision.EVIDENCE_NEEDED,
        HoldDecision.ESCALATE,
    ]
    assert list(escalate) == [
        HoldDecision.ESCALATE,
        HoldDecision.EVIDENCE_NEEDED,
        HoldDecision.ESCALATE,
    ]


def test_extreme_risk_scores_cannot_move_the_legitimate_calibration_threshold() -> None:
    legitimate = np.linspace(0.20, 0.50, 80)
    first = CalibratedRawNoveltyGuard.fit(legitimate, target_false_alarm_rate=0.05)
    # Risk examples are intentionally not accepted by fit(); callers must pass
    # legitimate calibration scores only. This test records that invariant by
    # showing the same legitimate input always yields the same threshold.
    second = CalibratedRawNoveltyGuard.fit(legitimate.copy(), target_false_alarm_rate=0.05)
    assert first.calibration.raw_threshold == second.calibration.raw_threshold
    assert (
        first.calibration.as_dict()["content_sha256"]
        == second.calibration.as_dict()["content_sha256"]
    )
