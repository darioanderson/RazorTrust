from __future__ import annotations

import numpy as np
import pandas as pd

from razortrust.domain import HoldDecision
from razortrust.ml.monitoring import EvtTailSeverity, OnlineChangeMonitor, batch_drift_report
from razortrust.ml.uncertainty import ApsConformalAbstainer


def test_mapie_aps_creates_sets_and_conservative_policy() -> None:
    probabilities = np.asarray([[0.85, 0.10, 0.05], [0.10, 0.80, 0.10], [0.05, 0.10, 0.85]] * 20)
    labels = np.asarray([0, 1, 2] * 20)
    abstainer = ApsConformalAbstainer(confidence_level=0.90).conformalize(probabilities, labels)
    sets = abstainer.prediction_sets(probabilities[:3])
    assert sets.shape == (3, 3)
    assert abstainer.evaluate(probabilities, labels).empirical_coverage >= 0.90
    assert (
        abstainer.policy_decision(np.asarray([True, True, False]), False)
        == HoldDecision.EVIDENCE_NEEDED
    )
    assert (
        abstainer.policy_decision(np.asarray([True, False, False]), True) == HoldDecision.ESCALATE
    )


def test_drift_change_detection_and_evt_tail_severity() -> None:
    rng = np.random.default_rng(42)
    reference = pd.DataFrame({"feature": rng.normal(0, 1, 500)})
    current = pd.DataFrame({"feature": rng.normal(2, 1, 500)})
    report = batch_drift_report(reference, current)
    assert report.features[0].drifted is True

    monitor = OnlineChangeMonitor(adwin_delta=0.01)
    alerts = [monitor.update(value) for value in [0.0] * 100 + [5.0] * 100]
    assert any(alert["adwin"] or alert["kswin"] for alert in alerts)

    scores = rng.exponential(scale=1, size=1_000)
    evt = EvtTailSeverity(quantile=0.95, min_exceedances=20).fit(scores)
    assert evt.survival_probability(float(scores.max() + 2)) < 0.1
