from __future__ import annotations

import numpy as np
import pytest

from razortrust.domain import HoldDecision
from razortrust.ml.phase2b_unknown import apply_evidence_only_guard, consensus_signal


def test_two_of_three_consensus() -> None:
    signal = consensus_signal(
        "TWO_OF_THREE",
        np.asarray([True, True, False, False]),
        np.asarray([True, False, True, False]),
        np.asarray([False, True, True, False]),
    )
    assert signal.tolist() == [True, True, True, False]


def test_novelty_only_diverts_release_to_evidence() -> None:
    base = np.asarray(
        [HoldDecision.RELEASE, HoldDecision.EVIDENCE_NEEDED, HoldDecision.ESCALATE],
        dtype=object,
    )
    guarded = apply_evidence_only_guard(base, np.asarray([True, True, True]))
    assert guarded.tolist() == [
        HoldDecision.EVIDENCE_NEEDED,
        HoldDecision.EVIDENCE_NEEDED,
        HoldDecision.ESCALATE,
    ]


def test_unknown_consensus_rule_fails_closed() -> None:
    values = np.asarray([True, False])
    with pytest.raises(ValueError):
        consensus_signal("UNKNOWN", values, values, values)
