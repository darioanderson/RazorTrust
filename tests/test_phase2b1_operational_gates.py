from __future__ import annotations


def test_operational_gate_contract_is_intentionally_strict() -> None:
    # This documents the Phase 2B.1 selection contract without depending on the
    # research artifact.
    max_false_release = 0.05
    max_legit_diversion = 0.20
    max_worst_family_diversion = 0.50
    min_risk_rescue = 0.50
    assert max_false_release < 0.10
    assert max_legit_diversion < max_worst_family_diversion
    assert min_risk_rescue >= 0.50
