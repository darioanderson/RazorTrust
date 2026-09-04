from __future__ import annotations

from razortrust.ml.dataset import build_training_frame
from razortrust.ml.loafo import evaluate_leave_one_attack_family_out
from razortrust.synthetic import RISK_FAMILIES, generate_dataset


def test_loafo_reports_every_held_family_and_legitimate_false_alarms() -> None:
    frame = build_training_frame(
        *generate_dataset(seed=71, merchants_per_family=2, transactions_per_merchant=12)
    )
    report = evaluate_leave_one_attack_family_out(frame, novelty_threshold=0.90)
    assert {result.held_out_family for result in report.families} == set(RISK_FAMILIES)
    assert all(0 <= result.combined_recall <= 1 for result in report.families)
    assert all(0 <= result.legitimate_novelty_fpr <= 1 for result in report.families)
