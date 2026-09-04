from __future__ import annotations

from datetime import timedelta

from razortrust.ml.sequence import point_in_time_sequence_features
from razortrust.synthetic import generate_dataset


def test_engineered_sequence_baseline_is_point_in_time() -> None:
    merchants, transactions, holds = generate_dataset(
        seed=81, merchants_per_family=1, transactions_per_merchant=12
    )
    merchant_id = merchants[0].merchant_id
    cutoff = holds[0].hold.triggered_at
    before = point_in_time_sequence_features(merchant_id, transactions, cutoff)
    transactions.append(
        transactions[0].model_copy(update={"timestamp": cutoff + timedelta(seconds=1)})
    )
    after = point_in_time_sequence_features(merchant_id, transactions, cutoff)
    assert before == after
    assert set(before) == {
        "burst_score_10m",
        "amount_autocorrelation",
        "auth_failure_run_max",
        "new_device_transition_rate",
    }
