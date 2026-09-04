from __future__ import annotations

from datetime import timedelta

from razortrust.ml.sequence_lstm import build_point_in_time_sequence_tensor
from razortrust.synthetic import generate_dataset


def test_lstm_tensor_is_strictly_point_in_time() -> None:
    merchants, transactions, holds = generate_dataset(
        seed=92, merchants_per_family=1, transactions_per_merchant=12
    )
    cutoff = holds[0].hold.triggered_at
    before, before_length = build_point_in_time_sequence_tensor(
        merchants[0].merchant_id, transactions, cutoff
    )
    transactions.append(
        transactions[0].model_copy(update={"timestamp": cutoff + timedelta(seconds=1)})
    )
    after, after_length = build_point_in_time_sequence_tensor(
        merchants[0].merchant_id, transactions, cutoff
    )
    assert before_length == after_length
    assert before.tolist() == after.tolist()
