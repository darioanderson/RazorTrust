from __future__ import annotations

from datetime import timedelta

import pytest

from razortrust.ml.graphsage import build_point_in_time_graph_snapshot
from razortrust.synthetic import generate_dataset


def test_graphsage_snapshot_is_strictly_point_in_time() -> None:
    pytest.importorskip("torch_geometric")
    merchants, transactions, holds = generate_dataset(
        seed=91, merchants_per_family=1, transactions_per_merchant=12
    )
    cutoff = holds[0].hold.triggered_at
    before = build_point_in_time_graph_snapshot(merchants[0].merchant_id, transactions, cutoff)
    transactions.append(
        transactions[0].model_copy(update={"timestamp": cutoff + timedelta(hours=1)})
    )
    after = build_point_in_time_graph_snapshot(merchants[0].merchant_id, transactions, cutoff)
    assert before.data.x.tolist() == after.data.x.tolist()
    assert before.data.edge_index.tolist() == after.data.edge_index.tolist()
    assert before.target_index == after.target_index
