from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from razortrust.ml.fdh_graph import build_history_index
from razortrust.ml.fdh_hetero_graph import (
    CUSTOMER_FEATURES,
    TERMINAL_FEATURES,
    TRANSACTION_FEATURES,
    build_point_in_time_hetero_snapshot,
    fit_hetero_normalizer,
)


def _frame() -> pd.DataFrame:
    base = datetime(2018, 4, 1, 12, 0, 0)
    return pd.DataFrame(
        [
            [1, base, 10, 100, 10.0, 0, 0, 1, 2],
            [2, base + timedelta(days=1), 11, 100, 20.0, 1, 1, 0, 0],
            [3, base + timedelta(days=8), 10, 101, 30.0, 8, 8, 0, 0],
            [4, base + timedelta(days=10), 10, 100, 40.0, 10, 10, 0, 0],
        ],
        columns=[
            "TRANSACTION_ID",
            "TX_DATETIME",
            "CUSTOMER_ID",
            "TERMINAL_ID",
            "TX_AMOUNT",
            "TX_TIME_SECONDS",
            "TX_TIME_DAYS",
            "TX_FRAUD",
            "TX_FRAUD_SCENARIO",
        ],
    )


def test_hetero_snapshot_preserves_transaction_nodes_and_relations() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    snapshot = build_point_in_time_hetero_snapshot(build_history_index(_frame()), 3)
    data = snapshot.data
    assert data["transaction"].x.shape[1] == len(TRANSACTION_FEATURES)
    assert data["customer"].x.shape[1] == len(CUSTOMER_FEATURES)
    assert data["terminal"].x.shape[1] == len(TERMINAL_FEATURES)
    assert data["transaction"].x.shape[0] >= 4
    assert data[("customer", "makes", "transaction")].edge_index.shape[1] >= 4
    assert data[("transaction", "at", "terminal")].edge_index.shape[1] >= 4
    target = data["transaction"].x[snapshot.target_transaction_index]
    assert target[-1].item() == 1.0
    assert torch.isfinite(target).all()


def test_current_label_and_future_event_do_not_change_target_snapshot() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    frame = _frame()
    before = build_point_in_time_hetero_snapshot(build_history_index(frame), 3)

    relabelled = frame.copy()
    relabelled.loc[3, "TX_FRAUD"] = 1
    relabelled.loc[3, "TX_FRAUD_SCENARIO"] = 3
    after_label = build_point_in_time_hetero_snapshot(build_history_index(relabelled), 3)

    future = frame.copy()
    future.loc[len(future)] = [
        5,
        frame.iloc[3]["TX_DATETIME"] + timedelta(days=1),
        10,
        100,
        999.0,
        11,
        11,
        1,
        3,
    ]
    after_future = build_point_in_time_hetero_snapshot(build_history_index(future), 3)

    for node_type in ("transaction", "customer", "terminal"):
        assert torch.equal(before.data[node_type].x, after_label.data[node_type].x)
        assert torch.equal(before.data[node_type].x, after_future.data[node_type].x)
    for edge_type in before.data.edge_types:
        assert torch.equal(
            before.data[edge_type].edge_index,
            after_label.data[edge_type].edge_index,
        )
        assert torch.equal(
            before.data[edge_type].edge_index,
            after_future.data[edge_type].edge_index,
        )


def test_normalizer_is_fit_from_supplied_snapshots_only() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    history = build_history_index(_frame())
    early = build_point_in_time_hetero_snapshot(history, 2)
    normalizer = fit_hetero_normalizer([early])
    assert set(normalizer.mean) == {"transaction", "customer", "terminal"}
    assert len(normalizer.mean["transaction"]) == len(TRANSACTION_FEATURES)
