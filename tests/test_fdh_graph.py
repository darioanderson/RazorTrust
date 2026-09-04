from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from razortrust.ml.fdh_graph import (
    FDH_NODE_FEATURES,
    build_history_index,
    build_point_in_time_graph_snapshot,
    point_in_time_graph_features,
)


def _frame() -> pd.DataFrame:
    base = datetime(2018, 4, 1, 12, 0, 0)
    rows = [
        [1, base, 10, 100, 50.0, 0, 0, 1, 2],
        [2, base + timedelta(days=1), 11, 100, 20.0, 1, 1, 0, 0],
        [3, base + timedelta(days=8), 10, 101, 40.0, 2, 8, 0, 0],
        [4, base + timedelta(days=10), 10, 100, 35.0, 3, 10, 0, 0],
    ]
    return pd.DataFrame(
        rows,
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


def test_fdh_graph_features_use_only_history_and_delayed_labels() -> None:
    frame = _frame()
    history = build_history_index(frame)
    features = point_in_time_graph_features(history, 3, label_delay_days=7)
    assert features["customer_tx_count_30d"] == 2.0
    assert features["terminal_tx_count_30d"] == 2.0
    assert features["terminal_known_fraud_rate_30d_delay7d"] == 0.5

    future = frame.copy()
    future.loc[len(future)] = [
        5,
        frame.iloc[3]["TX_DATETIME"] + timedelta(days=1),
        10,
        100,
        999.0,
        4,
        11,
        1,
        3,
    ]
    after = point_in_time_graph_features(build_history_index(future), 3, label_delay_days=7)
    assert after == features


def test_fdh_snapshot_target_is_current_unlabelled_transaction() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    frame = _frame()
    history = build_history_index(frame)
    snapshot = build_point_in_time_graph_snapshot(history, 3, label_delay_days=7)
    assert snapshot.merchant_id == "fdh_tx:4"
    assert snapshot.data.x.shape[1] == 12
    assert snapshot.target_index < snapshot.data.x.shape[0]
    # The target transaction is connected to its known customer and terminal,
    # carries only score-time features, and receives no delayed fraud-history signal.
    target = snapshot.data.x[snapshot.target_index]
    feature_index = {name: idx for idx, name in enumerate(FDH_NODE_FEATURES)}
    assert target[feature_index["is_transaction"]].item() == 1.0
    assert target[feature_index["degree"]].item() == 2.0
    assert target[feature_index["current_amount_log1p"]].item() > 0.0
    assert target[feature_index["known_fraud_rate_delay7d"]].item() == 0.0

    relabelled = frame.copy()
    relabelled.loc[3, "TX_FRAUD"] = 1
    relabelled.loc[3, "TX_FRAUD_SCENARIO"] = 3
    after = build_point_in_time_graph_snapshot(
        build_history_index(relabelled),
        3,
        label_delay_days=7,
    )
    assert torch.equal(snapshot.data.x, after.data.x)
    assert torch.equal(snapshot.data.edge_index, after.data.edge_index)
