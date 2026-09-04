from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from razortrust.ml.fdh_graph import build_history_index
from razortrust.ml.fdh_hetero_graph import (
    apply_hetero_normalizer,
    build_point_in_time_hetero_snapshot,
    fit_hetero_normalizer,
)
from razortrust.ml.hetero_graphsage import fit_hetero_graphsage, hetero_graphsage_embedding


def test_hetero_graphsage_embedding_is_finite_and_fixed_width() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    base = datetime(2018, 4, 1, 12, 0, 0)
    rows = []
    for index in range(12):
        rows.append(
            [
                index + 1,
                base + timedelta(days=index),
                10 + (index % 3),
                100 + (index % 2),
                10.0 + index,
                index,
                index,
                int(index in {2, 8}),
                2 if index in {2, 8} else 0,
            ]
        )
    frame = pd.DataFrame(
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
    history = build_history_index(frame)
    raw = [build_point_in_time_hetero_snapshot(history, idx) for idx in range(4, 10)]
    normalizer = fit_hetero_normalizer(raw[:3])
    snapshots = [apply_hetero_normalizer(snapshot, normalizer) for snapshot in raw]
    labels = frame.iloc[range(4, 7)]["TX_FRAUD"].astype(int).tolist()
    model = fit_hetero_graphsage(snapshots[:3], labels, epochs=1, hidden_dim=8, seed=7)
    embedding = hetero_graphsage_embedding(model, snapshots[3])
    assert embedding.shape == (8,)
    assert np.isfinite(embedding).all()
