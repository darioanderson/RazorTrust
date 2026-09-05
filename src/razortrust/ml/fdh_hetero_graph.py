from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from .fdh_graph import FDHHistoryIndex, _window_rows, tabular_features

TRANSACTION_FEATURES = (
    "amount_log1p",
    "hour_sin",
    "hour_cos",
    "weekend",
    "night",
    "age_days",
    "is_current",
)

CUSTOMER_FEATURES = (
    "tx_count_1d",
    "tx_count_7d",
    "tx_count_30d",
    "avg_amount_7d",
    "avg_amount_30d",
    "unique_terminals_30d",
    "known_fraud_rate_30d_delay7d",
    "is_target_customer",
)

TERMINAL_FEATURES = (
    "tx_count_1d",
    "tx_count_7d",
    "tx_count_30d",
    "avg_amount_7d",
    "avg_amount_30d",
    "unique_customers_30d",
    "known_fraud_rate_30d_delay7d",
    "is_target_terminal",
)

EDGE_TYPES = (
    ("customer", "makes", "transaction"),
    ("transaction", "made_by", "customer"),
    ("transaction", "at", "terminal"),
    ("terminal", "hosts", "transaction"),
)


@dataclass(frozen=True)
class HeteroSnapshot:
    data: Any
    target_transaction_index: int
    transaction_id: int
    cutoff: pd.Timestamp


@dataclass(frozen=True)
class HeteroNormalizer:
    mean: dict[str, np.ndarray]
    std: dict[str, np.ndarray]


def build_point_in_time_hetero_snapshot(
    history: FDHHistoryIndex,
    row_index: int,
    *,
    lookback_days: int = 30,
    label_delay_days: int = 7,
    max_history_events: int = 500,
) -> HeteroSnapshot:
    """Build an FDH heterogeneous graph using information available strictly at score time.

    Historical transaction nodes are preserved. The current transaction is added as an unlabelled
    target node. Fraud-derived entity features only use labels whose event timestamp is at least
    ``label_delay_days`` before the target cutoff.
    """
    torch, HeteroData = _torch_hetero()
    row = history.frame.iloc[int(row_index)]
    cutoff = pd.Timestamp(row["TX_DATETIME"])
    customer_id = int(row["CUSTOMER_ID"])
    terminal_id = int(row["TERMINAL_ID"])

    customer_rows = _window_rows(
        history, history.customer_rows[customer_id], cutoff, lookback_days
    )
    terminal_rows = _window_rows(
        history, history.terminal_rows[terminal_id], cutoff, lookback_days
    )
    historical_rows = np.unique(np.concatenate([customer_rows, terminal_rows]))
    if historical_rows.size > max_history_events:
        historical_rows = historical_rows[-max_history_events:]

    historical = history.frame.iloc[historical_rows]
    customer_ids = sorted(
        set(historical["CUSTOMER_ID"].astype(int).tolist()) | {customer_id}
    )
    terminal_ids = sorted(
        set(historical["TERMINAL_ID"].astype(int).tolist()) | {terminal_id}
    )
    customer_index = {value: idx for idx, value in enumerate(customer_ids)}
    terminal_index = {value: idx for idx, value in enumerate(terminal_ids)}

    transaction_features: list[list[float]] = []
    transaction_customer_edges: list[tuple[int, int]] = []
    transaction_terminal_edges: list[tuple[int, int]] = []

    for local_tx_index, hist_idx in enumerate(historical_rows):
        event = history.frame.iloc[int(hist_idx)]
        transaction_features.append(_transaction_features(event, cutoff, is_current=False))
        transaction_customer_edges.append(
            (local_tx_index, customer_index[int(event["CUSTOMER_ID"])])
        )
        transaction_terminal_edges.append(
            (local_tx_index, terminal_index[int(event["TERMINAL_ID"])])
        )

    target_transaction_index = len(transaction_features)
    transaction_features.append(_transaction_features(row, cutoff, is_current=True))
    transaction_customer_edges.append((target_transaction_index, customer_index[customer_id]))
    transaction_terminal_edges.append((target_transaction_index, terminal_index[terminal_id]))

    customer_features = [
        _customer_features(
            history,
            entity_id=value,
            cutoff=cutoff,
            target_customer_id=customer_id,
            label_delay_days=label_delay_days,
        )
        for value in customer_ids
    ]
    terminal_features = [
        _terminal_features(
            history,
            entity_id=value,
            cutoff=cutoff,
            target_terminal_id=terminal_id,
            label_delay_days=label_delay_days,
        )
        for value in terminal_ids
    ]

    data = HeteroData()
    data["transaction"].x = torch.tensor(transaction_features, dtype=torch.float32)
    data["customer"].x = torch.tensor(customer_features, dtype=torch.float32)
    data["terminal"].x = torch.tensor(terminal_features, dtype=torch.float32)

    tx_to_customer = _edge_tensor(torch, transaction_customer_edges)
    tx_to_terminal = _edge_tensor(torch, transaction_terminal_edges)
    data[("transaction", "made_by", "customer")].edge_index = tx_to_customer
    data[("customer", "makes", "transaction")].edge_index = tx_to_customer.flip(0)
    data[("transaction", "at", "terminal")].edge_index = tx_to_terminal
    data[("terminal", "hosts", "transaction")].edge_index = tx_to_terminal.flip(0)

    return HeteroSnapshot(
        data=data,
        target_transaction_index=target_transaction_index,
        transaction_id=int(row["TRANSACTION_ID"]),
        cutoff=cutoff,
    )


def fit_hetero_normalizer(snapshots: list[HeteroSnapshot]) -> HeteroNormalizer:
    if not snapshots:
        raise ValueError("cannot fit heterogeneous normalizer on zero snapshots")
    mean: dict[str, np.ndarray] = {}
    std: dict[str, np.ndarray] = {}
    for node_type in ("transaction", "customer", "terminal"):
        matrices = [
            snapshot.data[node_type].x.detach().cpu().numpy().astype(np.float64)
            for snapshot in snapshots
        ]
        values = np.concatenate(matrices, axis=0)
        node_mean = values.mean(axis=0)
        node_std = values.std(axis=0)
        node_std = np.where(node_std < 1e-8, 1.0, node_std)
        mean[node_type] = node_mean
        std[node_type] = node_std
    return HeteroNormalizer(mean=mean, std=std)


def apply_hetero_normalizer(
    snapshot: HeteroSnapshot,
    normalizer: HeteroNormalizer,
) -> HeteroSnapshot:
    torch, _ = _torch_hetero()
    data = snapshot.data.clone()
    for node_type in ("transaction", "customer", "terminal"):
        mean = torch.tensor(normalizer.mean[node_type], dtype=data[node_type].x.dtype)
        std = torch.tensor(normalizer.std[node_type], dtype=data[node_type].x.dtype)
        data[node_type].x = (data[node_type].x - mean) / std
    return HeteroSnapshot(
        data=data,
        target_transaction_index=snapshot.target_transaction_index,
        transaction_id=snapshot.transaction_id,
        cutoff=snapshot.cutoff,
    )


def _transaction_features(row: pd.Series, cutoff: pd.Timestamp, *, is_current: bool) -> list[float]:
    basic = tabular_features(row)
    event_time = pd.Timestamp(row["TX_DATETIME"])
    age_days = max(0.0, float((cutoff - event_time).total_seconds() / 86400.0))
    return [
        basic["tx_amount_log1p"],
        basic["tx_hour_sin"],
        basic["tx_hour_cos"],
        basic["tx_weekend"],
        basic["tx_night"],
        float(np.log1p(age_days)),
        float(is_current),
    ]


def _customer_features(
    history: FDHHistoryIndex,
    *,
    entity_id: int,
    cutoff: pd.Timestamp,
    target_customer_id: int,
    label_delay_days: int,
) -> list[float]:
    rows = history.customer_rows[entity_id]
    one = _window_rows(history, rows, cutoff, 1)
    seven = _window_rows(history, rows, cutoff, 7)
    thirty = _window_rows(history, rows, cutoff, 30)
    seven_amount = _mean_amount(history.frame, seven)
    thirty_amount = _mean_amount(history.frame, thirty)
    thirty_frame = history.frame.iloc[thirty]
    known = _window_rows(history, rows, cutoff - timedelta(days=label_delay_days), 30)
    return [
        float(one.size),
        float(seven.size),
        float(thirty.size),
        seven_amount,
        thirty_amount,
        float(thirty_frame["TERMINAL_ID"].nunique()) if thirty.size else 0.0,
        _fraud_rate(history.frame, known),
        float(entity_id == target_customer_id),
    ]


def _terminal_features(
    history: FDHHistoryIndex,
    *,
    entity_id: int,
    cutoff: pd.Timestamp,
    target_terminal_id: int,
    label_delay_days: int,
) -> list[float]:
    rows = history.terminal_rows[entity_id]
    one = _window_rows(history, rows, cutoff, 1)
    seven = _window_rows(history, rows, cutoff, 7)
    thirty = _window_rows(history, rows, cutoff, 30)
    seven_amount = _mean_amount(history.frame, seven)
    thirty_amount = _mean_amount(history.frame, thirty)
    thirty_frame = history.frame.iloc[thirty]
    known = _window_rows(history, rows, cutoff - timedelta(days=label_delay_days), 30)
    return [
        float(one.size),
        float(seven.size),
        float(thirty.size),
        seven_amount,
        thirty_amount,
        float(thirty_frame["CUSTOMER_ID"].nunique()) if thirty.size else 0.0,
        _fraud_rate(history.frame, known),
        float(entity_id == target_terminal_id),
    ]


def _mean_amount(frame: pd.DataFrame, indices: np.ndarray) -> float:
    if indices.size == 0:
        return 0.0
    return float(frame.iloc[indices]["TX_AMOUNT"].to_numpy(dtype=float).mean())


def _fraud_rate(frame: pd.DataFrame, indices: np.ndarray) -> float:
    if indices.size == 0:
        return 0.0
    return float(frame.iloc[indices]["TX_FRAUD"].to_numpy(dtype=float).mean())


def _edge_tensor(torch, pairs: list[tuple[int, int]]):
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def _torch_hetero():
    try:
        import torch
        from torch_geometric.data import HeteroData
    except ImportError as exc:  # pragma: no cover - optional research dependency
        raise RuntimeError("Phase 3B V2 requires the RazorTrust 'gnn' dependency") from exc
    return torch, HeteroData
