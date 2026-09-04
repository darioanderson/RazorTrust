from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .graphsage import GraphSnapshot, _torch_geometric

FDH_REQUIRED_COLUMNS = (
    "TRANSACTION_ID",
    "TX_DATETIME",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_AMOUNT",
    "TX_TIME_SECONDS",
    "TX_TIME_DAYS",
    "TX_FRAUD",
    "TX_FRAUD_SCENARIO",
)

FDH_TABULAR_FEATURES = (
    "tx_amount_log1p",
    "tx_hour_sin",
    "tx_hour_cos",
    "tx_weekend",
    "tx_night",
)

FDH_GRAPH_FEATURES = (
    "customer_tx_count_1d",
    "customer_tx_count_7d",
    "customer_tx_count_30d",
    "customer_avg_amount_1d",
    "customer_avg_amount_7d",
    "customer_avg_amount_30d",
    "terminal_tx_count_1d",
    "terminal_tx_count_7d",
    "terminal_tx_count_30d",
    "customer_unique_terminals_30d",
    "terminal_unique_customers_30d",
    "customer_terminal_prior_count_30d",
    "customer_known_fraud_rate_30d_delay7d",
    "terminal_known_fraud_rate_30d_delay7d",
)

FDH_NODE_FEATURES = (
    "is_transaction",
    "is_customer",
    "is_terminal",
    "degree",
    "transaction_count",
    "amount_sum_log1p",
    "known_fraud_rate_delay7d",
    "current_amount_log1p",
    "current_hour_sin",
    "current_hour_cos",
    "current_weekend",
    "current_night",
)


@dataclass(frozen=True)
class FDHHistoryIndex:
    frame: pd.DataFrame
    times_ns: np.ndarray
    customer_rows: dict[int, np.ndarray]
    terminal_rows: dict[int, np.ndarray]


@dataclass(frozen=True)
class PilotSplit:
    train: np.ndarray
    policy: np.ndarray
    test: np.ndarray


def load_fdh_transactions(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("*.pkl"))
    if not files:
        raise ValueError(f"no .pkl files found under {data_dir}")
    frames = [pd.read_pickle(path) for path in files]
    frame = pd.concat(frames, ignore_index=True)
    missing = set(FDH_REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"FDH dataset missing required columns: {sorted(missing)}")
    frame = frame.loc[:, list(FDH_REQUIRED_COLUMNS)].copy()
    frame["TX_DATETIME"] = pd.to_datetime(frame["TX_DATETIME"], errors="raise")
    frame["TX_FRAUD"] = pd.to_numeric(frame["TX_FRAUD"], errors="raise").astype(int)
    frame["TX_FRAUD_SCENARIO"] = pd.to_numeric(
        frame["TX_FRAUD_SCENARIO"], errors="raise"
    ).astype(int)
    if not set(frame["TX_FRAUD"].unique()).issubset({0, 1}):
        raise ValueError("TX_FRAUD must be binary")
    frame = frame.sort_values(["TX_DATETIME", "TRANSACTION_ID"], kind="mergesort").reset_index(
        drop=True
    )
    return frame


def build_history_index(frame: pd.DataFrame) -> FDHHistoryIndex:
    times_ns = frame["TX_DATETIME"].astype("int64").to_numpy()
    customer_rows = {
        int(key): np.asarray(group.index, dtype=np.int64)
        for key, group in frame.groupby("CUSTOMER_ID", sort=False)
    }
    terminal_rows = {
        int(key): np.asarray(group.index, dtype=np.int64)
        for key, group in frame.groupby("TERMINAL_ID", sort=False)
    }
    return FDHHistoryIndex(
        frame=frame,
        times_ns=times_ns,
        customer_rows=customer_rows,
        terminal_rows=terminal_rows,
    )


def tabular_features(row: pd.Series) -> dict[str, float]:
    timestamp = pd.Timestamp(row["TX_DATETIME"])
    angle = 2.0 * np.pi * (timestamp.hour + timestamp.minute / 60.0) / 24.0
    return {
        "tx_amount_log1p": float(np.log1p(float(row["TX_AMOUNT"]))),
        "tx_hour_sin": float(np.sin(angle)),
        "tx_hour_cos": float(np.cos(angle)),
        "tx_weekend": float(timestamp.dayofweek >= 5),
        "tx_night": float(timestamp.hour < 6),
    }


def point_in_time_graph_features(
    history: FDHHistoryIndex,
    row_index: int,
    *,
    label_delay_days: int = 7,
) -> dict[str, float]:
    row = history.frame.iloc[row_index]
    cutoff = pd.Timestamp(row["TX_DATETIME"])
    customer_id = int(row["CUSTOMER_ID"])
    terminal_id = int(row["TERMINAL_ID"])
    customer_history = _window_rows(history, history.customer_rows[customer_id], cutoff, 30)
    terminal_history = _window_rows(history, history.terminal_rows[terminal_id], cutoff, 30)

    features: dict[str, float] = {}
    for days in (1, 7, 30):
        customer_window = _window_rows(history, history.customer_rows[customer_id], cutoff, days)
        terminal_window = _window_rows(history, history.terminal_rows[terminal_id], cutoff, days)
        customer_amounts = history.frame.iloc[customer_window]["TX_AMOUNT"].to_numpy(dtype=float)
        features[f"customer_tx_count_{days}d"] = float(customer_window.size)
        features[f"customer_avg_amount_{days}d"] = float(
            customer_amounts.mean() if customer_amounts.size else 0.0
        )
        features[f"terminal_tx_count_{days}d"] = float(terminal_window.size)

    customer_30 = history.frame.iloc[customer_history]
    terminal_30 = history.frame.iloc[terminal_history]
    features["customer_unique_terminals_30d"] = float(customer_30["TERMINAL_ID"].nunique())
    features["terminal_unique_customers_30d"] = float(terminal_30["CUSTOMER_ID"].nunique())
    features["customer_terminal_prior_count_30d"] = float(
        np.sum(customer_30["TERMINAL_ID"].to_numpy(dtype=int) == terminal_id)
    )

    delayed_cutoff = cutoff - timedelta(days=label_delay_days)
    customer_known = _window_rows(
        history,
        history.customer_rows[customer_id],
        delayed_cutoff,
        30,
    )
    terminal_known = _window_rows(
        history,
        history.terminal_rows[terminal_id],
        delayed_cutoff,
        30,
    )
    features["customer_known_fraud_rate_30d_delay7d"] = _fraud_rate(
        history.frame, customer_known
    )
    features["terminal_known_fraud_rate_30d_delay7d"] = _fraud_rate(
        history.frame, terminal_known
    )
    return features


def build_point_in_time_graph_snapshot(
    history: FDHHistoryIndex,
    row_index: int,
    *,
    lookback_days: int = 30,
    label_delay_days: int = 7,
    max_history_events: int = 500,
) -> GraphSnapshot:
    torch, Data = _torch_geometric()
    row = history.frame.iloc[row_index]
    cutoff = pd.Timestamp(row["TX_DATETIME"])
    customer_id = int(row["CUSTOMER_ID"])
    terminal_id = int(row["TERMINAL_ID"])

    customer_rows = _window_rows(history, history.customer_rows[customer_id], cutoff, lookback_days)
    terminal_rows = _window_rows(history, history.terminal_rows[terminal_id], cutoff, lookback_days)
    historical_rows = np.unique(np.concatenate([customer_rows, terminal_rows]))
    if historical_rows.size > max_history_events:
        historical_rows = historical_rows[-max_history_events:]

    node_index: dict[tuple[str, int], int] = {}
    stats: dict[int, dict[str, float]] = {}
    edges: set[tuple[int, int]] = set()

    def node(kind: str, value: int) -> int:
        key = (kind, value)
        if key not in node_index:
            idx = len(node_index)
            node_index[key] = idx
            stats[idx] = {
                "is_transaction": float(kind == "transaction"),
                "is_customer": float(kind == "customer"),
                "is_terminal": float(kind == "terminal"),
                "degree": 0.0,
                "transaction_count": 0.0,
                "amount_sum_log1p": 0.0,
                "known_fraud_count": 0.0,
                "known_label_count": 0.0,
                "known_fraud_rate_delay7d": 0.0,
                "current_amount_log1p": 0.0,
                "current_hour_sin": 0.0,
                "current_hour_cos": 0.0,
                "current_weekend": 0.0,
                "current_night": 0.0,
            }
        return node_index[key]

    for hist_idx in historical_rows:
        event = history.frame.iloc[int(hist_idx)]
        customer = node("customer", int(event["CUSTOMER_ID"]))
        terminal = node("terminal", int(event["TERMINAL_ID"]))
        edges.add((customer, terminal))
        edges.add((terminal, customer))
        amount_log = float(np.log1p(float(event["TX_AMOUNT"])))
        for entity in (customer, terminal):
            stats[entity]["transaction_count"] += 1.0
            stats[entity]["amount_sum_log1p"] += amount_log
            if pd.Timestamp(event["TX_DATETIME"]) <= cutoff - timedelta(days=label_delay_days):
                stats[entity]["known_label_count"] += 1.0
                stats[entity]["known_fraud_count"] += float(int(event["TX_FRAUD"]))

    target_index = node("transaction", int(row["TRANSACTION_ID"]))
    target_customer = node("customer", customer_id)
    target_terminal = node("terminal", terminal_id)
    for entity in (target_customer, target_terminal):
        edges.add((target_index, entity))
        edges.add((entity, target_index))
    current = tabular_features(row)
    stats[target_index]["current_amount_log1p"] = current["tx_amount_log1p"]
    stats[target_index]["current_hour_sin"] = current["tx_hour_sin"]
    stats[target_index]["current_hour_cos"] = current["tx_hour_cos"]
    stats[target_index]["current_weekend"] = current["tx_weekend"]
    stats[target_index]["current_night"] = current["tx_night"]

    for left, _ in edges:
        stats[left]["degree"] += 1.0
    for values in stats.values():
        values["known_fraud_rate_delay7d"] = values["known_fraud_count"] / max(
            1.0, values["known_label_count"]
        )

    x = torch.tensor(
        [[stats[idx][name] for name in FDH_NODE_FEATURES] for idx in range(len(stats))],
        dtype=torch.float32,
    )
    edge_pairs = sorted(edges)
    edge_index = (
        torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        if edge_pairs
        else torch.empty((2, 0), dtype=torch.long)
    )
    return GraphSnapshot(
        data=Data(x=x, edge_index=edge_index),
        target_index=target_index,
        merchant_id=f"fdh_tx:{int(row['TRANSACTION_ID'])}",
        cutoff=cutoff.to_pydatetime(),
    )


def build_pilot_frame(
    history: FDHHistoryIndex,
    indices: np.ndarray,
    *,
    label_delay_days: int = 7,
) -> pd.DataFrame:
    rows = []
    for row_index in indices:
        row = history.frame.iloc[int(row_index)]
        values = {
            "row_index": int(row_index),
            "transaction_id": int(row["TRANSACTION_ID"]),
            "timestamp": pd.Timestamp(row["TX_DATETIME"]),
            "customer_id": int(row["CUSTOMER_ID"]),
            "terminal_id": int(row["TERMINAL_ID"]),
            "label": int(row["TX_FRAUD"]),
            "scenario": int(row["TX_FRAUD_SCENARIO"]),
        }
        values.update(tabular_features(row))
        values.update(
            point_in_time_graph_features(history, int(row_index), label_delay_days=label_delay_days)
        )
        rows.append(values)
    return pd.DataFrame(rows)


def select_chronological_case_control_pilot(
    frame: pd.DataFrame,
    *,
    seed: int,
    max_train: int,
    max_policy: int,
    max_test: int,
    legitimate_per_fraud: int = 3,
    warmup_days: int = 30,
) -> PilotSplit:
    if min(max_train, max_policy, max_test) < 20:
        raise ValueError("each pilot partition must allow at least 20 cases")
    start = pd.Timestamp(frame["TX_DATETIME"].min()) + timedelta(days=warmup_days)
    eligible = frame.index[frame["TX_DATETIME"] >= start].to_numpy(dtype=np.int64)
    if eligible.size < 100:
        raise ValueError("not enough post-warmup transactions")
    first = int(eligible[0])
    last = int(eligible[-1]) + 1
    span = last - first
    train_end = first + int(span * 0.60)
    policy_end = first + int(span * 0.80)
    rng = np.random.default_rng(seed)
    train = _case_control_sample(
        frame,
        np.arange(first, train_end),
        max_train,
        legitimate_per_fraud,
        rng,
    )
    policy = _case_control_sample(
        frame, np.arange(train_end, policy_end), max_policy, legitimate_per_fraud, rng
    )
    test = _case_control_sample(
        frame,
        np.arange(policy_end, last),
        max_test,
        legitimate_per_fraud,
        rng,
    )
    if not (frame.iloc[train]["TX_DATETIME"].max() < frame.iloc[policy]["TX_DATETIME"].min()):
        raise RuntimeError("train/policy chronology overlap")
    if not (frame.iloc[policy]["TX_DATETIME"].max() < frame.iloc[test]["TX_DATETIME"].min()):
        raise RuntimeError("policy/test chronology overlap")
    return PilotSplit(train=train, policy=policy, test=test)


def _case_control_sample(
    frame: pd.DataFrame,
    candidates: np.ndarray,
    max_cases: int,
    legitimate_per_fraud: int,
    rng: np.random.Generator,
) -> np.ndarray:
    labels = frame.iloc[candidates]["TX_FRAUD"].to_numpy(dtype=int)
    positive = candidates[labels == 1]
    negative = candidates[labels == 0]
    target_positive = min(len(positive), max(1, max_cases // (legitimate_per_fraud + 1)))
    if target_positive == 0:
        raise ValueError("pilot partition contains no fraud cases")
    positive_sample = (
        positive
        if len(positive) <= target_positive
        else rng.choice(positive, target_positive, replace=False)
    )
    negative_target = min(
        len(negative),
        max_cases - len(positive_sample),
        len(positive_sample) * legitimate_per_fraud,
    )
    negative_sample = rng.choice(negative, negative_target, replace=False)
    return np.asarray(sorted(np.concatenate([positive_sample, negative_sample])), dtype=np.int64)


def _window_rows(
    history: FDHHistoryIndex,
    entity_rows: np.ndarray,
    cutoff: pd.Timestamp,
    days: int,
) -> np.ndarray:
    cutoff_ns = int(cutoff.value)
    start_ns = int((cutoff - timedelta(days=days)).value)
    entity_times = history.times_ns[entity_rows]
    left = int(np.searchsorted(entity_times, start_ns, side="left"))
    right = int(np.searchsorted(entity_times, cutoff_ns, side="left"))
    return entity_rows[left:right]


def _fraud_rate(frame: pd.DataFrame, indices: np.ndarray) -> float:
    if indices.size == 0:
        return 0.0
    return float(frame.iloc[indices]["TX_FRAUD"].to_numpy(dtype=float).mean())
