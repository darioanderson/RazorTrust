from __future__ import annotations

from uuid import UUID

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

from ..domain import HoldCase, StrictModel, TransactionEvent
from ..features import FEATURE_COLUMNS
from ..synthetic import SyntheticHold, SyntheticMerchant, TrueRiskState
from .dataset import build_training_frame
from .graph import point_in_time_graph_features

RING_ATTACK_FAMILIES = {"distributed_device_ring", "slow_low_ring", "mixed_evasion"}


class GraphAblationReport(StrictModel):
    evaluation_mode: str
    folds: int
    tabular_pr_auc: float
    graph_stats_pr_auc: float
    tabular_expected_cost: float
    graph_stats_expected_cost: float
    tabular_ring_recall: float
    graph_stats_ring_recall: float
    gate_passed: bool


def build_graph_training_frame(
    merchants: list[SyntheticMerchant],
    transactions: list[TransactionEvent],
    holds: list[SyntheticHold],
) -> pd.DataFrame:
    frame = build_training_frame(merchants, transactions, holds)
    graph_rows = []
    for synthetic_hold in holds:
        create = synthetic_hold.hold
        hold = HoldCase(hold_id=UUID(str(create.request_id)), **create.model_dump())
        graph_rows.append(
            point_in_time_graph_features(
                hold.merchant_id,
                transactions,
                hold.triggered_at,
                identifier_hmac_key=b"synthetic-graph-evaluation-key",
            )
        )
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(graph_rows)], axis=1)


def evaluate_graph_statistics_gate(
    frame: pd.DataFrame,
    *,
    seed: int = 42,
    folds: int = 5,
) -> GraphAblationReport:
    """Use grouped development folds only; the sealed test remains untouched."""
    graph_columns = [
        column
        for column in frame
        if column.startswith("graph_")
        or column
        in {
            "shared_device_count_1h",
            "shared_device_count_24h",
            "shared_customer_count",
            "new_neighbor_ratio",
            "device_merchant_degree_max",
            "device_reuse_velocity",
            "component_growth_24h",
            "two_hop_merchant_count",
            "two_hop_known_risk_density",
            "geo_overlap",
            "edge_creation_rate",
        }
    ]
    if not graph_columns:
        raise ValueError("graph ablation frame has no graph statistics")
    labels = (frame["true_risk_state"].astype(str) == TrueRiskState.RISKY).astype(int).to_numpy()
    groups = frame["merchant_id"].astype(str)
    ring_mask = frame["attack_family"].astype(str).isin(RING_ATTACK_FAMILIES).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    tabular_probabilities = np.zeros(len(frame))
    graph_probabilities = np.zeros(len(frame))
    for train_index, validation_index in splitter.split(frame, labels, groups):
        for columns, output in (
            (list(FEATURE_COLUMNS), tabular_probabilities),
            ([*FEATURE_COLUMNS, *graph_columns], graph_probabilities),
        ):
            model = XGBClassifier(
                objective="binary:logistic",
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                eval_metric="logloss",
                random_state=seed,
                n_jobs=1,
                tree_method="hist",
            )
            model.fit(frame.iloc[train_index][columns], labels[train_index])
            output[validation_index] = model.predict_proba(frame.iloc[validation_index][columns])[
                :, 1
            ]
    tabular_actions = tabular_probabilities >= 0.5
    graph_actions = graph_probabilities >= 0.5
    tabular_cost = _binary_cost(labels, tabular_actions)
    graph_cost = _binary_cost(labels, graph_actions)
    tabular_ring_recall = float(recall_score(labels[ring_mask], tabular_actions[ring_mask]))
    graph_ring_recall = float(recall_score(labels[ring_mask], graph_actions[ring_mask]))
    tabular_pr_auc = float(average_precision_score(labels, tabular_probabilities))
    graph_pr_auc = float(average_precision_score(labels, graph_probabilities))
    gate_passed = (
        graph_pr_auc >= tabular_pr_auc + 0.01
        and graph_cost < tabular_cost
        and graph_ring_recall >= tabular_ring_recall
    )
    return GraphAblationReport(
        evaluation_mode="grouped_development_cold_subgraph",
        folds=folds,
        tabular_pr_auc=round(tabular_pr_auc, 8),
        graph_stats_pr_auc=round(graph_pr_auc, 8),
        tabular_expected_cost=round(tabular_cost, 8),
        graph_stats_expected_cost=round(graph_cost, 8),
        tabular_ring_recall=round(tabular_ring_recall, 8),
        graph_stats_ring_recall=round(graph_ring_recall, 8),
        gate_passed=gate_passed,
    )


def _binary_cost(labels: np.ndarray, actions: np.ndarray) -> float:
    false_release = (~actions) & (labels == 1)
    false_hold = actions & (labels == 0)
    return float(np.mean(false_release * 100 + false_hold * 25))
