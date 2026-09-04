from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import timedelta
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

from razortrust.features import FEATURE_COLUMNS
from razortrust.ml.dataset import build_training_frame
from razortrust.ml.sequence import point_in_time_sequence_features
from razortrust.synthetic import SyntheticHold, SyntheticMerchant, TrueRiskState
from razortrust.domain import TransactionEvent

FALSE_RELEASE_COST = 100
FALSE_HOLD_COST = 25
TEMPORAL_FEATURES = (
    "burst_score_10m",
    "amount_autocorrelation",
    "auth_failure_run_max",
    "new_device_transition_rate",
)


@dataclass(slots=True)
class MetricSet:
    pr_auc: float
    expected_cost: float
    risk_recall: float
    hold_rate: float


def _binary_cost(labels: np.ndarray, actions: np.ndarray) -> float:
    false_release = np.sum((labels == 1) & (~actions))
    false_hold = np.sum((labels == 0) & actions)
    return float((FALSE_RELEASE_COST * false_release + FALSE_HOLD_COST * false_hold) / len(labels))


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> MetricSet:
    actions = probabilities >= 0.5
    return MetricSet(
        pr_auc=round(float(average_precision_score(labels, probabilities)), 8),
        expected_cost=round(_binary_cost(labels, actions), 8),
        risk_recall=round(float(recall_score(labels, actions)), 8),
        hold_rate=round(float(np.mean(actions)), 8),
    )


def _build_model(seed: int) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
        tree_method="hist",
    )


def _evaluate_candidates(
    frame: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    candidates: dict[str, list[str]],
    *,
    seed: int,
    folds: int,
) -> tuple[dict[str, MetricSet], dict[str, list[MetricSet]]]:
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(frame, labels, groups))
    oof = {name: np.zeros(len(frame), dtype=float) for name in candidates}
    fold_metrics: dict[str, list[MetricSet]] = {name: [] for name in candidates}

    for fold_idx, (train_index, validation_index) in enumerate(splits):
        for name, columns in candidates.items():
            model = _build_model(seed + fold_idx)
            model.fit(frame.iloc[train_index][columns], labels[train_index])
            probabilities = model.predict_proba(frame.iloc[validation_index][columns])[:, 1]
            oof[name][validation_index] = probabilities
            fold_metrics[name].append(_metrics(labels[validation_index], probabilities))

    overall = {name: _metrics(labels, probabilities) for name, probabilities in oof.items()}
    return overall, fold_metrics


def _feature_stats(series: pd.Series) -> dict[str, float | int | bool]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values)
    finite_values = values[finite]
    if len(finite_values) == 0:
        return {
            "nan_count": int(np.isnan(values).sum()),
            "inf_count": int(np.isinf(values).sum()),
            "finite_count": 0,
            "zero_fraction": 0.0,
            "unique_count": 0,
            "zero_variance": True,
        }
    return {
        "nan_count": int(np.isnan(values).sum()),
        "inf_count": int(np.isinf(values).sum()),
        "finite_count": int(finite.sum()),
        "min": round(float(np.min(finite_values)), 8),
        "q01": round(float(np.quantile(finite_values, 0.01)), 8),
        "median": round(float(np.median(finite_values)), 8),
        "q99": round(float(np.quantile(finite_values, 0.99)), 8),
        "max": round(float(np.max(finite_values)), 8),
        "mean": round(float(np.mean(finite_values)), 8),
        "std": round(float(np.std(finite_values)), 8),
        "zero_fraction": round(float(np.mean(finite_values == 0.0)), 8),
        "unique_count": int(len(np.unique(finite_values))),
        "zero_variance": bool(np.std(finite_values) == 0.0),
    }


def _temporal_frame(
    transactions: list[TransactionEvent],
    holds: list[SyntheticHold],
    *,
    window_hours: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, float]] = []
    history_available: list[int] = []
    event_counts: list[int] = []
    point_in_time_violations = 0
    window_violations = 0

    tx_by_merchant: dict[str, list[TransactionEvent]] = {}
    for event in transactions:
        tx_by_merchant.setdefault(event.merchant_id, []).append(event)

    for item in holds:
        cutoff = item.hold.triggered_at
        lower = cutoff - timedelta(hours=window_hours)
        eligible = [
            event
            for event in tx_by_merchant.get(item.hold.merchant_id, [])
            if lower <= event.timestamp < cutoff
        ]
        point_in_time_violations += sum(event.timestamp >= cutoff for event in eligible)
        window_violations += sum(event.timestamp < lower for event in eligible)
        event_counts.append(len(eligible))
        history_available.append(int(bool(eligible)))
        rows.append(
            point_in_time_sequence_features(
                item.hold.merchant_id,
                transactions,
                cutoff,
                window_hours=window_hours,
            )
        )

    temporal = pd.DataFrame(rows).reset_index(drop=True)
    temporal["temporal_history_available"] = history_available
    temporal["temporal_event_count"] = event_counts
    audit = {
        "point_in_time_violations": int(point_in_time_violations),
        "window_violations": int(window_violations),
        "no_history_count": int(sum(1 for count in event_counts if count == 0)),
        "no_history_rate": round(float(np.mean(np.asarray(event_counts) == 0)), 8),
        "event_count": _feature_stats(temporal["temporal_event_count"]),
        "features": {name: _feature_stats(temporal[name]) for name in TEMPORAL_FEATURES},
    }
    return temporal, audit


def run_seed_diagnostic(
    merchants: list[SyntheticMerchant],
    transactions: list[TransactionEvent],
    holds: list[SyntheticHold],
    *,
    seed: int,
    folds: int,
    windows: Iterable[int],
) -> dict[str, object]:
    base = build_training_frame(merchants, transactions, holds).reset_index(drop=True)
    labels = base["true_risk_state"].astype(str).eq(TrueRiskState.RISKY).astype(int).to_numpy()
    groups = base["merchant_id"].astype(str).to_numpy()
    result: dict[str, object] = {
        "seed": seed,
        "case_count": len(base),
        "risk_prevalence": round(float(np.mean(labels)), 8),
        "windows": {},
    }

    for window_hours in windows:
        temporal, audit = _temporal_frame(transactions, holds, window_hours=window_hours)
        frame = pd.concat([base, temporal], axis=1)
        temporal_cols = list(TEMPORAL_FEATURES)
        full_cols = [*FEATURE_COLUMNS, *temporal_cols]
        candidates: dict[str, list[str]] = {
            "tabular": list(FEATURE_COLUMNS),
            "temporal_full": full_cols,
            "temporal_plus_history_flag": [*full_cols, "temporal_history_available"],
            "temporal_plus_history_count": [*full_cols, "temporal_history_available", "temporal_event_count"],
        }
        for feature in temporal_cols:
            candidates[f"add_one::{feature}"] = [*FEATURE_COLUMNS, feature]
            candidates[f"drop_one::{feature}"] = [
                *FEATURE_COLUMNS,
                *(name for name in temporal_cols if name != feature),
            ]

        overall, fold_metrics = _evaluate_candidates(
            frame,
            labels,
            groups,
            candidates,
            seed=seed,
            folds=folds,
        )
        tabular = overall["tabular"]
        full = overall["temporal_full"]
        audit["scale_note"] = (
            "XGBoost gbtree/hist is split-based; distribution is audited, but feature standardization "
            "is not treated as a primary fix."
        )
        audit["construction_fold_fitted_state"] = False
        audit["construction_note"] = (
            "Temporal features are deterministic pre-cutoff transaction summaries; no scaler, encoder, "
            "or target-derived transform is fitted outside CV folds."
        )
        result["windows"][str(window_hours)] = {
            "audit": audit,
            "metrics": {name: asdict(metrics) for name, metrics in overall.items()},
            "fold_metrics": {
                name: [asdict(metrics) for metrics in values]
                for name, values in fold_metrics.items()
            },
            "delta_vs_tabular": {
                "pr_auc": round(full.pr_auc - tabular.pr_auc, 8),
                "expected_cost": round(full.expected_cost - tabular.expected_cost, 8),
                "risk_recall": round(full.risk_recall - tabular.risk_recall, 8),
            },
        }
    return result


def summarize_seed_results(seed_results: list[dict[str, object]], windows: Iterable[int]) -> dict[str, object]:
    summary: dict[str, object] = {"windows": {}}
    for window in windows:
        window_key = str(window)
        rows = []
        for result in seed_results:
            data = result["windows"][window_key]
            rows.append(data["delta_vs_tabular"])
        pr = np.asarray([row["pr_auc"] for row in rows], dtype=float)
        cost = np.asarray([row["expected_cost"] for row in rows], dtype=float)
        recall = np.asarray([row["risk_recall"] for row in rows], dtype=float)
        summary["windows"][window_key] = {
            "seed_count": len(rows),
            "pr_auc_delta_mean": round(float(np.mean(pr)), 8),
            "pr_auc_delta_std": round(float(np.std(pr)), 8),
            "pr_auc_delta_min": round(float(np.min(pr)), 8),
            "pr_auc_delta_max": round(float(np.max(pr)), 8),
            "expected_cost_delta_mean": round(float(np.mean(cost)), 8),
            "risk_recall_delta_mean": round(float(np.mean(recall)), 8),
            "pr_auc_improved_seed_count": int(np.sum(pr > 0)),
            "cost_improved_seed_count": int(np.sum(cost < 0)),
            "recall_improved_seed_count": int(np.sum(recall > 0)),
        }
    return summary
