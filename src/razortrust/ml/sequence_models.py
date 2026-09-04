from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

from ..domain import TransactionEvent
from ..features import FEATURE_COLUMNS
from ..synthetic import SyntheticHold, SyntheticMerchant, TrueRiskState
from .dataset import build_training_frame
from .sequence import point_in_time_sequence_features

SEQUENCE_FEATURES = (
    "log_amount",
    "log_gap_seconds",
    "auth_failed",
    "device_transition",
    "geo_transition",
    "position",
)

FALSE_RELEASE_COST = 100
FALSE_HOLD_COST = 25


@dataclass(slots=True)
class SequenceExample:
    values: np.ndarray
    length: int
    label: int
    merchant_id: str
    attack_family: str | None


@dataclass(slots=True)
class SequenceBaselineReport:
    evaluation_mode: str
    folds: int
    tabular_pr_auc: float
    temporal_pr_auc: float
    tabular_expected_cost: float
    temporal_expected_cost: float
    tabular_risk_recall: float
    temporal_risk_recall: float
    tabular_hold_rate: float
    temporal_hold_rate: float
    case_count: int
    gate_passed: bool


@dataclass(slots=True)
class SequenceModelReport:
    model_type: str
    evaluation_mode: str
    folds: int
    pr_auc: float
    expected_cost: float
    risk_recall: float
    hold_rate: float
    case_count: int


def evaluate_temporal_feature_gate(
    merchants: list[SyntheticMerchant],
    transactions: list[TransactionEvent],
    holds: list[SyntheticHold],
    *,
    seed: int = 42,
    folds: int = 3,
) -> SequenceBaselineReport:
    """Evaluate 13-feature tabular vs 13+engineered-temporal on grouped development folds."""
    frame = build_training_frame(merchants, transactions, holds).reset_index(drop=True)
    temporal_rows = [
        point_in_time_sequence_features(
            item.hold.merchant_id,
            transactions,
            item.hold.triggered_at,
        )
        for item in holds
    ]
    if not temporal_rows:
        raise ValueError("no engineered temporal examples available")
    temporal_frame = pd.DataFrame(temporal_rows).reset_index(drop=True)
    frame = pd.concat([frame, temporal_frame], axis=1)
    temporal_columns = list(temporal_frame.columns)

    labels = (
        frame["true_risk_state"].astype(str).eq(TrueRiskState.RISKY).astype(int).to_numpy()
    )
    groups = frame["merchant_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    tabular_probabilities = np.zeros(len(frame), dtype=float)
    temporal_probabilities = np.zeros(len(frame), dtype=float)

    for train_index, validation_index in splitter.split(frame, labels, groups):
        for columns, output in (
            (list(FEATURE_COLUMNS), tabular_probabilities),
            ([*FEATURE_COLUMNS, *temporal_columns], temporal_probabilities),
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
            output[validation_index] = model.predict_proba(
                frame.iloc[validation_index][columns]
            )[:, 1]

    tabular_actions = tabular_probabilities >= 0.5
    temporal_actions = temporal_probabilities >= 0.5
    tabular_pr_auc = float(average_precision_score(labels, tabular_probabilities))
    temporal_pr_auc = float(average_precision_score(labels, temporal_probabilities))
    tabular_cost = _binary_cost(labels, tabular_actions)
    temporal_cost = _binary_cost(labels, temporal_actions)
    tabular_risk_recall = float(recall_score(labels, tabular_actions))
    temporal_risk_recall = float(recall_score(labels, temporal_actions))
    gate_passed = (
        temporal_pr_auc >= tabular_pr_auc + 0.01
        and temporal_cost < tabular_cost
        and temporal_risk_recall >= tabular_risk_recall
    )
    return SequenceBaselineReport(
        evaluation_mode="grouped_development_temporal_ablation_cold_merchant",
        folds=folds,
        tabular_pr_auc=round(tabular_pr_auc, 8),
        temporal_pr_auc=round(temporal_pr_auc, 8),
        tabular_expected_cost=round(tabular_cost, 8),
        temporal_expected_cost=round(temporal_cost, 8),
        tabular_risk_recall=round(tabular_risk_recall, 8),
        temporal_risk_recall=round(temporal_risk_recall, 8),
        tabular_hold_rate=round(float(np.mean(tabular_actions)), 8),
        temporal_hold_rate=round(float(np.mean(temporal_actions)), 8),
        case_count=len(frame),
        gate_passed=gate_passed,
    )


def build_sequence_example(
    *,
    merchant_id: str,
    transactions: list[TransactionEvent],
    cutoff: datetime,
    label: int,
    attack_family: str,
    window_hours: int = 24,
    max_events: int = 128,
) -> SequenceExample:
    """Create a fixed-length, point-in-time event sequence without outcome leakage."""
    events = sorted(
        (
            event
            for event in transactions
            if event.merchant_id == merchant_id
            and cutoff - timedelta(hours=window_hours) <= event.timestamp < cutoff
        ),
        key=lambda item: item.timestamp,
    )[-max_events:]
    matrix = np.zeros((max_events, len(SEQUENCE_FEATURES)), dtype=np.float32)
    previous: TransactionEvent | None = None
    for row, event in enumerate(events):
        gap_seconds = (
            0.0
            if previous is None
            else (event.timestamp - previous.timestamp).total_seconds()
        )
        matrix[row] = np.asarray(
            [
                min(2.0, np.log1p(max(0.0, float(event.amount))) / 12.0),
                min(2.0, np.log1p(max(0.0, gap_seconds)) / 12.0),
                1.0 if event.auth_status == "FAILED" else 0.0,
                0.0
                if previous is None
                else float(event.device_fingerprint != previous.device_fingerprint),
                0.0
                if previous is None
                else float(event.customer_geo != previous.customer_geo),
                float(row / max(1, max_events - 1)),
            ],
            dtype=np.float32,
        )
        previous = event
    return SequenceExample(
        values=matrix,
        length=max(1, len(events)),
        label=int(label),
        merchant_id=merchant_id,
        attack_family=attack_family,
    )


def build_sequence_examples(
    transactions: list[TransactionEvent],
    holds: list[SyntheticHold],
    *,
    max_events: int = 128,
) -> list[SequenceExample]:
    return [
        build_sequence_example(
            merchant_id=item.hold.merchant_id,
            transactions=transactions,
            cutoff=item.hold.triggered_at,
            label=int(item.true_risk_state == TrueRiskState.RISKY),
            attack_family=item.attack_family,
            max_events=max_events,
        )
        for item in holds
    ]


def evaluate_sequence_challenger(
    examples: list[SequenceExample],
    *,
    model_type: Literal["lstm", "transformer"],
    seed: int = 42,
    folds: int = 3,
    epochs: int = 30,
    hidden_size: int = 32,
) -> SequenceModelReport:
    torch = _require_torch()
    if len(examples) < folds * 2:
        raise ValueError("not enough sequence examples for grouped evaluation")
    labels = np.asarray([item.label for item in examples], dtype=int)
    groups = np.asarray([item.merchant_id for item in examples], dtype=object)
    probabilities = np.zeros(len(examples), dtype=float)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)

    for fold, (train_index, validation_index) in enumerate(
        splitter.split(np.zeros(len(examples)), labels, groups)
    ):
        torch.manual_seed(seed + fold)
        model = _build_sequence_model(
            torch,
            model_type,
            len(SEQUENCE_FEATURES),
            hidden_size,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            losses = []
            for index in train_index:
                item = examples[int(index)]
                values = torch.tensor(item.values[None, :, :], dtype=torch.float32)
                lengths = torch.tensor([item.length], dtype=torch.long)
                target = torch.tensor([float(item.label)], dtype=torch.float32)
                losses.append(loss_fn(model(values, lengths), target))
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            for index in validation_index:
                item = examples[int(index)]
                values = torch.tensor(item.values[None, :, :], dtype=torch.float32)
                lengths = torch.tensor([item.length], dtype=torch.long)
                probabilities[int(index)] = float(
                    torch.sigmoid(model(values, lengths))[0].item()
                )

    actions = probabilities >= 0.5
    return SequenceModelReport(
        model_type=model_type,
        evaluation_mode="grouped_development_sequence_cold_merchant",
        folds=folds,
        pr_auc=round(float(average_precision_score(labels, probabilities)), 8),
        expected_cost=round(_binary_cost(labels, actions), 8),
        risk_recall=round(float(recall_score(labels, actions)), 8),
        hold_rate=round(float(np.mean(actions)), 8),
        case_count=len(examples),
    )


def sequence_gate_passed(
    *,
    temporal_pr_auc: float,
    temporal_expected_cost: float,
    temporal_risk_recall: float,
    challenger: SequenceModelReport,
) -> bool:
    return (
        challenger.pr_auc >= temporal_pr_auc + 0.01
        and challenger.expected_cost < temporal_expected_cost
        and challenger.risk_recall >= temporal_risk_recall
    )


def _build_sequence_model(
    torch: Any,
    model_type: str,
    input_size: int,
    hidden_size: int,
) -> Any:
    if model_type == "lstm":

        class LstmClassifier(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
                self.head = torch.nn.Linear(hidden_size, 1)

            def forward(self, values: Any, lengths: Any) -> Any:
                output, _ = self.lstm(values)
                index = (lengths - 1).clamp(min=0)
                selected = output[torch.arange(output.shape[0]), index]
                return self.head(selected).squeeze(-1)

        return LstmClassifier()

    if model_type == "transformer":

        class TransformerClassifier(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.project = torch.nn.Linear(input_size, hidden_size)
                layer = torch.nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=4,
                    dim_feedforward=hidden_size * 2,
                    dropout=0.1,
                    batch_first=True,
                    activation="gelu",
                )
                self.encoder = torch.nn.TransformerEncoder(layer, num_layers=2)
                self.head = torch.nn.Linear(hidden_size, 1)

            def forward(self, values: Any, lengths: Any) -> Any:
                steps = torch.arange(values.shape[1], device=values.device)[None, :]
                padding_mask = steps >= lengths[:, None]
                encoded = self.encoder(
                    self.project(values),
                    src_key_padding_mask=padding_mask,
                )
                valid = (~padding_mask).unsqueeze(-1)
                pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
                return self.head(pooled).squeeze(-1)

        return TransformerClassifier()

    raise ValueError("model_type must be 'lstm' or 'transformer'")


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency gate is environment specific
        raise RuntimeError(
            "sequence challengers require the RazorTrust 'sequence' extra"
        ) from exc
    return torch


def _binary_cost(labels: np.ndarray, actions: np.ndarray) -> float:
    false_release = (~actions) & (labels == 1)
    false_hold = actions & (labels == 0)
    return float(
        np.mean(false_release * FALSE_RELEASE_COST + false_hold * FALSE_HOLD_COST)
    )
