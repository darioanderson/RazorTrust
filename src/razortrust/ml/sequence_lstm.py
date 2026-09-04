from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np

from ..domain import TransactionEvent

if TYPE_CHECKING:
    from torch import Tensor

SEQUENCE_FEATURES = (
    "log_amount",
    "gap_minutes_log1p",
    "auth_failed",
    "device_changed",
    "geo_changed",
    "refund_observed",
    "chargeback_observed",
)


def build_point_in_time_sequence_tensor(
    merchant_id: str,
    transactions: list[TransactionEvent],
    cutoff: datetime,
    *,
    max_events: int = 128,
    window_hours: int = 168,
):
    """Return a fixed-length event tensor using only information observable at cutoff."""
    torch = _torch()
    events = sorted(
        (
            event
            for event in transactions
            if event.merchant_id == merchant_id
            and cutoff - timedelta(hours=window_hours) <= event.timestamp < cutoff
        ),
        key=lambda event: event.timestamp,
    )[-max_events:]
    rows: list[list[float]] = []
    previous = None
    for event in events:
        gap_minutes = (
            (event.timestamp - previous.timestamp).total_seconds() / 60.0
            if previous is not None
            else 0.0
        )
        rows.append(
            [
                float(np.log1p(event.amount)),
                float(np.log1p(max(0.0, gap_minutes))),
                float(event.auth_status == "FAILED"),
                float(
                    previous is not None and event.device_fingerprint != previous.device_fingerprint
                ),
                float(previous is not None and event.customer_geo != previous.customer_geo),
                float(event.refund_timestamp is not None and event.refund_timestamp < cutoff),
                float(
                    event.chargeback_timestamp is not None and event.chargeback_timestamp < cutoff
                ),
            ]
        )
        previous = event
    length = len(rows)
    tensor = torch.zeros((max_events, len(SEQUENCE_FEATURES)), dtype=torch.float32)
    if rows:
        tensor[-length:] = torch.tensor(rows, dtype=torch.float32)
    return tensor, length


def make_lstm_classifier(input_dim: int = len(SEQUENCE_FEATURES), hidden_dim: int = 32):
    torch = _torch()

    if TYPE_CHECKING:
        from torch.nn import Module as TorchModule
    else:
        TorchModule = torch.nn.Module

    class SequenceRiskClassifier(TorchModule):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.head = torch.nn.Sequential(
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.Linear(hidden_dim, 1),
            )

        def forward(self, sequence: Tensor, length: int) -> Tensor:
            if sequence.ndim == 2:
                sequence = sequence.unsqueeze(0)
            if length <= 0:
                return self.head(torch.zeros((1, hidden_dim), device=sequence.device)).squeeze()
            trimmed = sequence[:, -length:, :]
            output, _ = self.lstm(trimmed)
            return self.head(output[:, -1, :]).squeeze()

    return SequenceRiskClassifier()


def fit_lstm_sequences(
    examples: list[tuple[Tensor, int]],
    labels: list[int],
    *,
    epochs: int = 40,
    hidden_dim: int = 32,
    learning_rate: float = 1e-3,
    seed: int = 42,
):
    if len(examples) != len(labels) or not examples:
        raise ValueError("examples and labels must have the same non-zero length")
    torch = _torch()
    torch.manual_seed(seed)
    model = make_lstm_classifier(examples[0][0].shape[-1], hidden_dim=hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        losses = []
        for (sequence, length), label in zip(examples, labels, strict=True):
            logit = model(sequence, length)
            target = torch.tensor(float(label), dtype=torch.float32)
            losses.append(criterion(logit, target))
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
    return model


def predict_lstm_probability(model, example: tuple[Tensor, int]) -> float:
    torch = _torch()
    model.eval()
    with torch.no_grad():
        return float(torch.sigmoid(model(*example)).item())


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "sequence LSTM requires the RazorTrust 'novelty' optional dependency"
        ) from exc
    return torch
