from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .fdh_hetero_graph import (
    CUSTOMER_FEATURES,
    TERMINAL_FEATURES,
    TRANSACTION_FEATURES,
    HeteroSnapshot,
)

if TYPE_CHECKING:
    from torch import Tensor


def make_hetero_graphsage_classifier(hidden_dim: int = 32):
    torch = _torch()
    from torch_geometric.nn import HeteroConv, SAGEConv

    class HeteroGraphSageRiskClassifier(torch.nn.Module):  # type: ignore[name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = HeteroConv(
                {
                    ("customer", "makes", "transaction"): SAGEConv(
                        (len(CUSTOMER_FEATURES), len(TRANSACTION_FEATURES)), hidden_dim
                    ),
                    ("terminal", "hosts", "transaction"): SAGEConv(
                        (len(TERMINAL_FEATURES), len(TRANSACTION_FEATURES)), hidden_dim
                    ),
                    ("transaction", "made_by", "customer"): SAGEConv(
                        (len(TRANSACTION_FEATURES), len(CUSTOMER_FEATURES)), hidden_dim
                    ),
                    ("transaction", "at", "terminal"): SAGEConv(
                        (len(TRANSACTION_FEATURES), len(TERMINAL_FEATURES)), hidden_dim
                    ),
                },
                aggr="sum",
            )
            self.conv2 = HeteroConv(
                {
                    ("customer", "makes", "transaction"): SAGEConv(
                        (hidden_dim, hidden_dim), hidden_dim
                    ),
                    ("terminal", "hosts", "transaction"): SAGEConv(
                        (hidden_dim, hidden_dim), hidden_dim
                    ),
                    ("transaction", "made_by", "customer"): SAGEConv(
                        (hidden_dim, hidden_dim), hidden_dim
                    ),
                    ("transaction", "at", "terminal"): SAGEConv(
                        (hidden_dim, hidden_dim), hidden_dim
                    ),
                },
                aggr="sum",
            )
            self.head = torch.nn.Linear(hidden_dim, 1)

        def embedding(self, data, target_index: int) -> Tensor:
            hidden = self.conv1(data.x_dict, data.edge_index_dict)
            hidden = {key: value.relu() for key, value in hidden.items()}
            hidden = self.conv2(hidden, data.edge_index_dict)
            hidden = {key: value.relu() for key, value in hidden.items()}
            return hidden["transaction"][target_index]

        def forward(self, data, target_index: int) -> Tensor:
            embedding = self.embedding(data, target_index)
            return self.head(embedding).squeeze(-1)

    return HeteroGraphSageRiskClassifier()


def fit_hetero_graphsage(
    snapshots: list[HeteroSnapshot],
    labels: list[int],
    *,
    epochs: int = 8,
    hidden_dim: int = 32,
    learning_rate: float = 1e-3,
    positive_weight: float | None = None,
    seed: int = 42,
):
    if len(snapshots) != len(labels) or not snapshots:
        raise ValueError("snapshots and labels must have the same non-zero length")
    torch = _torch()
    torch.manual_seed(seed)
    model = make_hetero_graphsage_classifier(hidden_dim=hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if positive_weight is None:
        positives = max(1, int(np.sum(labels)))
        negatives = max(1, len(labels) - positives)
        positive_weight = float(negatives / positives)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(positive_weight), dtype=torch.float32)
    )
    model.train()
    scale = 1.0 / float(len(snapshots))
    for _ in range(epochs):
        optimizer.zero_grad()
        for snapshot, label in zip(snapshots, labels, strict=True):
            logit = model(snapshot.data, snapshot.target_transaction_index)
            target = torch.tensor(float(label), dtype=torch.float32)
            loss = criterion(logit, target) * scale
            loss.backward()
        optimizer.step()
    return model


def hetero_graphsage_embedding(model, snapshot: HeteroSnapshot) -> np.ndarray:
    torch = _torch()
    model.eval()
    with torch.no_grad():
        embedding = model.embedding(snapshot.data, snapshot.target_transaction_index)
        return embedding.detach().cpu().numpy().astype(np.float64)


def hetero_graphsage_probability(model, snapshot: HeteroSnapshot) -> float:
    torch = _torch()
    model.eval()
    with torch.no_grad():
        logit = model(snapshot.data, snapshot.target_transaction_index)
        return float(torch.sigmoid(logit).item())


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional research dependency
        raise RuntimeError("Phase 3B V2 requires PyTorch") from exc
    return torch
