from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from ..domain import TransactionEvent

if TYPE_CHECKING:
    from torch import Tensor
    from torch_geometric.data import Data


NODE_FEATURES = (
    "is_merchant",
    "is_device",
    "is_customer",
    "is_geo",
    "degree",
    "transaction_count",
    "amount_sum_log1p",
    "failed_auth_rate",
)


@dataclass(frozen=True)
class GraphSnapshot:
    data: Data
    target_index: int
    merchant_id: str
    cutoff: datetime


def build_point_in_time_graph_snapshot(
    merchant_id: str,
    transactions: list[TransactionEvent],
    cutoff: datetime,
) -> GraphSnapshot:
    """Build a heterogeneous-identity graph using only events strictly before ``cutoff``.

    Raw identifiers are mapped to ephemeral integer nodes and never become model features.
    ``ring_id`` and future labels are deliberately ignored.
    """
    torch, Data = _torch_geometric()
    events = [event for event in transactions if event.timestamp < cutoff]
    node_index: dict[tuple[str, str], int] = {}
    stats: dict[int, dict[str, float]] = {}
    edges: set[tuple[int, int]] = set()

    def node(kind: str, value: str) -> int:
        key = (kind, value)
        if key not in node_index:
            idx = len(node_index)
            node_index[key] = idx
            stats[idx] = {
                "is_merchant": float(kind == "merchant"),
                "is_device": float(kind == "device"),
                "is_customer": float(kind == "customer"),
                "is_geo": float(kind == "geo"),
                "degree": 0.0,
                "transaction_count": 0.0,
                "amount_sum_log1p": 0.0,
                "failed_auth_rate": 0.0,
                "failed_auth_count": 0.0,
            }
        return node_index[key]

    target_index = node("merchant", merchant_id)
    for event in events:
        merchant = node("merchant", event.merchant_id)
        device = node("device", event.device_fingerprint)
        geo = node("geo", event.customer_geo)
        linked = [(merchant, device), (merchant, geo), (device, geo)]
        if event.customer_id:
            customer = node("customer", event.customer_id)
            linked.extend(((merchant, customer), (customer, device), (customer, geo)))
        for left, right in linked:
            if left == right:
                continue
            edges.add((left, right))
            edges.add((right, left))
        merchant_stats = stats[merchant]
        merchant_stats["transaction_count"] += 1.0
        merchant_stats["amount_sum_log1p"] += float(np.log1p(event.amount))
        merchant_stats["failed_auth_count"] += float(event.auth_status == "FAILED")

    for left, _ in edges:
        stats[left]["degree"] += 1.0
    for values in stats.values():
        count = values["transaction_count"]
        values["failed_auth_rate"] = values["failed_auth_count"] / max(1.0, count)

    x = torch.tensor(
        [[stats[idx][name] for name in NODE_FEATURES] for idx in range(len(stats))],
        dtype=torch.float32,
    )
    edge_pairs = sorted(edges)
    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return GraphSnapshot(
        data=Data(x=x, edge_index=edge_index),
        target_index=target_index,
        merchant_id=merchant_id,
        cutoff=cutoff,
    )


def make_graphsage_classifier(input_dim: int, hidden_dim: int = 32):
    """Construct the actual GraphSAGE binary challenger lazily.

    PyTorch Geometric is an optional research dependency, so importing the production
    API does not require it.
    """
    torch, _ = _torch_geometric()
    from torch_geometric.nn import SAGEConv

    if TYPE_CHECKING:
        from torch.nn import Module as TorchModule
    else:
        TorchModule = torch.nn.Module

    class GraphSageRiskClassifier(TorchModule):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = SAGEConv(input_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
            self.head = torch.nn.Linear(hidden_dim, 1)

        def forward(self, x: Tensor, edge_index: Tensor, target_index: int) -> Tensor:
            hidden = self.conv1(x, edge_index).relu()
            hidden = self.conv2(hidden, edge_index).relu()
            return self.head(hidden[target_index]).squeeze(-1)

    return GraphSageRiskClassifier()


def fit_graphsage_snapshots(
    snapshots: list[GraphSnapshot],
    labels: list[int],
    *,
    epochs: int = 40,
    hidden_dim: int = 32,
    learning_rate: float = 1e-3,
    seed: int = 42,
):
    """Train a small GraphSAGE challenger over point-in-time graph snapshots."""
    if len(snapshots) != len(labels) or not snapshots:
        raise ValueError("snapshots and labels must have the same non-zero length")
    torch, _ = _torch_geometric()
    torch.manual_seed(seed)
    model = make_graphsage_classifier(snapshots[0].data.x.shape[1], hidden_dim=hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        losses = []
        for snapshot, label in zip(snapshots, labels, strict=True):
            logit = model(snapshot.data.x, snapshot.data.edge_index, snapshot.target_index)
            target = torch.tensor(float(label), dtype=torch.float32)
            losses.append(criterion(logit, target))
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
    return model


def predict_graphsage_probability(model, snapshot: GraphSnapshot) -> float:
    torch, _ = _torch_geometric()
    model.eval()
    with torch.no_grad():
        logit = model(snapshot.data.x, snapshot.data.edge_index, snapshot.target_index)
        return float(torch.sigmoid(logit).item())


def _torch_geometric():
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:  # pragma: no cover - exercised only without optional extra
        raise RuntimeError("GraphSAGE requires the RazorTrust 'gnn' optional dependency") from exc
    return torch, Data
