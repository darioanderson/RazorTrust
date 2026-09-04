from __future__ import annotations

import hashlib
import hmac
from datetime import datetime, timedelta

import networkx as nx

from ..domain import TransactionEvent


def point_in_time_graph_features(
    merchant_id: str,
    transactions: list[TransactionEvent],
    cutoff: datetime,
    *,
    identifier_hmac_key: bytes,
    known_risk_merchants: set[str] | None = None,
) -> dict[str, float]:
    """Compute interpretable merchant-network statistics using only pre-cutoff events."""
    graph = nx.Graph()
    merchant_node = _node("merchant", merchant_id, identifier_hmac_key)
    target_devices_1h: set[str] = set()
    target_devices_24h: set[str] = set()
    historic_target_devices: set[str] = set()
    target_geos: set[str] = set()
    other_merchant_geos: set[str] = set()
    recent_edge_count = 0
    historic_nodes: set[str] = set()
    for event in transactions:
        if event.timestamp >= cutoff:
            continue
        event_merchant = _node("merchant", event.merchant_id, identifier_hmac_key)
        device = _node("device", event.device_fingerprint, identifier_hmac_key)
        geo = _node("geo", event.customer_geo, identifier_hmac_key)
        graph.add_edge(event_merchant, device)
        graph.add_edge(device, geo)
        if event.timestamp < cutoff - timedelta(hours=24):
            historic_nodes.update((event_merchant, device, geo))
        if event.customer_id:
            customer = _node("customer", event.customer_id, identifier_hmac_key)
            graph.add_edge(customer, device)
            graph.add_edge(customer, event_merchant)
        if event.merchant_id == merchant_id:
            target_geos.add(geo)
            if event.timestamp >= cutoff - timedelta(hours=1):
                target_devices_1h.add(device)
            if event.timestamp >= cutoff - timedelta(hours=24):
                target_devices_24h.add(device)
                recent_edge_count += 1
            else:
                historic_target_devices.add(device)
        else:
            other_merchant_geos.add(geo)
    devices = (
        {node for node in graph.neighbors(merchant_node) if node.startswith("device:")}
        if merchant_node in graph
        else set()
    )
    device_merchant_degrees = {
        device: sum(neighbor.startswith("merchant:") for neighbor in graph.neighbors(device))
        for device in devices
    }
    two_hop_merchants = {
        neighbor
        for device in devices
        for neighbor in graph.neighbors(device)
        if neighbor.startswith("merchant:") and neighbor != merchant_node
    }
    shared_customers = (
        {
            neighbor
            for neighbor in graph.neighbors(merchant_node)
            if neighbor.startswith("customer:")
            and sum(candidate.startswith("merchant:") for candidate in graph.neighbors(neighbor))
            > 1
        }
        if merchant_node in graph
        else set()
    )
    new_devices = target_devices_24h - historic_target_devices
    known_risk_nodes = {
        _node("merchant", value, identifier_hmac_key) for value in (known_risk_merchants or set())
    }
    component = (
        nx.node_connected_component(graph, merchant_node) if merchant_node in graph else set()
    )
    return {
        "graph_device_count": float(len(devices)),
        "graph_connected_nodes": float(len(component)),
        "shared_device_count_1h": float(
            sum(device_merchant_degrees.get(device, 0) > 1 for device in target_devices_1h)
        ),
        "shared_device_count_24h": float(
            sum(device_merchant_degrees.get(device, 0) > 1 for device in target_devices_24h)
        ),
        "shared_customer_count": float(len(shared_customers)),
        "new_neighbor_ratio": float(len(new_devices) / max(1, len(target_devices_24h))),
        "device_merchant_degree_max": float(max(device_merchant_degrees.values(), default=0)),
        "device_reuse_velocity": float(
            sum(max(0, device_merchant_degrees.get(device, 0) - 1) for device in target_devices_1h)
        ),
        "component_growth_24h": float(len(component - historic_nodes)),
        "two_hop_merchant_count": float(len(two_hop_merchants)),
        "two_hop_known_risk_density": float(
            len(two_hop_merchants & known_risk_nodes) / max(1, len(two_hop_merchants))
        ),
        "geo_overlap": float(len(target_geos & other_merchant_geos) / max(1, len(target_geos))),
        "edge_creation_rate": float(recent_edge_count / 24),
    }


def _node(kind: str, identifier: str, key: bytes) -> str:
    if len(key) < 16:
        raise ValueError("graph identifier HMAC key must contain at least 16 bytes")
    token = hmac.new(key, identifier.encode(), hashlib.sha256).hexdigest()
    return f"{kind}:{token}"
